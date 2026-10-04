"""GPU-assisted license-plate crop extractor, optimized for OCR legibility.

YOLO detection, plate-region quality measurements, and image enlargement run on CUDA.
ByteTrack bookkeeping, OpenCV video decoding, and image encoding/writing use CPU.
No enhancement can reconstruct detail missing from the source video.

Install: python -m pip install ultralytics opencv-python numpy
Install CUDA PyTorch for your GPU: https://pytorch.org/get-started/locally/

Example:
  python extract_license_plates_gpu_ocr.py --model "best(1).pt" --input videos \
      --output detected_plates --device 0

The filters are heuristics, NOT an OCR model or a guarantee of readable text.
Use --save-rejected-preview to inspect samples that failed quality thresholds.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}


def parse_args():
    p = argparse.ArgumentParser(description="CUDA YOLO + OCR-oriented plate crop selection")
    p.add_argument("--model", default="best.pt", help="YOLO .pt model path")
    p.add_argument("--input", default="videos", help="Folder containing videos")
    p.add_argument("--output", default="detected_plates", help="Output folder")
    p.add_argument("--device", default="0", help="CUDA GPU index, e.g. 0")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--padding", type=int, default=1, help="Extra original-frame pixels around detection")
    p.add_argument("--min-width", type=int, default=80, help="Min detector box width before padding")
    p.add_argument("--min-height", type=int, default=24, help="Min detector box height before padding")
    p.add_argument("--min-relative-area", type=float, default=0.00025)
    p.add_argument("--min-aspect", type=float, default=1.10)
    p.add_argument("--max-aspect", type=float, default=8.0)
    p.add_argument("--min-brightness", type=float, default=25.0)
    p.add_argument("--max-brightness", type=float, default=235.0)
    # OCR-oriented measurements on the inner, LOWER portion of detector box.
    # This excludes much of the surrounding vehicle and upper plate band.
    p.add_argument("--roi-left", type=float, default=0.08)
    p.add_argument("--roi-right", type=float, default=0.92)
    p.add_argument("--roi-top", type=float, default=0.34)
    p.add_argument("--roi-bottom", type=float, default=0.92)
    p.add_argument("--min-char-sharpness", "--min-sharpness", dest="min_char_sharpness",
                   type=float, default=17.0, help="Laplacian variance in likely text region")
    p.add_argument("--min-char-contrast", "--min-contrast", dest="min_char_contrast",
                   type=float, default=16.0, help="90th minus 10th percentile in text region")
    p.add_argument("--min-edge-strength", type=float, default=6.0,
                   help="Mean absolute horizontal gradient within text region")
    p.add_argument("--min-edge-density", type=float, default=0.035,
                   help="Fraction of text region with horizontal gradient > 18")
    p.add_argument("--min-quality-score", type=float, default=0.36)
    p.add_argument("--best-per-track", type=int, default=3)
    p.add_argument("--min-frame-gap", type=int, default=4,
                   help="Avoid saving several nearly identical adjacent frames")
    p.add_argument("--upscale", type=int, default=3)
    p.add_argument("--save-rejected-preview", type=int, default=0,
                   help="Save up to this many rejected crops per video for diagnostics")
    p.add_argument("--recursive", action="store_true")
    return p.parse_args()


def validate(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install CUDA-enabled PyTorch and verify torch.cuda.is_available().")
    try:
        index = int(args.device)
    except ValueError as ex:
        raise ValueError("This script requires --device 0 (or another CUDA GPU index).") from ex
    if not (0 <= index < torch.cuda.device_count()):
        raise ValueError(f"GPU {index} unavailable; found {torch.cuda.device_count()} CUDA GPUs")
    args.cuda_device = torch.device(f"cuda:{index}")
    if args.frame_skip < 0 or args.padding < 0 or args.min_frame_gap < 0:
        raise ValueError("frame-skip, padding and min-frame-gap must be nonnegative")
    if min(args.min_width, args.min_height, args.best_per_track, args.upscale, args.imgsz) < 1:
        raise ValueError("min-width/min-height/best-per-track/upscale/imgsz must be positive")
    if not 0 <= args.min_quality_score <= 1:
        raise ValueError("min-quality-score must be between 0 and 1")
    if not 0 <= args.roi_left < args.roi_right <= 1 or not 0 <= args.roi_top < args.roi_bottom <= 1:
        raise ValueError("ROI fractions must be ordered and within [0, 1]")


def find_videos(folder, recursive):
    entries = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in entries if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def clamp(v, low, high):
    return max(low, min(v, high))


def grayscale(bgr):
    x = bgr.permute(2, 0, 1).unsqueeze(0).float()
    return x[:, 0:1] * 0.114 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.299


@torch.inference_mode()
def ocr_metrics_gpu(plate_gpu, args):
    """GPU metrics on detector box WITHOUT padding. Plate is [H,W,3] uint8 BGR.

    Text ROI intentionally favors the lower center of Egyptian-style plates;
    adjust ROI CLI flags when plate layout differs. Sobel-x approximates
    vertical strokes, not a guarantee that the edges are actual characters.
    """
    h, w = plate_gpu.shape[:2]
    x1 = clamp(int(round(args.roi_left * w)), 0, w - 1)
    x2 = clamp(int(round(args.roi_right * w)), x1 + 1, w)
    y1 = clamp(int(round(args.roi_top * h)), 0, h - 1)
    y2 = clamp(int(round(args.roi_bottom * h)), y1 + 1, h)
    g = grayscale(plate_gpu)
    region = g[:, :, y1:y2, x1:x2]
    lum = g.mean()
    # Normalize using simple robust percentile range; far less influenced by one edge.
    lo, hi = torch.quantile(region.reshape(-1), torch.tensor([0.10, 0.90], device=region.device))
    contrast = hi - lo
    if region.shape[-2] >= 3 and region.shape[-1] >= 3:
        lap_k = region.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        lap = F.conv2d(F.pad(region, (1, 1, 1, 1), mode="replicate"), lap_k)
        sharpness = lap.var(unbiased=False)
        sobel = region.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
        gx = F.conv2d(F.pad(region, (1, 1, 1, 1), mode="replicate"), sobel).abs()
        edge_strength = gx.mean()
        edge_density = (gx > 18.0).float().mean()
    else:
        sharpness = region.new_zeros(())
        edge_strength = region.new_zeros(())
        edge_density = region.new_zeros(())
    vals = torch.stack((lum, contrast, sharpness, edge_strength, edge_density))
    return tuple(map(float, vals.cpu().tolist()))


def quality_score(conf, w, h, contrast, sharpness, edge_strength, edge_density):
    """OCR-oriented ranking: detector confidence intentionally low-weight."""
    size_score = 0.65 * min(w / 155., 1.) + 0.35 * min(h / 48., 1.)
    return float(
        0.08 * conf + 0.20 * size_score
        + 0.22 * min(sharpness / 110., 1.)
        + 0.20 * min(contrast / 75., 1.)
        + 0.20 * min(edge_strength / 20., 1.)
        + 0.10 * min(edge_density / 0.18, 1.)
    )


@torch.inference_mode()
def enhance_for_ocr_gpu(original_bgr, item, upscale, device):
    """Bicubic GPU enlargement; subtle enhancement only for already-clear text.

    Blurry crops are only enlarged. Never sharpen them into invented strokes.
    """
    a = torch.from_numpy(np.ascontiguousarray(original_bgr)).to(device)
    im = a.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    if upscale > 1:
        im = F.interpolate(im, scale_factor=upscale, mode="bicubic", align_corners=False).clamp(0, 1)
    if item["char_sharpness"] >= 55 and item["edge_strength"] >= 11:
        gray = im[:, 0:1] * 0.114 + im[:, 1:2] * 0.587 + im[:, 2:3] * 0.299
        local = F.avg_pool2d(F.pad(gray, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
        im = (im + 0.07 * (gray - local)).clamp(0, 1)
        k = im.new_tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
        blur = F.conv2d(F.pad(im, (1, 1, 1, 1), mode="replicate"),
                        k.view(1, 1, 3, 3).repeat(3, 1, 1, 1), groups=3)
        im = (im + 0.12 * (im - blur)).clamp(0, 1)
    return (im.squeeze(0).permute(1, 2, 0) * 255.).round().byte().cpu().numpy()


def accept_candidate(by_track, track_id, item, count, min_gap):
    """Keep up to N top candidates; replace worse near-duplicate frames."""
    selected = by_track.setdefault(track_id, [])
    nearby = [s for s in selected if abs(s["frame"] - item["frame"]) < min_gap]
    if nearby:
        if any(s["score"] >= item["score"] for s in nearby):
            return False
        for s in nearby:
            selected.remove(s)
    elif len(selected) >= count:
        worst = min(selected, key=lambda x: x["score"])
        if item["score"] <= worst["score"]:
            return False
        selected.remove(worst)
    selected.append(item)
    return True


def write_image(path, image):
    # PNG lossless preserves every pixel in the original crop.
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Failed to save {path}")


def save_candidates(out, by_track, args):
    raw = out / "raw"
    enhanced = out / "enhanced"
    raw.mkdir(parents=True, exist_ok=True)
    enhanced.mkdir(parents=True, exist_ok=True)
    rows = []
    for tid in sorted(by_track):
        for rank, item in enumerate(sorted(by_track[tid], key=lambda v: v["score"], reverse=True), 1):
            stem = f"trk_{tid:05d}_k{rank}_f{item['frame']:06d}_q{item['score']:.2f}_c{item['confidence']:.2f}"
            raw_path = raw / (stem + ".png")
            enh_path = enhanced / (stem + "_enh.png")
            write_image(raw_path, item["crop"])
            improved = enhance_for_ocr_gpu(item["crop"], item, args.upscale, args.cuda_device)
            write_image(enh_path, improved)
            rows.append({k: v for k, v in item.items() if k != "crop"} | {
                "track_id": tid, "rank": rank, "raw_file": str(raw_path), "enhanced_file": str(enh_path)
            })
    csv_path = out / "crop_metrics.csv"
    columns = ["track_id", "rank", "frame", "score", "confidence", "width", "height",
               "brightness", "char_contrast", "char_sharpness", "edge_strength", "edge_density",
               "raw_file", "enhanced_file"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def process_video(model, path, output_root, args):
    # Separate by video filename including extension to avoid .mp4/.avi collisions.
    out = output_root / f"{path.stem}_{path.suffix.lstrip('.').lower()}"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {path}")
        return 0
    model.predictor = None  # reset tracking state between videos
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    by_track = {}
    counts = {"small": 0, "quality": 0, "accepted": 0, "untracked": 0}
    rejected_saved = 0
    rejected_dir = out / "rejected_preview"
    fallback_id = 1_000_000
    frame_num = 0
    print(f"\nProcessing {path.name} ({total} frames)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_num % (args.frame_skip + 1):
                frame_num += 1
                continue
            fh, fw = frame.shape[:2]
            frame_area = max(1, fw * fh)
            # CPU OpenCV decodes, then one upload of the decoded frame for GPU cropping.
            frame_gpu = torch.from_numpy(np.ascontiguousarray(frame)).to(args.cuda_device)
            result = model.track(source=frame, persist=True, tracker=args.tracker,
                                 conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                                 device=str(args.device), verbose=False)[0]
            if result.boxes is not None:
                for box in result.boxes:
                    coords = box.xyxy[0].round().int().cpu().tolist()
                    bx1, by1, bx2, by2 = map(int, coords)
                    bx1, bx2 = clamp(bx1, 0, fw), clamp(bx2, 0, fw)
                    by1, by2 = clamp(by1, 0, fh), clamp(by2, 0, fh)
                    pw, ph = bx2 - bx1, by2 - by1
                    if pw < args.min_width or ph < args.min_height or pw * ph / frame_area < args.min_relative_area:
                        counts["small"] += 1
                        continue
                    ratio = pw / ph
                    if not args.min_aspect <= ratio <= args.max_aspect:
                        counts["quality"] += 1
                        continue
                    # Score ONLY the true detector box; no surrounding body/background.
                    plate = frame_gpu[by1:by2, bx1:bx2]
                    if plate.numel() == 0:
                        continue
                    brightness, char_contrast, char_sharpness, edge_strength, edge_density = ocr_metrics_gpu(plate, args)
                    conf = float(box.conf[0].item())
                    score = quality_score(conf, pw, ph, char_contrast, char_sharpness,
                                          edge_strength, edge_density)
                    passes = (args.min_brightness <= brightness <= args.max_brightness
                              and char_contrast >= args.min_char_contrast
                              and char_sharpness >= args.min_char_sharpness
                              and edge_strength >= args.min_edge_strength
                              and edge_density >= args.min_edge_density
                              and score >= args.min_quality_score)
                    if not passes:
                        counts["quality"] += 1
                        if rejected_saved < args.save_rejected_preview:
                            rejected_dir.mkdir(parents=True, exist_ok=True)
                            preview = plate.cpu().numpy().copy()
                            write_image(rejected_dir / f"f{frame_num:06d}_s{score:.2f}.png", preview)
                            rejected_saved += 1
                        continue
                    if box.id is None:
                        # Untracked detections have no reliable track identity.
                        track_id = fallback_id
                        fallback_id += 1
                        counts["untracked"] += 1
                    else:
                        track_id = int(box.id[0].item())
                    heap = by_track.get(track_id, [])
                    # Avoid GPU->CPU image transfers unless selection could improve.
                    near = [v for v in heap if abs(v["frame"] - frame_num) < args.min_frame_gap]
                    if any(v["score"] >= score for v in near):
                        continue
                    if not near and len(heap) >= args.best_per_track and score <= min(v["score"] for v in heap):
                        continue
                    x1, y1 = max(0, bx1 - args.padding), max(0, by1 - args.padding)
                    x2, y2 = min(fw, bx2 + args.padding), min(fh, by2 + args.padding)
                    original_crop = frame_gpu[y1:y2, x1:x2].cpu().numpy().copy()
                    item = {"frame": frame_num, "confidence": conf, "score": score,
                            "width": pw, "height": ph, "brightness": brightness,
                            "char_contrast": char_contrast, "char_sharpness": char_sharpness,
                            "edge_strength": edge_strength, "edge_density": edge_density,
                            "crop": original_crop}
                    if accept_candidate(by_track, track_id, item, args.best_per_track, args.min_frame_gap):
                        counts["accepted"] += 1
            if frame_num % 200 == 0:
                print(f"\r  frame {frame_num}/{total} | tracks: {len(by_track)} | kept updates: {counts['accepted']}",
                      end="", flush=True)
            frame_num += 1
    finally:
        cap.release()
    saved = save_candidates(out, by_track, args)
    print(f"\n  Saved {saved} raw/enhanced pairs from {len(by_track)} tracks. "
          f"Rejected: small={counts['small']}, quality={counts['quality']}. "
          f"Untracked={counts['untracked']}. Metrics: {out / 'crop_metrics.csv'}")
    return saved


def main():
    args = parse_args()
    validate(args)
    model_path, source, dest = Path(args.model), Path(args.input), Path(args.output)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model missing: {model_path}")
    if not source.is_dir():
        raise NotADirectoryError(f"Video input directory missing: {source}")
    videos = find_videos(source, args.recursive)
    if not videos:
        print(f"No videos found in {source}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"CUDA GPU: {torch.cuda.get_device_name(args.cuda_device)}")
    print("Video decode/ByteTrack bookkeeping/image encoding still use CPU.")
    print(f"OCR ROI: x={args.roi_left:.2f}..{args.roi_right:.2f}; "
          f"y={args.roi_top:.2f}..{args.roi_bottom:.2f}")
    model = YOLO(str(model_path))
    total = sum(process_video(model, video, dest, args) for video in videos)
    print(f"\nFinished {len(videos)} videos; saved {total} selected crop pairs in {dest.resolve()}")


if __name__ == "__main__":
    main()
