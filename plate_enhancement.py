"""Batch GPU enhancement for license plate crops.

Preserves the original enhancement algorithm from plate_enhancement(1).py:
3x bicubic enlargement; subtle local contrast and sharpening ONLY when
character-region sharpness >= 55 and edge strength >= 11.

Run:
    python plate_enhancement_batch.py
or override folder paths:
    python plate_enhancement_batch.py --input "E:\\plates\\raw" --output "E:\\plates\\enhanced"

Dependencies: pip install numpy opencv-python torch
Use CUDA-enabled PyTorch to run on an NVIDIA GPU.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# EDIT THESE TWO PATHS for running without command-line options.
INPUT_FOLDER = r"E:\plate-det-model\v2\images"
OUTPUT_FOLDER = r"E:\plate-det-model\v2\images-enhanced"

DEVICE = "cuda:0"  # Change to "cpu" only for CPU testing.
UPSCALE = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def grayscale(bgr_gpu: torch.Tensor) -> torch.Tensor:
    """BGR uint8 [H,W,3] -> grayscale float32 [1,1,H,W] on same device."""
    x = bgr_gpu.permute(2, 0, 1).unsqueeze(0).float()
    return x[:, 0:1] * 0.114 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.299


@torch.inference_mode()
def character_metrics(crop_bgr: np.ndarray, device: str = DEVICE):
    """Calculate original-extractor-style metrics in the likely character ROI.

    ROI: horizontal 8%-92%, vertical 34%-92%. The fixed ROI was designed for
    plates like the original Egyptian plate crops; adapt for another layout.
    """
    plate = torch.from_numpy(np.ascontiguousarray(crop_bgr)).to(device)
    h, w = plate.shape[:2]
    x1, x2 = min(round(.08 * w), w - 1), min(max(round(.92 * w), 1), w)
    y1, y2 = min(round(.34 * h), h - 1), min(max(round(.92 * h), 1), h)
    region = grayscale(plate)[:, :, y1:y2, x1:x2]

    if region.shape[-2] < 3 or region.shape[-1] < 3:
        return 0.0, 0.0

    lap_kernel = region.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    lap = F.conv2d(F.pad(region, (1, 1, 1, 1), mode="replicate"), lap_kernel)
    sharpness = lap.var(unbiased=False)

    sobel_x = region.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(F.pad(region, (1, 1, 1, 1), mode="replicate"), sobel_x).abs()
    edge_strength = gx.mean()
    return float(sharpness.item()), float(edge_strength.item())


@torch.inference_mode()
def enhance_for_ocr_gpu(original_bgr, item, upscale=3, device="cuda:0"):
    """Original enhancement: bicubic enlargement and conditional gentle enhancement."""
    if not isinstance(original_bgr, np.ndarray) or original_bgr.dtype != np.uint8:
        raise TypeError("original_bgr must be a uint8 NumPy array")
    if original_bgr.ndim != 3 or original_bgr.shape[2] != 3 or not original_bgr.size:
        raise ValueError("original_bgr must be a nonempty H x W x 3 BGR image")
    if upscale < 1:
        raise ValueError("upscale must be at least 1")

    a = torch.from_numpy(np.ascontiguousarray(original_bgr)).to(device)
    im = a.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    if upscale > 1:
        im = F.interpolate(im, scale_factor=upscale, mode="bicubic", align_corners=False).clamp(0, 1)
    if item["char_sharpness"] >= 55 and item["edge_strength"] >= 11:
        gray = im[:, 0:1] * 0.114 + im[:, 1:2] * 0.587 + im[:, 2:3] * 0.299
        local = F.avg_pool2d(F.pad(gray, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
        im = (im + 0.07 * (gray - local)).clamp(0, 1)
        k = im.new_tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
        blur = F.conv2d(
            F.pad(im, (1, 1, 1, 1), mode="replicate"),
            k.view(1, 1, 3, 3).repeat(3, 1, 1, 1), groups=3,
        )
        im = (im + 0.12 * (im - blur)).clamp(0, 1)
    return (im.squeeze(0).permute(1, 2, 0) * 255.).round().byte().cpu().numpy()


def enhance_crop(crop_bgr, *, char_sharpness=0.0, edge_strength=0.0,
                 upscale=3, device="cuda:0"):
    """Single-image API: explicit metrics; zero metrics mean upscale only."""
    item = {"char_sharpness": char_sharpness, "edge_strength": edge_strength}
    return enhance_for_ocr_gpu(crop_bgr, item, upscale=upscale, device=device)


def read_image(path: Path):
    # imdecode handles Windows paths that may contain non-ASCII characters.
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def write_png(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def process_folder(input_folder, output_folder, *, device=DEVICE, upscale=UPSCALE,
                   recursive=False, auto_metrics=True):
    source = Path(input_folder).expanduser().resolve()
    target = Path(output_folder).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Input folder not found: {source}")
    if source == target or source in target.parents:
        raise ValueError("Output folder must be outside the input folder to avoid reprocessing")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python environment. Install CUDA-enabled PyTorch.")
    if upscale < 1:
        raise ValueError("upscale must be >= 1")

    files = source.rglob("*") if recursive else source.iterdir()
    paths = sorted(p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    target.mkdir(parents=True, exist_ok=True)
    if not paths:
        print(f"No supported images in {source}")
        return 0

    rows, completed = [], 0
    print(f"Found {len(paths)} images | device={device} | upscale={upscale}x")
    for i, path in enumerate(paths, 1):
        relative = path.relative_to(source)
        destination = target / relative.parent / f"{path.stem}_{path.suffix[1:].lower()}_enh.png"
        try:
            img = read_image(path)
            if img is None:
                raise ValueError("Failed to decode image")
            sharp, edges = character_metrics(img, device) if auto_metrics else (0.0, 0.0)
            result = enhance_crop(
                img, char_sharpness=sharp, edge_strength=edges,
                upscale=upscale, device=device,
            )
            write_png(destination, result)
            enhanced_detail = sharp >= 55 and edges >= 11
            rows.append({"input": str(path), "output": str(destination),
                         "char_sharpness": f"{sharp:.3f}", "edge_strength": f"{edges:.3f}",
                         "contrast_and_sharpening": enhanced_detail, "error": ""})
            completed += 1
            print(f"[{i}/{len(paths)}] {relative} -> {destination.name} "
                  f"({'contrast + sharpening' if enhanced_detail else 'upscale only'})")
        except Exception as exc:
            rows.append({"input": str(path), "output": "", "char_sharpness": "",
                         "edge_strength": "", "contrast_and_sharpening": "", "error": str(exc)})
            print(f"[{i}/{len(paths)}] ERROR {relative}: {exc}")
    with (target / "enhancement_report.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Completed {completed}/{len(paths)} images. Output: {target}")
    return completed


def main():
    parser = argparse.ArgumentParser(description="Enhance every plate crop in a folder")
    parser.add_argument("--input", default=INPUT_FOLDER, help="Folder containing raw crops")
    parser.add_argument("--output", default=OUTPUT_FOLDER, help="Folder for enhanced PNGs")
    parser.add_argument("--device", default=DEVICE, help="cuda:0 (GPU) or cpu")
    parser.add_argument("--upscale", type=int, default=UPSCALE, help="Bicubic scale factor")
    parser.add_argument("--recursive", action="store_true", help="Include input subfolders")
    parser.add_argument("--upscale-only", action="store_true", help="Skip metric-based contrast and sharpening")
    args = parser.parse_args()
    process_folder(args.input, args.output, device=args.device, upscale=args.upscale,
                   recursive=args.recursive, auto_metrics=not args.upscale_only)


if __name__ == "__main__":
    main()
