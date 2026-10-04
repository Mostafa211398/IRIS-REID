import math
import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def quality(crop, confidence, device):
    plate = torch.from_numpy(np.ascontiguousarray(crop)).to(device)
    h, w = plate.shape[:2]
    x1, x2 = min(round(.08*w), w-1), min(max(round(.92*w), 1), w)
    y1, y2 = min(round(.34*h), h-1), min(max(round(.92*h), 1), h)
    x = plate.permute(2, 0, 1).unsqueeze(0).float()
    gray = x[:, :1]*.114 + x[:, 1:2]*.587 + x[:, 2:3]*.299
    region = gray[:, :, y1:y2, x1:x2]
    low, high = torch.quantile(region.reshape(-1), region.new_tensor([.1, .9]))
    sharp = edges = density = region.new_zeros(())
    if min(region.shape[-2:]) >= 3:
        padded = F.pad(region, (1, 1, 1, 1), mode="replicate")
        lap = region.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).reshape(1, 1, 3, 3)
        sharp = F.conv2d(padded, lap).var(unbiased=False)
        sobel = region.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).reshape(1, 1, 3, 3)/8
        gradient = F.conv2d(padded, sobel).abs()
        edges, density = gradient.mean(), (gradient > 18).float().mean()
    brightness, contrast, sharpness, edge_strength, edge_density = map(float, torch.stack((gray.mean(), high-low, sharp, edges, density)).cpu().tolist())
    size = .65*min(w/155, 1) + .35*min(h/48, 1)
    score = .08*(confidence or 0) + .20*size + .22*min(sharpness/110, 1) + .20*min(contrast/75, 1) + .20*min(edge_strength/20, 1) + .10*min(edge_density/.18, 1)
    flags = []
    for bad, label in ((w < 80 or h < 24, "small"), (sharpness < 17, "blurry"),
                       (contrast < 16, "low contrast"), (brightness < 25 or brightness > 235, "exposure"),
                       (edge_strength < 6 or edge_density < .035, "weak character edges")):
        if bad:
            flags.append(label)
    return dict(width=w, height=h, brightness=brightness, contrast=contrast, char_sharpness=sharpness,
                edge_strength=edge_strength, edge_density=edge_density, quality_score=score, flags=flags)


class Detector:
    def __init__(self, path, device):
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        self.device = device

    def detect(self, image, options):
        result = self.model.predict(image, conf=options.confidence, iou=options.iou,
                                    imgsz=options.image_size, device=self.device, verbose=False)[0]
        h, w = image.shape[:2]
        for box in result.boxes:
            raw = box.xyxy[0].cpu().tolist()
            if not all(math.isfinite(v) for v in raw):
                continue
            x1, y1 = max(0, math.floor(raw[0])), max(0, math.floor(raw[1]))
            x2, y2 = min(w, math.ceil(raw[2])), min(h, math.ceil(raw[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            confidence = float(box.conf[0])
            measurements = quality(image[y1:y2, x1:x2], confidence, self.device)
            pad = options.padding
            crop = image[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)].copy()
            yield [x1, y1, x2, y2], confidence, crop, measurements
