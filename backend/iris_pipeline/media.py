import math
from pathlib import Path
import cv2
import numpy as np

IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEOS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def scan(paths, recursive, mode):
    found = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        candidates = (path.rglob("*") if recursive else path.iterdir()) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in IMAGES | VIDEOS:
                if mode == "crop" and candidate.suffix.lower() in VIDEOS:
                    raise ValueError("Plate-crop mode accepts images only")
                found[str(candidate.resolve())] = candidate.resolve()
    if not found:
        raise ValueError("No supported images or videos found")
    return list(found.values())


def frames(path, sample_fps, cancelled=lambda: False):
    if Path(path).suffix.lower() in IMAGES:
        yield 0, 0.0, read_image(path)
        return
    import av
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream: {path}")
        origin, target = None, 0.0
        for index, frame in enumerate(container.decode(video=0)):
            if cancelled():
                return
            if frame.time is None:
                raise ValueError(f"Missing source timestamp at frame {index}: {path}")
            if origin is None:
                origin = float(frame.time)
            timestamp = float(frame.time)-origin
            if timestamp + 1e-6 >= target:
                yield index, timestamp, frame.to_ndarray(format="bgr24")
                target = (math.floor((timestamp+1e-6)*sample_fps)+1)/sample_fps
