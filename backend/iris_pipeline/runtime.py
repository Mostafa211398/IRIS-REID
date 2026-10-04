import threading
import numpy as np
import torch
from .schemas import JobOptions
from .stages.detection import Detector
from .stages.enhancement import Enhancer
from .stages.recognition import Recognizer, fingerprint


class Runtime:
    def __init__(self, settings):
        self.settings = settings
        self.device = "cuda:0" if settings.device == "auto" and torch.cuda.is_available() else ("cpu" if settings.device == "auto" else settings.device)
        self.detector = self.enhancer = self.recognizer = None
        self.lock = threading.RLock()
        self.status = {"state": "checking", "device": self.device, "stages": {}, "gpu": None}

    def ensure(self, stage):
        with self.lock:
            if stage == "detect" and self.detector is None:
                self.detector = Detector(self.settings.root / "best.pt", self.device)
            if stage == "enhance" and self.enhancer is None:
                self.enhancer = Enhancer(self.settings.root, self.device)
            if stage == "ocr" and self.recognizer is None:
                self.recognizer = Recognizer(self.settings, self.device)
            return {"detect": self.detector, "enhance": self.enhancer, "ocr": self.recognizer}[stage]

    def diagnostics(self):
        with self.lock:
            status = {"state": "checking", "device": self.device, "torch": torch.__version__, "stages": {}, "gpu": None}
            if self.device.startswith("cuda"):
                try:
                    index = int(self.device.split(":")[-1])
                    prop = torch.cuda.get_device_properties(index)
                    x = torch.ones((32, 32), device=self.device)
                    probe = float((x @ x).mean())
                    assert probe == 32
                    free, total = torch.cuda.mem_get_info(index)
                    status["gpu"] = {"name": prop.name, "vram_bytes": total, "free_bytes": free,
                                     "compute_capability": f"{prop.major}.{prop.minor}", "probe": probe}
                except Exception as exc:
                    status["gpu_error"] = str(exc)
            for stage in ("detect", "enhance", "ocr"):
                try:
                    module = self.ensure(stage)
                    image = np.zeros((48, 160, 3), dtype=np.uint8)
                    if stage == "detect":
                        list(module.detect(image, JobOptions(image_size=320)))
                        result = {"sha256": fingerprint(self.settings.root / "best.pt"), "model_inference": True}
                    elif stage == "enhance":
                        enhanced, _ = module.enhance(image, JobOptions())
                        assert enhanced.shape == (144, 480, 3)
                        result = {"sha256": fingerprint(self.settings.root / "plate_enhancement.py"), "model_inference": True}
                    else:
                        result = {**module.request("health"), "archive_sha256": module.archive_hash}
                    status["stages"][stage] = {"ready": True, "device": self.device, **result}
                except Exception as exc:
                    status["stages"][stage] = {"ready": False, "error": str(exc)}
            status["state"] = "ready" if all(s["ready"] for s in status["stages"].values()) else "attention"
            self.status = status
            return status

    def close(self):
        if self.recognizer:
            self.recognizer.close()
