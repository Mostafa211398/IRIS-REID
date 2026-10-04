import importlib
import threading
import time
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from iris_pipeline.config import ROOT, Settings
from iris_pipeline.media import write_png
from iris_pipeline.stages.detection import quality
from iris_pipeline.stages.enhancement import Enhancer


class FakeRecognizer:
    archive_hash = "test-model-fingerprint"
    fail = False

    def read(self, path):
        if self.fail:
            self.fail = False
            raise RuntimeError("Injected OCR failure")
        return {"raw_text": "123بسم", "ocr_confidence": .95, "digits": "123", "arabic_letters": "مسب",
                "visual_letters": "بسم", "complete": True}


class FakeRuntime:
    def __init__(self, settings):
        self.device = "cpu"
        self.status = {"state": "ready", "device": "cpu", "stages": {}}
        self.recognizer = FakeRecognizer()
        self.enhancer = Enhancer(ROOT, "cpu")
        self.detector = self
        self.calls = 0
        self.wait = None

    def diagnostics(self):
        return self.status

    def ensure(self, stage):
        return {"ocr": self.recognizer, "enhance": self.enhancer, "detect": self}[stage]

    def detect(self, image, options):
        self.calls += 1
        if self.wait:
            self.wait.wait(5)
        for box in [[0, 0, 20, 12], [20, 0, 40, 12]]:
            x1,y1,x2,y2 = box
            crop = image[y1:y2,x1:x2].copy()
            yield box, .9, crop, quality(crop, .9, "cpu")

    def close(self):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    module = importlib.import_module("iris_pipeline.app")
    monkeypatch.setattr(module, "Runtime", FakeRuntime)
    app = module.create_app(Settings(root=tmp_path, device="cpu"))
    with TestClient(app) as client:
        yield client


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "plate.png"
    write_png(path, np.zeros((30, 100, 3), dtype=np.uint8))
    return path


def finished(client, job_id):
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"failed", "completed", "cancelled"}:
            return job
        time.sleep(.02)
    raise AssertionError("Job did not finish")
