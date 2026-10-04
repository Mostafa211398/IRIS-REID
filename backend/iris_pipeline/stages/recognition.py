import hashlib
import json
import os
import queue
import subprocess
import threading
import unicodedata
import zipfile
from pathlib import Path
import yaml


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parts(text):
    digits = "".join(str(unicodedata.digit(c)) for c in text if c.isdecimal())
    visual_letters = "".join(c for c in text if "\u0621" <= c <= "\u064a")
    # Archive labels use left-to-right visual order; store logical Arabic separately.
    letters = visual_letters[::-1]
    return {"digits": digits, "arabic_letters": letters, "visual_letters": visual_letters,
            "complete": 1 <= len(digits) <= 4 and 1 <= len(letters) <= 3}


class Recognizer:
    def __init__(self, settings, device):
        self.settings, self.device = settings, device
        self.process = None
        self.lock = threading.RLock()
        self.archive_hash = fingerprint(settings.root / "v5_arabic_mobile-fine-tuned.zip")
        self.model_dir = settings.data / "models" / self.archive_hash[:16]
        required = ["inference/inference.json", "inference/inference.pdiparams", "inference/inference.yml", "plate_dict.txt"]
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(settings.root / "v5_arabic_mobile-fine-tuned.zip") as archive:
            # Extract only named inference assets; never extract arbitrary ZIP paths.
            for name in required:
                target = self.model_dir / Path(name).name
                expected = hashlib.sha256(archive.read(name)).hexdigest()
                if not target.is_file() or fingerprint(target) != expected:
                    target.write_bytes(archive.read(name))
        meta = yaml.safe_load((self.model_dir / "inference.yml").read_text(encoding="utf-8"))
        dictionary = (self.model_dir / "plate_dict.txt").read_text(encoding="utf-8-sig").splitlines()
        if meta["PostProcess"]["character_dict"] != dictionary:
            raise ValueError("OCR export dictionary mismatch")

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        if self.process is not None:
            self.close()
        if not self.settings.paddle_python.is_file():
            raise RuntimeError("Paddle runtime missing. Run scripts/setup.ps1")
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        worker = Path(__file__).parents[1] / "workers" / "paddle_worker.py"
        self.log = (self.settings.root / ".runtime" / "ocr-worker.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen([str(self.settings.paddle_python), str(worker)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8", env=env,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.responses = queue.Queue()
        process, responses = self.process, self.responses
        def reader():
            for line in process.stdout:
                try:
                    responses.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
            responses.put({"error": "OCR worker exited. See .runtime/ocr-worker.log"})
        threading.Thread(target=reader, daemon=True).start()

    def request(self, action, **kwargs):
        with self.lock:
            self.start()
            try:
                self.process.stdin.write(json.dumps({"action": action, "model_dir": str(self.model_dir), "device": self.device, **kwargs}) + "\n")
                self.process.stdin.flush()
                result = self.responses.get(timeout=self.settings.ocr_timeout)
            except (queue.Empty, BrokenPipeError, OSError) as exc:
                self.close()
                raise RuntimeError("OCR worker timed out or exited; inspect .runtime/ocr-worker.log") from exc
            if "error" in result:
                raise RuntimeError(result["error"])
            return result

    def read(self, path):
        result = self.request("recognize", path=str(path))
        return {"raw_text": result["text"], "ocr_confidence": result["confidence"], **parts(result["text"])}

    def close(self):
        with self.lock:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                self.process.stdin.close()
                self.process.stdout.close()
                self.process = None
                self.log.close()
