import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4, uuid5, NAMESPACE_URL
from .accuracy import Accumulator, normalize
from .db import now
from .media import frames, read_image, scan, write_png
from .schemas import JobOptions
from .stages.detection import quality


class Cancelled(Exception):
    pass


class JobManager:
    def __init__(self, settings, db, runtime):
        self.settings, self.db, self.runtime = settings, db, runtime
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="iris-inference")
        self.lock = threading.RLock()
        # Interrupted work is explicit and retryable after an application restart.
        for job in db.jobs(limit=100000)["items"]:
            if job["status"] in {"queued", "running", "cancelling"}:
                db.update_job(job["id"], status="failed", error="Application stopped during processing; retry to resume saved work.")
        self.executor.submit(runtime.diagnostics)

    def path(self, job_id, relative=""):
        return self.settings.data / "jobs" / job_id / relative

    def create(self, paths, options, recursive=False, uploaded=None, truths=None):
        files = scan(paths, recursive, options.mode)
        job_id = uuid4().hex
        job = {"id": job_id, "created": now(), "updated": now(), "status": "queued", "stage": "queued",
               "options": options.model_dump(), "input_options": options.model_dump(), "sources": [], "counts": {"detections": 0}, "progress": {},
               "error": None, "cancel_requested": False, "detection_complete": False,
               "kind": "benchmark" if truths is not None else "analysis"}
        for index, file in enumerate(files):
            source_id = f"s{index:05d}"
            source = {"id": source_id, "name": (uploaded or {}).get(str(file), str(file)),
                      "input_path": str(file), "stored": f"media/{source_id}{file.suffix.lower()}", "uploaded": uploaded is not None}
            if truths is not None:
                source["truth"] = truths[str(file)]
            job["sources"].append(source)
        self.path(job_id).mkdir(parents=True)
        self.db.save_job(job)
        self.submit(job_id, "detect", options)
        return job

    def submit(self, job_id, stage, options):
        run = {"id": uuid4().hex, "job_id": job_id, "stage": stage, "options": options.model_dump(),
               "created": now(), "status": "queued", "timings": {}, "models": self.runtime.status.get("stages", {})}
        self.db.save_run(run)
        self.db.update_job(job_id, active_run=run["id"], status="queued", stage="queued", error=None,
                           cancel_requested=False, options=options.model_dump())
        self.executor.submit(self.execute, job_id, run, options)

    def rerun(self, job_id, stage, options=None):
        with self.lock:
            job = self.db.job(job_id)
            if job["status"] in {"running", "queued", "cancelling"}:
                raise ValueError("Job already active")
            options = options or JobOptions(**job["options"])
            if stage != "detect":
                if self.db.detections(job_id, limit=1)["total"] == 0:
                    raise ValueError("No saved crops to rerun")
                if stage == "ocr" and any(not d.get("enhanced") for d in self.db.iter_detections(job_id)):
                    raise ValueError("Enhance the saved crops before running OCR")
            if stage == "enhance" and options.stop_after == "detect":
                options = options.model_copy(update={"stop_after": "enhance"})
            if stage == "ocr":
                options = options.model_copy(update={"stop_after": "ocr"})
            self.submit(job_id, stage, options)
            return self.db.job(job_id)

    def cancel(self, job_id):
        with self.lock, self.db.lock:
            job = self.db.job(job_id)
            if job["status"] not in {"running", "queued", "cancelling"}:
                raise ValueError("Job is not active")
            return self.db.update_job(job_id, cancel_requested=True, status="cancelling")

    def check(self, job_id):
        if self.db.job(job_id)["cancel_requested"]:
            raise Cancelled()

    def progress(self, job_id, stage, done, total=None, **extra):
        self.check(job_id)
        self.db.update_job(job_id, stage=stage, progress={"done": done, "total": total, **extra})

    def execute(self, job_id, run, options):
        started = time.perf_counter()
        try:
            self.check(job_id)
            self.db.update_job(job_id, status="running", started=now())
            run.update(status="running", models=self.runtime.status.get("stages", {}))
            self.db.save_run(run)
            stages = []
            if run["stage"] == "detect":
                if not self.db.job(job_id)["detection_complete"]:
                    stages.append(("detect", self.detect))
                if options.stop_after in {"enhance", "ocr"}:
                    stages.append(("enhance", self.enhance))
                if options.stop_after == "ocr":
                    stages.append(("ocr", self.recognize))
            elif run["stage"] == "enhance":
                stages.append(("enhance", self.enhance))
                if options.stop_after == "ocr":
                    stages.append(("ocr", self.recognize))
            else:
                stages.append(("ocr", self.recognize))
            for stage, function in stages:
                self.check(job_id)
                begin = time.perf_counter()
                function(job_id, run, options)
                run["timings"][stage] = time.perf_counter()-begin
                self.db.save_run(run)
            self.check(job_id)
            self.db.update_job(job_id, status="completed", stage="complete", finished=now())
            run["status"] = "completed"
        except Cancelled:
            self.db.update_job(job_id, status="cancelled", finished=now())
            run["status"] = "cancelled"
        except Exception as exc:
            self.db.update_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}", finished=now())
            run.update(status="failed", error=str(exc))
        finally:
            run["duration"] = time.perf_counter()-started
            run["finished"] = now()
            count = self.db.detections(job_id, limit=1)["total"]
            run["detections_per_second"] = count / max(run["duration"], .001)
            self.db.update_job(job_id, counts={"detections": count})
            self.db.save_run(run)

    def detect(self, job_id, run, options):
        sources = self.db.job(job_id)["sources"]
        detector = self.runtime.ensure("detect") if options.mode == "vehicle" else None
        count = self.db.detections(job_id, limit=1)["total"]
        for source_index, source in enumerate(sources):
            self.check(job_id)
            destination = self.path(job_id, source["stored"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                temporary = destination.with_suffix(destination.suffix + ".partial")
                with Path(source["input_path"]).open("rb") as src, temporary.open("wb") as dst:
                    while block := src.read(4*1024*1024):
                        self.check(job_id)
                        dst.write(block)
                temporary.replace(destination)
                if source.get("uploaded"):
                    uploaded_path = Path(source["input_path"])
                    uploaded_path.unlink(missing_ok=True)
                    if not any(uploaded_path.parent.iterdir()):
                        uploaded_path.parent.rmdir()
            sampled = 0
            for index, timestamp, image in frames(destination, options.sample_fps,
                                                   lambda: self.db.job(job_id)["cancel_requested"]):
                self.check(job_id)
                results = detector.detect(image, options) if detector else [([0, 0, image.shape[1], image.shape[0]], None, image, quality(image, None, self.runtime.device))]
                frame_saved = False
                for ordinal, (bbox, confidence, crop, measurements) in enumerate(results):
                    self.check(job_id)
                    detection_id = uuid5(NAMESPACE_URL, f"{job_id}:{source['id']}:{index}:{ordinal}").hex
                    original = f"crops/{detection_id}.png"
                    frame = f"frames/{source['id']}_{index:09d}.png"
                    write_png(self.path(job_id, original), crop)
                    if not frame_saved:
                        write_png(self.path(job_id, frame), image)
                        frame_saved = True
                    flags = [*measurements["flags"]]
                    if measurements["quality_score"] < options.review_quality:
                        flags.append("low crop quality")
                    item = {"id": detection_id, "job_id": job_id, "source_id": source["id"], "source_name": source["name"],
                            "frame_index": index, "timestamp": timestamp, "bbox": bbox, "detector_confidence": confidence,
                            "quality": measurements, "original": original, "source_frame": frame, "enhanced": None,
                            "raw_text": None, "ocr_confidence": None, "review_flags": flags, "needs_review": True,
                            "created": now(), "detection_run": run["id"], "truth": source.get("truth")}
                    self.db.save_detection(item)
                    count += 1
                sampled += 1
                self.progress(job_id, "detect", source_index, len(sources), source=source["name"], sampled_frames=sampled, detections=count)
                self.db.update_job(job_id, counts={"detections": self.db.detections(job_id, limit=1)["total"]})
        self.db.update_job(job_id, detection_complete=True)

    def enhance(self, job_id, run, options):
        enhancer = self.runtime.ensure("enhance")
        total = self.db.detections(job_id, limit=1)["total"]
        for index, item in enumerate(self.db.iter_detections(job_id), 1):
            self.check(job_id)
            image = read_image(self.path(job_id, item["original"]))
            enhanced, metadata = enhancer.enhance(image, options)
            relative = f"enhanced/{run['id']}/{item['id']}.png"
            write_png(self.path(job_id, relative), enhanced)
            metadata.update(path=relative, run_id=run["id"], created=now())
            self.db.record("enhancements", item["id"], run["id"], metadata)
            item.update(enhanced=relative, enhancement=metadata, raw_text=None, ocr_confidence=None,
                        digits="", arabic_letters="", needs_review=not item.get("reviewed", False))
            self.db.save_detection(item)
            self.progress(job_id, "enhance", index, total)

    def recognize(self, job_id, run, options):
        recognizer = self.runtime.ensure("ocr")
        total = self.db.detections(job_id, limit=1)["total"]
        benchmark = self.db.job(job_id)["kind"] == "benchmark"
        original_metrics, enhanced_metrics = Accumulator(), Accumulator()
        for index, item in enumerate(self.db.iter_detections(job_id), 1):
            self.check(job_id)
            if not item.get("enhanced"):
                raise ValueError("Enhanced crop missing")
            prediction = recognizer.read(self.path(job_id, item["enhanced"]))
            prediction.update(run_id=run["id"], created=now(), input=item["enhanced"], model_sha256=recognizer.archive_hash)
            self.db.record("predictions", item["id"], run["id"], prediction)
            flags = list(item["quality"]["flags"])
            if item["quality"]["quality_score"] < options.review_quality:
                flags.append("low crop quality")
            if prediction["ocr_confidence"] < options.review_confidence:
                flags.append("low OCR confidence")
            if not prediction["complete"]:
                flags.append("incomplete reading")
            item.update(**{k: prediction[k] for k in ("raw_text", "ocr_confidence", "digits", "arabic_letters", "visual_letters")},
                        prediction_run=run["id"], review_flags=flags, needs_review=bool(flags) and not item.get("reviewed", False))
            if benchmark:
                self.check(job_id)
                raw = recognizer.read(self.path(job_id, item["original"]))
                raw.update(run_id=run["id"], created=now(), input=item["original"], model_sha256=recognizer.archive_hash)
                self.db.record("predictions", item["id"], run["id"], raw)
                matched = normalize(item["truth"]) == normalize(prediction["raw_text"])
                sample = {"id": item["id"], "truth": item["truth"], "original": raw, "enhanced": prediction,
                          "original_match": normalize(item["truth"]) == normalize(raw["raw_text"]), "enhanced_match": matched}
                if not matched:
                    item["review_flags"].append("ground-truth mismatch")
                    item["needs_review"] = not item.get("reviewed", False)
                original_metrics.add(item["truth"], raw["raw_text"])
                enhanced_metrics.add(item["truth"], prediction["raw_text"])
                item["comparison"] = sample
            self.db.save_detection(item)
            self.progress(job_id, "ocr", index, total)
        if benchmark:
            self.db.update_job(job_id, benchmark={"original": original_metrics.result(), "enhanced": enhanced_metrics.result(),
                                                  "run_id": run["id"], "label_order": "visual_left_to_right"})

    def close(self):
        for job in self.db.jobs(limit=100000)["items"]:
            if job["status"] in {"queued", "running", "cancelling"}:
                self.db.update_job(job["id"], cancel_requested=True)
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.runtime.close()
