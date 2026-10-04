import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from uuid import uuid4
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from .config import settings as default_settings
from .db import Database
from .exports import Exporter
from .jobs import JobManager
from .media import IMAGES, VIDEOS
from .runtime import Runtime
from .schemas import BenchmarkRequest, Correction, JobOptions, PathRequest, RerunRequest


def create_app(settings=default_settings):
    settings.prepare()
    db = Database(settings.data / "pipeline.sqlite3")
    runtime = Runtime(settings)

    @asynccontextmanager
    async def lifespan(app):
        manager = JobManager(settings, db, runtime)
        app.state.manager = manager
        app.state.exporter = Exporter(settings, db, manager)
        yield
        await asyncio.to_thread(manager.close)

    app = FastAPI(title="IRIS Pipeline", lifespan=lifespan)
    app.state.db, app.state.runtime = db, runtime
    app.add_middleware(CORSMiddleware, allow_origins=[f"http://127.0.0.1:{settings.frontend_port}", f"http://localhost:{settings.frontend_port}"], allow_methods=["GET", "POST", "PUT"], allow_headers=["Content-Type"])

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Record not found"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/health")
    def health():
        return runtime.status

    @app.post("/api/diagnostics")
    def diagnostics():
        # Queue with inference to avoid concurrent model/GPU access.
        app.state.manager.executor.submit(runtime.diagnostics)
        return {"scheduled": True}

    @app.get("/api/settings")
    def get_settings():
        return {"defaults": db.preference("defaults") or JobOptions().model_dump(), "presets": db.preference("presets") or {}}

    @app.put("/api/settings")
    def save_settings(body: dict):
        defaults = JobOptions(**body.get("defaults", {}))
        presets = {str(name): JobOptions(**options).model_dump() for name, options in body.get("presets", {}).items()}
        db.preference("defaults", defaults.model_dump())
        db.preference("presets", presets)
        return get_settings()

    @app.get("/api/jobs")
    def jobs(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
        return db.jobs(offset, limit)

    @app.post("/api/jobs/paths", status_code=201)
    def from_paths(body: PathRequest):
        return app.state.manager.create(body.paths, body.options, body.recursive)

    @app.post("/api/jobs/upload", status_code=201)
    async def upload(files: list[UploadFile] = File(...), options: str = Form("{}")):
        try:
            parsed = JobOptions.model_validate_json(options)
        except ValidationError as exc:
            raise HTTPException(422, str(exc))
        upload_dir = settings.data / "uploads" / uuid4().hex
        upload_dir.mkdir(parents=True)
        paths, names = [], {}
        try:
            for index, file in enumerate(files):
                name = PurePosixPath((file.filename or "").replace("\\", "/"))
                if name.is_absolute() or ".." in name.parts or name.suffix.lower() not in IMAGES | VIDEOS:
                    raise HTTPException(400, f"Unsupported or unsafe media filename: {file.filename}")
                if parsed.mode == "crop" and name.suffix.lower() in VIDEOS:
                    raise HTTPException(400, "Plate-crop mode accepts images only")
                target = upload_dir / f"{index:05d}{name.suffix.lower()}"
                with target.open("wb") as handle:
                    while chunk := await file.read(1024*1024):
                        handle.write(chunk)
                await file.close()
                paths.append(target)
                names[str(target.resolve())] = str(name)
            return app.state.manager.create(paths, parsed, uploaded=names)
        except Exception:
            import shutil
            shutil.rmtree(upload_dir)
            raise

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return {**db.job(job_id), "runs": db.runs(job_id)}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return app.state.manager.cancel(job_id)

    @app.post("/api/jobs/{job_id}/retry")
    def retry(job_id: str):
        job = db.job(job_id)
        if job["status"] not in {"failed", "cancelled"}:
            raise HTTPException(409, "Only failed or cancelled jobs can be retried")
        previous_runs = db.runs(job_id)
        return app.state.manager.rerun(job_id, previous_runs[-1]["stage"] if previous_runs else "detect")

    @app.post("/api/jobs/{job_id}/rerun")
    def rerun(job_id: str, body: RerunRequest):
        return app.state.manager.rerun(job_id, body.stage, body.options)

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str, request: Request):
        db.job(job_id)
        async def stream():
            previous = None
            while not await request.is_disconnected():
                current = db.job(job_id)
                serialized = json.dumps(current)
                if serialized != previous:
                    yield "data: " + serialized + "\n\n"
                    previous = serialized
                else:
                    yield ": heartbeat\n\n"
                if current["status"] in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(.5)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/detections")
    def detections(job_id: str, offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=200),
                   source: str | None = None, review: bool | None = None, max_confidence: float | None = Query(None, ge=0, le=1)):
        db.job(job_id)
        return db.detections(job_id, offset, limit, source, review, max_confidence)

    @app.get("/api/detections/{detection_id}/history")
    def history(detection_id: str):
        db.detection(detection_id)
        return db.audit(detection_id)

    @app.post("/api/detections/{detection_id}/correction")
    def correct(detection_id: str, body: Correction):
        item = db.detection(detection_id)
        if db.job(item["job_id"])["status"] in {"running", "queued", "cancelling"}:
            raise HTTPException(409, "Wait for processing to finish before correcting results")
        return db.correct(detection_id, body.text, body.note)

    @app.get("/api/jobs/{job_id}/artifacts/{relative:path}")
    def artifact(job_id: str, relative: str):
        db.job(job_id)
        root = app.state.manager.path(job_id).resolve()
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path)

    @app.get("/api/jobs/{job_id}/export/{kind}")
    def export(job_id: str, kind: str):
        target = app.state.exporter.export(job_id, kind)
        return FileResponse(target, filename=f"iris-{job_id[:8]}-{kind}{target.suffix}")

    @app.post("/api/benchmarks", status_code=201)
    def benchmark(body: BenchmarkRequest):
        manifest = Path(body.manifest_path).expanduser().resolve()
        if not manifest.is_file():
            raise ValueError("Benchmark manifest does not exist")
        rows = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("Benchmark manifest must be a nonempty JSON list of {path, truth}")
        paths, truths = [], {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("truth"), str):
                raise ValueError("Benchmark rows must contain string path and truth fields")
            path = (manifest.parent / row["path"]).resolve()
            if path.suffix.lower() not in IMAGES or not path.is_file() or not isinstance(row.get("truth"), str) or not row["truth"].strip():
                raise ValueError("Each benchmark row needs an existing plate image and nonempty truth")
            if str(path) in truths:
                raise ValueError("Duplicate benchmark image")
            paths.append(path)
            truths[str(path)] = row["truth"]
        options = body.options.model_copy(update={"mode": "crop", "stop_after": "ocr"})
        return app.state.manager.create(paths, options, truths=truths)

    frontend = settings.root / "frontend" / "dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
