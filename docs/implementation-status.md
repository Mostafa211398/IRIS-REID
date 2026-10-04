# Implementation and verification status

Implemented on October 4, 2026 inside `full pipeline/`. This is a working initial application; client-footage acceptance testing remains necessary.

See [How the full pipeline works](how-the-pipeline-works.md) for the detailed architecture, algorithms, data lifecycle, controls, API, and operational guide.

## Implemented

- Independent React/TypeScript/Vite frontend and FastAPI backend, relative-root configuration, SQLite persistence, input snapshots, and Windows setup/start/development/diagnostic/verification scripts.
- Separate detection, enhancement, and recognition modules. Detection retains all valid sampled crops and flags poor quality. Enhancement calls the supplied reference algorithm once and preserves its coefficients and thresholds. Recognition reads the supplied archive's dictionary and preprocessing configuration, performs CTC decoding, and retains both raw visual text and logical Arabic parts.
- Isolated CUDA PyTorch and Paddle environments, automatic device selection, one inference queue, batch size one, persistent OCR worker, and real startup inference diagnostics.
- File/folder uploads, local file/folder paths, recursive scanning, plate-crop mode, stage stopping, saved-crop reruns, retries, cancellation, server-sent progress, correction history, result filters/pagination, and versioned predictions/enhancements.
- Analyze, Review, History, Pipeline / Settings, and Accuracy screens, saved presets, separate confidence/quality values, source-frame inspection, and stage timings.
- CSV/JSON/crop ZIP/annotated media exports, distinct source IDs, shaped Arabic labels, original video timestamps, and same-crop original-versus-enhanced OCR scoring.

## Verified

- GPU: NVIDIA GeForce GTX 1650, compute capability 7.5, approximately 4 GB VRAM. The machine has 8 GB system RAM.
- PyTorch `2.6.0+cu118`: actual CUDA tensor execution, supplied `best.pt` inference, and CUDA reference enhancement.
- Paddle GPU `3.2.0`: actual GPU tensor execution and supplied Arabic PP-OCRv5 mobile static-model inference inside its separate environment.
- Backend: 11 integration/stage tests passed. They cover enhancement parity on both sharp and flat crops, CTC decoding, digit/Arabic normalization, constant/variable frame sampling, multiple/repeated detections, quality retention, uploads/recursive paths/repeated names, reruns, corrections, cancellation, retries, filters, exports, malformed manifests, and preserved annotated video timestamps.
- Both runtime dependency compatibility checks passed.
- Real inference: six copied labeled crops, a constructed image containing two real plate crops, and a two-second 30 FPS video. The detector saved two image detections and 20 video detections across all ten five-FPS sample times. CSV, JSON, crop ZIP, annotated video, unchanged video frame timestamps, and saved-crop OCR rerun passed.
- Browser: two tests passed, covering the five screens, preset editing, mobile width, absence of page script errors, a real crop upload, Arabic results, and correction history.
- Frontend production TypeScript/Vite build and launch-script PowerShell syntax checks passed.

Evidence generated during verification is stored in `.runtime/real-inference-report.json`, `.runtime/verification-media/`, browser screenshots, and browser test output. Verification jobs remain in History. These artifacts are local and excluded from source control. The regression tests use a controlled model substitute; the separate real-inference script tests the supplied model assets.

## Remaining deployment verification and limits

- Run representative **full vehicle photos and actual camera recordings**, including night footage, motion blur, occlusion, and prolonged video. The verified video was constructed from real crops; it does not establish accuracy on client footage.
- Evaluate a representative labeled client dataset. The archive's reported 164-image metrics are not an application acceptance result, and the small verification sample is not an accuracy estimate.
- Exercise long-job storage growth, large exports, and uncommon audio/video formats. Video timestamps were verified for constant and variable rates; audio remux compatibility across codecs still needs representative media.
- Validate setup from scratch on another deployment computer. The application contains no imports from the parent IRIS backend/frontend, but newly copied machines must recreate their Python environments.
- Disk pressure occurred during GPU dependency installation, and generated download/bytecode caches were cleared. Approximately 24.6 GB was available at final verification. Originals, source frames, crop versions, and exports persist on disk, so storage growth still needs testing with long recordings. Frontend export downloads currently buffer the result in browser memory.
- On this 8 GB RAM computer, keep frontend builds/backend verification separate from the running inference application. A parallel verification attempt encountered memory pressure; sequential backend verification passed.
