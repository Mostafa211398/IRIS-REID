# IRIS Pipeline

A standalone local application for plate detection, crop enhancement, Arabic PP-OCRv5 recognition, review, and exports. All application code, model assets, data, and runtime environments live in this folder. The parent IRIS application is not imported or used at runtime.

For a detailed explanation of the architecture, processing algorithms, video sampling, Arabic OCR, saved data, job controls, API, and exports, read [How the full pipeline works](docs/how-the-pipeline-works.md).

## Run on this computer

The environments and production frontend have been installed during implementation. From this folder:

```powershell
./scripts/start.ps1
```

Open **http://127.0.0.1:8014**. The backend serves the built React frontend. Stop it with Ctrl+C in the launch terminal. Processing and corrections persist in `.data/pipeline.sqlite3`.

For a hidden background process, use `./scripts/start.ps1 -Background` and stop it with `./scripts/stop.ps1`. Stop the current background instance before launching another instance on the same port. Background stop ends the application and its OCR worker; unfinished jobs remain retryable.

Launch scripts check the configured port before loading models. If it is already occupied by IRIS Pipeline, open the existing instance or stop it before starting again. For a foreground instance, press Ctrl+C in its terminal; for a background instance, use `./scripts/stop.ps1`.

## Install on another Windows computer

Install 64-bit Python 3.11 with the `py` launcher, Node.js, and a working NVIDIA driver for CUDA inference. Copy this entire application folder, including `best.pt`, `plate_enhancement.py`, and `v5_arabic_mobile-fine-tuned.zip`. Runtime environments are machine-specific: recreate them using setup rather than copying `.runtime` between computers.

```powershell
./scripts/setup.ps1
./scripts/start.ps1
```

Setup creates independent `.runtime/backend` and `.runtime/paddle` environments, installs CUDA 11.8 PyTorch and Paddle on NVIDIA computers, builds the frontend, and verifies real model execution. Downloads and unpacking can require more than 15 GB of free space. Keep additional space for uploaded media, saved source frames, crops, and exports. Use `-Python C:/path/to/python.exe` if the Python launcher is unavailable. `-Device cpu` installs CPU packages for a computer without CUDA.

Defaults are copied from `.env.example` into `.env`. Backend port `IRIS_PORT=8014` and development frontend port `IRIS_FRONTEND_PORT=5187` are independently configurable. Device defaults to automatic CUDA discovery; `IRIS_DEVICE=cpu` or `cuda:1` explicitly selects a device. Arabic annotations use Windows Arial by default; `IRIS_ANNOTATION_FONT` can select another Arabic-capable font.

For frontend development, run `./scripts/start-dev.ps1` and open **http://127.0.0.1:5187**.

## Processing and review

- **Analyze:** Upload files or folders, or enter backend-accessible local paths. Folder uploads preserve source names. Local scanning optionally includes subfolders. Supported formats include PNG/JPEG/BMP/WebP/TIFF and MP4/AVI/MOV/MKV/M4V/WMV/FLV/WebM.
- **Stages:** Detection only, detection plus enhancement, or full OCR. Plate-crop mode accepts images and starts at enhancement. Videos default to five samples per second using source timestamps. Every sampled detection is retained, including repeated sightings and multiple plates. Invalid bounds are excluded; poor quality is flagged and saved.
- **Review:** Compare lossless original and enhanced crops, inspect source frames, and view detector confidence, crop quality, and OCR confidence separately. Filter and paginate results. Corrections preserve raw predictions and append an audit entry.
- **History:** Reopen jobs, cancel active work, retry failed or cancelled work, and download artifacts. Interrupted jobs become failed and can be retried after restart. Reruns use saved crops. Enhancement reruns use originals and preserve previous enhanced files; OCR reruns use the current enhanced crop.
- **Pipeline / Settings:** Inspect verified model/device readiness, edit defaults, and save camera presets. One inference job is active at a time, with batch size one and a persistent Paddle worker.
- **Accuracy:** Compare original and enhanced OCR on the same labeled crops. See `docs/benchmark.md` for label order and scoring.

Jobs snapshot inputs into their own media storage, so completed jobs remain reviewable if the original input is moved. Results, enhancement configurations, model fingerprints, predictions, corrections, and run timings are saved. Cancellation is cooperative between frames/crops and waits for an in-flight inference call to finish. An interrupted detection pass is retried with stable detection identifiers to avoid duplicate records. Exports are available after a job stops processing.

## Exports

- **CSV:** One row per detection, including raw/corrected text, Arabic parts, independent confidence values, quality metrics, bounding box, frame, timestamp, and review flags.
- **JSON:** Job settings, source identities, all detections, run/model metadata, and prediction/correction/enhancement histories.
- **Crop ZIP:** Original crops, every saved enhancement version, a JSON-lines manifest, and job/run metadata.
- **Annotated ZIP:** Source images and videos with plate boxes and shaped Arabic labels. Video frame timestamps are preserved, including variable frame rates. Audio streams are remuxed where compatible with MP4. Each sampled box is displayed until its sampling interval expires; labels do not imply detections on unsampled frames. Sources have distinct IDs, so repeated filenames cannot overwrite each other.

The current client loads completed exports as a browser download. Very large exports can use considerable browser memory; production deployment for long recordings should use streamed downloads and a retention policy.

## Verification

```powershell
./scripts/diagnostics.ps1
./scripts/verify.ps1
# With the application running:
cd frontend
npm.cmd run test:e2e
```

Diagnostics verify actual CUDA matrix operations, YOLO inference, reference enhancement, and static PP-OCR inference. Backend tests cover source scanning/uploads, multiple detections per frame, timestamp sampling, poor-crop persistence, enhancement parity, retries/cancellation, versioned reruns, correction audit, filters, accuracy calculations, traversal protection, exports, and constant/variable video timestamps. Integration tests use a controlled detector/OCR implementation; `scripts/smoke-real.py` exercises the supplied models through the live API on real plate crops and a generated video.

On this 8 GB RAM computer, stop the background application before running backend tests or a frontend build, and run those checks sequentially. Restart the application for browser and real-inference checks. The default inference queue already limits processing to one job at a time.

See `docs/implementation-status.md` for evidence and remaining deployment verification.
