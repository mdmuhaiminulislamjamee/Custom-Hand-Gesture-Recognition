# Gesture Dashboard: ONNX + USB-NCM Camera

This repository is the complete Windows reference implementation for an eight-command, static hand-gesture system. It contains the browser dashboard, Python/FastAPI inference backend, USB-NCM/JLIP camera receiver, MediaPipe-to-ONNX pipeline, data-collection and model-qualification tools, and the assets needed to port the classifier to iOS.

Current model release: **v18_20**.

> Important: the Windows application is runnable as delivered. The iOS material is an **integration package**, not a finished Xcode project. An iOS developer must still create the app shell, camera or NCM receiver, ONNX Runtime wrapper, decision state machine, and UI. See [Implementing the pipeline on iOS](#implementing-the-pipeline-on-ios).

## Contents

- [Quick start for the person receiving this folder](#quick-start-for-the-person-receiving-this-folder)
- [What the system does](#what-the-system-does)
- [How the implementation works](#how-the-implementation-works)
- [Gesture and model contract](#gesture-and-model-contract)
- [How the 10-FPS application works](#how-the-10-fps-application-works)
- [Repository map](#repository-map)
- [Windows setup](#windows-setup)
- [Running the applications](#running-the-applications)
- [Using the dashboard](#using-the-dashboard)
- [Collecting and retraining data](#collecting-and-retraining-data)
- [Implementing the pipeline on iOS](#implementing-the-pipeline-on-ios)
- [Verification and troubleshooting](#verification-and-troubleshooting)
- [Creating a clean handover package](#creating-a-clean-handover-package)

## Quick start for the person receiving this folder

These are the shortest steps for running the full, strict 10-FPS Windows dashboard.

1. Copy or extract the folder to a writable local directory. Do not run it from inside a ZIP file, a read-only share, or a cloud placeholder.
2. Install the prerequisites listed in [Windows setup](#windows-setup).
3. Open PowerShell in the project root and run:

   ```powershell
   .\setup_ncm_onnx_dashboard.bat
   ```

4. Connect the development-board camera by USB and configure its Windows USB-NCM adapter as `192.168.50.1/30`. The board is expected at `192.168.50.2`.
5. Check the link:

   ```powershell
   .\check_ncm_link.bat
   ```

6. Start both the backend and frontend:

   ```powershell
   .\start_ncm_onnx_dashboard.bat
   ```

7. Open <http://127.0.0.1:3200>, select **Run UDP discovery**, and then select **Connect board camera**.
8. Stop the two managed servers when finished:

   ```powershell
   .\stop_ncm_onnx_dashboard.bat
   ```

Useful local URLs:

| Service | URL |
|---|---|
| Dashboard | <http://127.0.0.1:3200> |
| Interactive API documentation | <http://127.0.0.1:8200/docs> |
| Readiness | <http://127.0.0.1:8200/readyz> |
| Full health and NCM state | <http://127.0.0.1:8200/api/health> |
| Prometheus-format metrics | <http://127.0.0.1:8200/metrics> |
| NCM prediction WebSocket | `ws://127.0.0.1:8200/ws/ncm-live` |

The start script rebuilds the frontend on every launch, starts both processes in the background, opens the browser, and writes PIDs and logs under `.runtime/`.

## What the system does

The application recognizes eight approved hand gestures and maps them to demo actions. It emits `no_gesture` when the image does not contain a usable known command.

```text
Development-board camera
        |
        | JPEG images inside JLIP v1 packets
        v
USB-NCM network: Windows 192.168.50.1/30 -> board 192.168.50.2
        |
        | TCP 5000; optional UDP discovery on 5001
        v
JLIP stream parser and validated latest-JPEG buffer
        |
        v
low-light checks -> MediaPipe 21-point hand landmarks -> landmark stabilization
        |
        v
76-D feature extraction -> eight-output ONNX MLP + known-gesture-mass score
        |
        v
pose geometry + open-set rejection + probability EMA + hold/release/cooldown
        |
        v
prediction/action JSON -> FastAPI WebSocket -> React dashboard
```

The browser never asks for webcam permission in the NCM dashboard. The Python backend owns the board connection and sends predictions to the browser. The separate OpenCV viewer can use either a PC webcam or the NCM camera.

The application is intentionally local-only by default:

- frontend: `127.0.0.1:3200`;
- backend: `127.0.0.1:8200`;
- board link: `192.168.50.1` to `192.168.50.2`;
- no cloud upload is performed by the runtime or collection UI.

The frontend is React/TypeScript (Next-style `app/` source built by Vinext on Vite). The backend is FastAPI/Uvicorn with NumPy, OpenCV, MediaPipe, Pillow, and ONNX Runtime CPU. Exact JavaScript versions are pinned in `package.json`/`pnpm-lock.yaml`; exact Windows runtime Python versions are pinned in `requirements.lock.txt`.

## How the implementation works

### 1. Camera transport

`backend/ncm_camera.py` implements a reconnecting TCP client and an incremental JLIP parser. It binds the client to the configured NCM host address, connects to the board, handles fragmented or coalesced TCP reads, validates packet headers and CRC32, tracks sequence gaps, replies to heartbeats, validates JPEG markers, and retains the newest valid frame.

The protocol is based on `reference/NcmCameraTester.cpp`:

| Setting | Value |
|---|---|
| Windows/iOS host address | `192.168.50.1` |
| Board address | `192.168.50.2` |
| JPEG TCP service | `5000` |
| UDP discovery service | `5001` |
| Discovery payload | ASCII `JL-CAMERA-DISCOVER` |
| JLIP magic/version | `JLIP`, version `1` |
| Header size/order | 24 bytes, big-endian/network byte order |
| Maximum payload | 200 KiB by default |
| Hello ACK | `0x02` |
| Heartbeat | `0x03` |
| Heartbeat ACK constant | `0x04` |
| JPEG | `0x10` |

The supplied reference receiver replies to a heartbeat with packet type `0x03`. A TCP callback is not guaranteed to contain exactly one JLIP packet or one JPEG; every implementation, including iOS, must keep a persistent byte buffer and parse complete packets from it.

### 2. Image preparation and hand detection

`backend/model_runtime.py` decodes the JPEG, applies the configured `0.92` central square ROI for the NCM dashboard, preserves native detail up to a maximum analysis dimension of 960 pixels, and runs adaptive low-light analysis. Very dark/blank frames are rejected before hand detection. Underexposed usable frames receive bounded CLAHE/gamma enhancement without horizontal mirroring.

`backend/hand_detection.py` runs MediaPipe Hand Landmarker in VIDEO mode with monotonic timestamps and bounded IMAGE-mode recovery. It accepts one hand. `backend/face_guard.py` uses the bundled BlazeFace model to reject small false hand candidates overlapping the face/ear region.

Accepted landmarks are checked for clipping, scale, plausibility, and sudden jumps. `RuntimeSession` in `backend/model_runtime.py` keeps a velocity-adaptive EMA per client and reacquires persistent real motion instead of following a single-frame jump.

### 3. Exact 76-D feature vector

`backend/geometry.py` converts 21 normalized `(x, y)` landmarks to the model input. `swift/GestureFeatureExtractor.swift` is the corresponding iOS implementation.

| Feature group | Count |
|---|---:|
| Wrist-centered, rotated, scaled landmark `(x, y)` values | 42 |
| Radius of each normalized landmark | 21 |
| Index-segment and wrist-to-index direction values | 4 |
| Thumb-to-index gap ratio | 1 |
| Thumb/finger extension and additional hand-shape values | 8 |
| **Total** | **76** |

The extractor centers the hand at the wrist, rotates it so the middle-finger axis is normalized, scales it, normalizes hand-side orientation, and preserves the original screen-direction features needed for Left/Right/Up/Down.

### 4. ONNX inference

`backend/model_runtime.py::OnnxClassifier` loads `models/gesture_mlp_production.onnx` with ONNX Runtime's CPU execution provider. It verifies the model hash, class order, feature order, tensor types/shapes, and recorded conversion parity before accepting the model. The small MLP uses sequential execution with one intra-op and one inter-op thread.

The graph contract is:

| Tensor | Type and shape | Purpose |
|---|---|---|
| Input `landmark_features` | `float32 [N, 76]` | Exact feature rows |
| Output `label` | `int64 [N]` | Internal argmax output; the runtime does not depend on it |
| Output `probabilities` | `float32 [N, 8]` | Canonical command probabilities |
| Output `known_gesture_mass` | `float32 [N, 1]` | Evidence that the input belongs to the known vocabulary |

### 5. Rejection, stabilization, and actions

`backend/geometry.py::GeometryResolver` rejects incompatible directional, Dorsal, Like, OK, and Open Palm shapes. `backend/runtime.py::TemporalGate` then applies known-mass, confidence, probability-margin, EMA, stable-frame, and minimum-hold checks. `RuntimeSession` also handles release frames, action latching, and cooldown.

This layered decision path is why the top ONNX class is not automatically an action. A hand can be detected and scored while the final result remains `no_gesture` / `Wait / No Action`.

### 6. API and dashboard

`backend/app.py` composes the services and exposes the REST and WebSocket API. `app/dashboard.tsx` connects to `/ws/ncm-live`, displays the MJPEG preview and landmarks, shows decision/timing diagnostics, and applies confirmed actions to an uploaded demo image or video. The demo affects only media inside the browser; it does not control Windows, iOS, or board firmware.

### 7. Reviewed feedback and data collection

The Feedback tab writes reviewed correction records through `backend/storage.py`. **Save only** records evidence. **Safe Learn** may update a separate, bounded NumPy residual adapter only after holdout/replay safety checks in `backend/online_learning.py`; it never rewrites the base ONNX graph.

The Data Collection tab uses `backend/data_collection.py` to freeze the exact NCM frame being reviewed, then stores the raw JPEG and an authoritative JSON sidecar. The human prompt/confirmation is the label; the current prediction is diagnostic only.

## Gesture and model contract

The class order is fixed and must be identical on Windows, iOS, and any replacement implementation:

| Index | Model label | Display name | Action |
|---:|---|---|---|
| 0 | `left` | Left | Move Left |
| 1 | `right` | Right | Move Right |
| 2 | `up` | Up | Move Up |
| 3 | `down` | Down | Move Down |
| 4 | `open_palm` | Open Palm | Enable / Disable Object Tracking |
| 5 | `like` | Like | Play / Pause |
| 6 | `dorsal` | Dorsal | Return to Default Position |
| 7 | `ok` | OK | Start / Stop Recording |

`no_gesture` is a rejection/feedback label, not output index 8. It means no hand, an invalid hand, an unknown shape, an incompatible pose, or a candidate that has not passed temporal gates.

### Direction and mirroring contract

Pixels, preview frames, landmarks, and feature extraction stay **unmirrored**. The model/config contains the NCM-specific semantic calibration:

- positive index-finger screen `x` direction maps to `left`;
- negative index-finger screen `x` direction maps to `right`;
- no horizontal image flip is applied;
- `NCM_ROTATE_180=true` is available only when the sensor is physically upside down.

Do not add a second Left/Right swap in the UI or iOS code.

## How the 10-FPS application works

There is no separate executable or folder named "10-FPS application." In this repository that term means the **FastAPI-backed web dashboard**, which is the strict capped path. It does not mean the standalone OpenCV viewer.

The current production settings in `models/gesture_mobile_runtime_config.json` are:

| Setting | Current value |
|---|---:|
| Target rate | 10 FPS |
| Minimum interval between admitted inference starts | 100 ms |
| Frame processing budget | 100 ms |
| Probability EMA alpha | 0.45 |
| Confidence floor | 0.80 |
| Probability-margin floor | 0.18 |
| Known-gesture-mass floor | **0.95** |
| Stable frames | 3 |
| Minimum stable hold | 0.18 s |
| Release frames | 3 |
| Action cooldown | 0.80 s |

`backend/runtime.py::InferenceStartLimiter` has a hard maximum of 10 admitted inference starts per second. The singleton limiter in `backend/app.py` is shared by all WebSocket and image-inference clients, so the cap is backend-wide rather than 10 FPS per browser. Failed inference attempts consume their reserved slot too.

Camera FPS and application FPS are different:

- the board may deliver more than 10 JPEGs per second;
- the backend admits inference starts at no more than 10 per second;
- the dashboard's 100 ms **capacity pass** means a measured frame finished inside the budget; it is not an accuracy result;
- sustained physical-camera validation is still required.

Historical qualification sections in metadata may show an evaluation threshold of `0.80`. That describes a recorded offline experiment. The live deployment gate is `temporal_smoothing.known_mass_floor` in the current runtime JSON, which is `0.95`.

The standalone `scripts/run_live_camera.py` calls `InferenceEngine.process_frame()` for every captured frame and does **not** use the backend-wide limiter or the dashboard's 0.92 crop. It also does not attach the Safe Learn adapter or perform the full production manifest check. Use it as a convenient same-model viewer, not as proof that the qualified dashboard path is reproduced.

## Repository map

### Top-level folders

| Path | What it contains |
|---|---|
| `app/` | React/TypeScript dashboard source |
| `backend/` | FastAPI API, camera transport, image/landmark/model runtime, gates, feedback, collection, tests |
| `models/` | Production ONNX, MediaPipe/BlazeFace assets, runtime JSON, metadata, integrity manifest, evaluation caches/CSVs |
| `scripts/` | Setup/start/stop/verify utilities, standalone viewer, collection, training, qualification, packaging |
| `research/` | Reusable v18_20 and multiview data/training/evaluation code |
| `data/v18_20/` | Preserved source-derived dataset artifacts and provenance used by v18_20 |
| `artifacts/v18_20/` | Candidate, baselines, metrics, plots, audits, and qualification evidence |
| `artifacts/multiview/` | Generated multiview audit/candidate outputs; not the active production model |
| `public/gesture-guides/` | Images used by the guided collection UI |
| `reference/` | Supplied C++ NCM/JLIP receiver used as the transport contract |
| `swift/` | Corrected 76-D Swift feature extractor and partial parity smoke test |
| `handover_02/` | Latest committed split and all-in-one iOS handover snapshot |
| `handover/` | Older iOS handover 01 kept for history; do not start a new integration from it |
| `examples/` | Minimal ONNX inference example for an already prepared 76-D feature row |
| `feedback/` | Local reviewed feedback generated at runtime; normally empty in a clean handover |
| `.runtime/` | Generated PIDs, logs, local diagnostics, and temporary runtime data |
| `.venv/`, `node_modules/`, `.next/`, `dist/` | Generated dependencies/build output; recreate on the recipient's machine |

### Frontend source files

| File | Responsibility |
|---|---|
| `app/page.tsx` | Renders the dashboard route |
| `app/layout.tsx` | Page metadata and global stylesheet hookup |
| `app/dashboard.tsx` | Main five-tab UI, API/WebSocket connection, NCM controls, predictions, feedback, analytics, uploaded-media demo |
| `app/data-collection.tsx` | Guided participant/image collection UI |
| `app/range-diagnostics.tsx` | Range calibration, trials, and CSV export UI |
| `app/range-math.ts` | Range calibration/estimation math |
| `app/range-math.test.mjs` | Range-math regression tests |
| `app/globals.css` | Dashboard layout, responsive behavior, and visual styling |

### Backend source files

| File | Responsibility |
|---|---|
| `backend/app.py` | Service construction, routes/WebSockets, CORS, shared 10-FPS admission limiter |
| `backend/config.py` | Fixed classes/actions/feature names plus runtime and operational environment loading |
| `backend/ncm_camera.py` | UDP discovery, TCP/JLIP parser, heartbeat, reconnect, latest validated JPEG, transport metrics |
| `backend/model_runtime.py` | ONNX loading, image preprocessing, landmark stabilization, inference orchestration, result/timing payload |
| `backend/hand_detection.py` | MediaPipe VIDEO tracker and bounded recovery passes |
| `backend/face_guard.py` | BlazeFace-based small false-hand exclusion near faces/ears |
| `backend/geometry.py` | Exact 76-D feature construction and gesture pose/direction resolution |
| `backend/runtime.py` | Hard rate limiter, probability EMA, temporal decision gate |
| `backend/follow_object.py` | Open-Palm object-tracking demo state machine |
| `backend/storage.py` | Feedback records/images and metric-file loading |
| `backend/online_learning.py` | Validation-gated residual adapter, negative prototypes, backups, rollback |
| `backend/data_collection.py` | Capture plan, frozen frame, participant/sample storage and deletion |
| `backend/artifact_integrity.py` | Trusted artifact hashes and release fingerprint |
| `backend/observability.py` | JSON logging and in-memory/Prometheus runtime metrics |
| `backend/tests/` | Transport, model, geometry, timing, collection, online-learning, and regression tests |

### Important model files

| File | Responsibility |
|---|---|
| `models/gesture_mlp_production.onnx` | Immutable eight-command base classifier plus known-mass output |
| `models/gesture_mlp_onnx_metadata.json` | Exact tensor/class/feature contract, model SHA-256, parity and qualification evidence |
| `models/gesture_mobile_runtime_config.json` | Live thresholds, 10-FPS target, actions, detector/geometry/temporal settings |
| `models/gesture_artifact_manifest.json` | Trusted sizes/hashes for the complete runtime artifact set |
| `models/hand_landmarker.task` | MediaPipe 21-point hand detector/tracker model |
| `models/blaze_face_short_range.tflite` | Pinned face-guard model |
| `models/gesture_online_*_cache.npz` | Replay, validation, and untouched-test evidence for guarded updates/regression |
| `models/eight_gesture_*.csv` | Human-readable release metrics and confusion/hard-case reports |

### Operational and research scripts

| File | When to use it |
|---|---|
| `setup_ncm_onnx_dashboard.bat` | First-time Windows setup wrapper |
| `start_ncm_onnx_dashboard.bat` | Preflight, build, and launch the full dashboard |
| `stop_ncm_onnx_dashboard.bat` | Stop the two processes launched by the start script |
| `check_ncm_link.bat` | Read-only Windows NCM address/neighbor/TCP diagnostic |
| `verify_ncm_onnx_release.bat` | Python tests, preflight, TypeScript, ESLint, frontend build |
| `run_live_camera.bat` | Interactive launcher for the standalone OpenCV viewer |
| `scripts/run_live_camera.py` | Direct standalone webcam/NCM viewer |
| `scripts/onnx_preflight.py` | Validate the runtime/model contract and load both detector/classifier models |
| `scripts/build_artifact_manifest.py` | Re-hash reviewed deployment artifacts after an intentional release change |
| `scripts/download_hand_landmarker.py` | Install the MediaPipe task asset when absent |
| `scripts/download_face_detector.py` | Install/verify the pinned face-guard asset |
| `scripts/capture_multiview_data.py` | Technical directional capture alternative to the dashboard collector |
| `scripts/train_multiview_model.py` | Audit a collection or train/qualify a non-production multiview candidate |
| `scripts/collect_v18_20_data.py` | Re-download/rebuild public-source inputs with provenance |
| `scripts/qualify_v18_20.py` | Evaluate, and only with `--deploy`, promote the v18_20 candidate |
| `scripts/build_eight_gesture_model.py` | Convert a legacy sibling 15-class release to the eight-output wrapper; requires external source/cache paths |
| `scripts/create_v18_20_notebook.py` | Regenerate the experiment notebook source |
| `scripts/package_v18_20.py` | Rebuild the portable notebook/model evidence ZIP |
| `scripts/create_handover_zip.ps1` | Build a clean timestamped full-project handover ZIP |
| `v18_20.ipynb` | Executed, reviewable model training/evaluation workflow |

### Root configuration files

| File | Responsibility |
|---|---|
| `package.json`, `pnpm-lock.yaml` | Frontend engines, commands, packages, and pinned dependency graph |
| `requirements.lock.txt` | Exact Python packages used by normal setup/runtime verification |
| `requirements.txt` | Supported runtime dependency ranges |
| `requirements-model-build.txt` | Additional ONNX/scikit-learn model-build dependencies |
| `requirements-notebook.txt` | Additional notebook, plotting, download, and training tools |
| `.env.example` | Environment-variable reference; not loaded automatically |
| `vite.config.ts` | Vinext/Vite production build configuration |
| `vite.local.config.ts` | Windows-local Vite development server configuration |
| `tsconfig.json`, `eslint.config.mjs` | TypeScript and lint configuration |
| `pytest.ini` | Python test discovery/configuration |

More focused documentation:

- [DATA_COLLECTION.md](DATA_COLLECTION.md)
- [MULTIVIEW_TRAINING.md](MULTIVIEW_TRAINING.md)
- [V18_20_README.md](V18_20_README.md)
- [MODEL_CARD.md](MODEL_CARD.md)
- [HAND_DISTANCE.md](HAND_DISTANCE.md)
- [HANDOVER_CHECKLIST.md](HANDOVER_CHECKLIST.md)

## Windows setup

### Requirements

- 64-bit Windows 10 or 11;
- 64-bit Python 3.11 or 3.12;
- Node.js 22.13 or newer (`node --version`);
- Corepack/pnpm (`corepack enable`, if pnpm is not already available);
- Microsoft Visual C++ 2015-2022 x64 Redistributable for ONNX Runtime;
- a working USB-NCM driver/interface for the development board;
- board firmware that exposes UDP discovery on 5001 and JLIP/TCP on 5000;
- internet access during first setup if dependencies or detector assets are absent.

Open the project itself in VS Code:

```powershell
cd "C:\path\to\gesture-dashboard-onnx-ncm-camera"
code .
```

Run setup:

```powershell
.\setup_ncm_onnx_dashboard.bat
```

The setup script:

1. finds a supported 64-bit Python;
2. creates `.venv` if needed;
3. upgrades pip and installs `requirements.lock.txt`;
4. confirms the ONNX Runtime CPU provider loads;
5. ensures the detector assets are present (the face-model download is checksum-pinned);
6. installs the pinned pnpm dependencies;
7. rebuilds the artifact integrity manifest;
8. runs the ONNX deployment preflight.

Do not reuse another person's `.venv` or `node_modules`. Native Python/Node packages are machine-specific; rerun setup after transfer.

If the received package must be authenticated, verify its provenance and original hashes **before** setup. The setup script intentionally rebuilds `models/gesture_artifact_manifest.json` from the files on disk, so the regenerated manifest describes the local post-setup bundle rather than proving who supplied it.

### Configure the USB-NCM interface

The host-side USB-NCM adapter must own `192.168.50.1` with prefix length `30` (`255.255.255.252`). The board must own `192.168.50.2`. No gateway or DNS entry is required for this point-to-point subnet.

First identify the adapter:

```powershell
Get-NetAdapter
Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object IPAddress -eq "192.168.50.1"
```

Use Windows network settings to assign the address to the USB-NCM adapter, or have an administrator run the equivalent command after replacing the placeholder with the verified adapter name:

```powershell
New-NetIPAddress `
  -InterfaceAlias "<USB-NCM adapter name>" `
  -IPAddress "192.168.50.1" `
  -PrefixLength 30
```

Do not assign `192.168.50.1` to Wi-Fi or ordinary Ethernet. Then run:

```powershell
.\check_ncm_link.bat
```

This diagnostic is read-only. It reports the host address/prefix, neighbor/ARP state, route source, and whether TCP 5000 accepts a connection.

### Configuration variables

`.env.example` documents defaults, but it is **not automatically loaded**. Set overrides in the same PowerShell session before launching:

```powershell
$env:NCM_HOST_IP = "192.168.50.1"
$env:NCM_DEVICE_IP = "192.168.50.2"
$env:NCM_TCP_PORT = "5000"
$env:NCM_DISCOVERY_PORT = "5001"
$env:NCM_DISCOVERY_PAYLOAD = "JL-CAMERA-DISCOVER"
$env:NCM_ROTATE_180 = "false"
$env:GESTURE_COLLECTION_DIR = "D:\Data-Collection"
.\start_ncm_onnx_dashboard.bat
```

Other supported NCM variables include connect/receive/discovery timeouts, reconnect delay, maximum JPEG bytes, and auto-connect; see `.env.example` and `backend/ncm_camera.py`. Useful runtime paths include `GESTURE_MODELS_DIR`, `GESTURE_FEEDBACK_DIR`, `GESTURE_COLLECTION_DIR`, and `GESTURE_LOG_DIR`.

The standard start script deliberately forces production safeguards: the manifest is required, forced learning is disabled, and artifact reload/rollback endpoints are disabled.

## Running the applications

### A. Full dashboard: the strict 10-FPS application

```powershell
.\start_ncm_onnx_dashboard.bat
```

To launch without automatically opening a browser:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\start_ncm_onnx_local.ps1 `
  -NoBrowser
```

Stop it with:

```powershell
.\stop_ncm_onnx_dashboard.bat
```

Runtime files:

- `.runtime/backend.stdout.log`
- `.runtime/backend.stderr.log`
- `.runtime/frontend.stdout.log`
- `.runtime/frontend.stderr.log`
- `.runtime/logs/backend.jsonl`
- `.runtime/backend.pid`
- `.runtime/frontend.pid`

### B. Manual development mode

Use two PowerShell terminals from the project root.

Terminal 1 - backend:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:GESTURE_ENVIRONMENT = "development"
$env:GESTURE_REQUIRE_ARTIFACT_MANIFEST = "true"
.\.venv\Scripts\python.exe -m uvicorn backend.app:app `
  --host 127.0.0.1 `
  --port 8200
```

Terminal 2 - frontend:

```powershell
$env:NEXT_PUBLIC_GESTURE_API_URL = "http://127.0.0.1:8200"
pnpm dev
```

`pnpm dev` uses `vite.local.config.ts` and serves `127.0.0.1:3200`. Set `NEXT_PUBLIC_GESTURE_API_URL` before starting the frontend because it is compiled into the client bundle.

### C. Backend/API only

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m uvicorn backend.app:app `
  --host 127.0.0.1 `
  --port 8200
```

Then use <http://127.0.0.1:8200/docs> or the endpoints listed below.

### D. Standalone Python/OpenCV viewer

Interactive launcher:

```powershell
.\run_live_camera.bat
```

Choose `1` for a PC webcam or `2` for the USB-NCM board.

Direct PC-webcam commands:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_live_camera.py --source webcam
.\.venv\Scripts\python.exe .\scripts\run_live_camera.py --source webcam --webcam-id 1
```

Direct NCM command:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_live_camera.py --source ncm
```

Press `q` or `Esc` in the OpenCV window to exit. The overlay shows the skeleton, predicted gesture, confidence, action, known mass, and observed processing FPS.

The standalone NCM path currently hard-codes `192.168.50.1 -> 192.168.50.2:5000`; its address does not come from `.env.example`. Change `scripts/run_live_camera.py` if a different board subnet is required.

### E. Raw ONNX example

`examples/onnx_inference_example.py` accepts one already prepared 76-number JSON array:

```powershell
.\.venv\Scripts\python.exe .\examples\onnx_inference_example.py `
  --features-json .\features.json
```

It demonstrates raw model probabilities only. It does not perform image decoding, MediaPipe detection, geometry checks, temporal gating, or action execution.

### F. Notebook live test

Install the notebook dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
```

Open `v18_20.ipynb` in VS Code and select the project's `.venv` kernel. The opt-in live helper in `research/live_camera.py` supports `ncm`, `webcam`, and Colab sources, consumes the latest observation at no more than 10 FPS, and returns records for inspection. NCM notebook mode expects the backend on port 8200.

## Using the dashboard

The five tabs are:

| Tab | Purpose |
|---|---|
| **01 Live** | Connect/disconnect the board, watch preview/landmarks, inspect the decision, run the uploaded-media demo |
| **02 Analytics** | Inspect decisions, processing timing, recent actions, and release metric CSVs |
| **03 Feedback** | Save a reviewed correction or attempt a validation-gated Safe Learn update |
| **04 Setup** | Inspect model/artifact readiness, runtime configuration, NCM state, and setup guidance |
| **05 Data Collection** | Collect balanced, human-reviewed JPEG/JSON training records |

Recommended live sequence:

1. Select **Run UDP discovery** and confirm a reply from `192.168.50.2:5001`.
2. Select **Connect board camera**.
3. Confirm `Connected`, increasing frame count, a moving preview, and zero CRC/header/JPEG errors.
4. Place one hand inside the frame and hold a gesture until the three-frame/hold gate confirms it.
5. Check the candidate class, final runtime result, confidence, known mass, stability, and processing time.
6. Upload an image or a video up to 60 seconds if the browser action demo is needed.
7. Use Range Diagnostics for a controlled distance test and CSV export; do not treat apparent hand size as absolute depth without user calibration.

Dashboard action history/runtime aggregates are held in backend memory and reset on restart. Range-test rows live in browser component state until export/reload. Browser recording creates a 10-FPS canvas WebM demo; it is separate from NCM capture and requires a compatible browser such as current Chrome or Edge.

### Main NCM/API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /livez` | Process liveness |
| `GET /readyz` | Runtime readiness |
| `GET /api/health` | Model, MediaPipe, NCM, config, integrity, and runtime status |
| `GET /api/config` | Public live configuration |
| `GET /api/ncm/status` | NCM connection/counters/error state |
| `POST /api/ncm/discover` | Send the configured UDP discovery packet |
| `POST /api/ncm/connect` | Start the reconnecting NCM client |
| `POST /api/ncm/disconnect` | Stop the NCM client |
| `GET /api/ncm/frame.jpg` | Latest validated board JPEG |
| `GET /api/ncm/stream.mjpg` | Browser MJPEG preview |
| `WS /ws/ncm-live` | Server-pushed NCM predictions; browser sends commands, not frames |
| `WS /ws/live` | Generic byte-frame inference WebSocket for another local client |
| `POST /api/predict-image` | Run a still image through the stabilized pipeline |
| `GET /api/models` | Active/available model state |
| `GET /api/artifacts` | Artifact paths and integrity status |
| `GET /api/metrics` | Release metric CSV data |
| `GET /api/runtime-metrics` | In-memory runtime summary |
| `GET /api/actions` | Recent in-memory confirmed action history |
| `GET/POST /api/feedback` | Feedback summary/save |
| `/api/collection/*` | Participant, freeze, save, list, image, and delete operations |

OpenAPI at `/docs` is the authoritative request/response reference for REST routes.

## Collecting and retraining data

### Choose the correct workflow

| Goal | Workflow |
|---|---|
| Run the supplied release | Do not retrain; run setup/preflight and use `models/` |
| Collect normal commands and hard negatives | Dashboard **05 Data Collection** |
| Collect only technical multi-angle direction data | `scripts/capture_multiview_data.py` |
| Audit or train a participant-isolated replacement candidate | `scripts/train_multiview_model.py` |
| Reproduce/review the v18_20 experiment | `v18_20.ipynb` plus `research/v18_20.py` |
| Convert the older sibling 15-class model | `scripts/build_eight_gesture_model.py` with explicit external source/cache paths |

### Dashboard collection

Set a writable collection root before starting if `D:\Data-Collection` is not appropriate:

```powershell
$env:GESTURE_COLLECTION_DIR = "C:\GestureData"
.\start_ncm_onnx_dashboard.bat
```

In **05 Data Collection**:

1. create/reuse an anonymous participant ID;
2. follow the displayed label, view, hand, finger orientation, light, and distance prompt;
3. capture, review the frozen image, and confirm or retake;
4. collect 12 varied images per prompt (3 for each hand/orientation group where applicable);
5. complete 54 prompts, or 648 images for one full participant pass;
6. delete incorrect samples using the UI so both JPEG and JSON are removed together.

The saved structure is:

```text
<collection-root>/
  dataset.json
  participants/
    person-001/
      participant.json
      <label>/<view>/<session-id>/
        <sample-id>.jpg
        <sample-id>.json
```

The JPEG is the original board frame without overlay. The JSON holds the reviewed label, provenance, hand/view/lighting/distance fields, checksum, frame ID, and any available landmarks/features/diagnostics. Saving images does not retrain the model or invoke Safe Learn.

Collect with consent and anonymous IDs. Images can contain faces. Back up the entire collection root, including every JSON sidecar.

### Audit a dashboard collection

Install training dependencies first:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
```

Run an audit without fitting a model:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_multiview_model `
  --dataset "D:\Data-Collection" `
  --audit-only
```

The reader verifies schema, labels, file paths, checksums, duplicate images, and participant-isolated train/validation/test assignment. Raw images without valid 21-point landmarks remain reported raw cases; the 76-D classifier cannot learn directly from pixels.

### Technical multiview capture

With the dashboard/backend running and NCM connected:

```powershell
.\.venv\Scripts\python.exe -m scripts.capture_multiview_data `
  --participant operator-001 `
  --source ncm
```

For a PC webcam:

```powershell
.\.venv\Scripts\python.exe -m scripts.capture_multiview_data `
  --participant operator-001 `
  --source webcam `
  --webcam-device 0
```

Controls: `Space` accepts a reviewed matching frame, `N` skips the current label/view bucket, and `Esc` exits while retaining prior saves. The default is eight samples for each of eight viewpoints across the four directional commands, or 256 images per participant. Output defaults to `data/multiview/`.

### Train and qualify a multiview candidate

Audit first:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_multiview_model --audit-only
```

Train only after coverage passes:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_multiview_model
```

The workflow validates participant isolation, balances the training set, exports `artifacts/multiview/candidate.onnx`, and compares it with production on public and captured untouched tests. It never overwrites `models/gesture_mlp_production.onnx`. `--allow-incomplete-coverage` creates an explicitly exploratory candidate that cannot qualify for release.

At least seven participants are needed by the current minimum split rules, and deterministic 70/15/15 allocation can require more. See [MULTIVIEW_TRAINING.md](MULTIVIEW_TRAINING.md) for exact per-split coverage gates.

There is no automatic production-promotion command for the multiview candidate. Copying only `candidate.onnx` will fail the metadata/manifest contract. A maintainer must create a complete versioned release after offline qualification and live NCM acceptance.

### Reproduce the v18_20 model work

The executed `v18_20.ipynb` is the main reproducible narrative. It uses `research/v18_20.py`, inputs under `data/v18_20/`, and evidence under `artifacts/v18_20/`.

The notebook's default switches are read-only (`INSTALL_REQUIREMENTS`, `REBUILD_DATA`, `RETRAIN`, and `RUN_LIVE_CAMERA` are false). Enable only the stages you intend to run.

If source inputs must be rebuilt, review the HaGRID, Dorsal Hands, and 11K Hands licensing/provenance in `data/v18_20/sources.json` and [V18_20_README.md](V18_20_README.md), then run:

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_v18_20_data --source all
```

Network downloads are large and dataset licenses still apply. Open the notebook, keep test data out of model/threshold selection, run the training/evaluation cells, and inspect the generated candidate and audit evidence.

Qualification without deployment:

```powershell
.\.venv\Scripts\python.exe -m scripts.qualify_v18_20
```

Only after offline review and a separate live NCM acceptance pass, the explicit promotion command is:

```powershell
.\.venv\Scripts\python.exe -m scripts.qualify_v18_20 --deploy
```

`--deploy` intentionally replaces production model/config/metadata/CSV outputs and rebuilds release evidence. Do not use it merely to test a candidate. After any intentional artifact release change, run:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_artifact_manifest.py
.\.venv\Scripts\python.exe .\scripts\onnx_preflight.py
.\verify_ncm_onnx_release.bat
```

The manifest proves file integrity/consistency, not real-camera accuracy.

### Developer-only and legacy tools

- `scripts/build_eight_gesture_model.py` converts an older 15-output sibling model. Its defaults require sibling `gesture-dashboard-onnx` and `gesture-dashboard-joblib` folders and it writes into `models/`; it is not the normal v18_20 handover workflow.
- `scripts/check_collection_runtime.py` compares against Git history and retained `.runtime` camera cases. It is a developer regression tool and will not work from every clean handover.
- `scripts/create_v18_20_notebook.py` regenerates and overwrites `v18_20.ipynb`; recipients normally open the checked-in notebook instead.
- `scripts/build_artifact_manifest.py` hashes artifacts. It does not qualify a model or repair wrong metadata.

## Implementing the pipeline on iOS

### What is included and what is not

Use the newest committed iOS snapshot under:

```text
handover_02/v2026.09.12_ios_onnx_01/
```

It contains:

- `landmarker/models/hand_landmarker.task`;
- `landmarker/swift/GestureFeatureExtractor.swift`;
- `model/models/gesture_mlp_production.onnx`;
- model metadata/runtime JSON/manifests;
- split landmarker/model ZIPs and an all-in-one ZIP.

The root `swift/GestureFeatureExtractor.swift` is available for direct inspection. The current root runtime/config can evolve after the dated Handover 02 snapshot. **Do not mix files from the root and Handover 02 silently.** Use one internally consistent, versioned snapshot, compare its hashes/config, and publish a new iOS package when root behavior changes.

The Handover 02 copy of `gesture_artifact_manifest.json` is a desktop-wide manifest and references files that are not all present inside its small model subfolder. Do not try to validate that whole manifest relative to only `model/models/`; either ship the complete referenced set or create a mobile-specific manifest for the exact iOS bundle.

This repository does **not** contain an `.xcodeproj`, `.xcworkspace`, `Podfile`, `Info.plist`, Swift NCM/JLIP client, ONNX session wrapper, iOS geometry resolver, temporal gate, or finished iOS screen. There is no command in this folder that runs an iOS app.

There are two possible iOS targets:

1. **Classifier demonstration:** MediaPipe -> supplied feature extractor -> ONNX -> basic thresholds/action. Faster, but not behaviorally equivalent to the Windows runtime.
2. **Production-equivalent port:** also port/test low-light policy, detector recovery, face guard, landmark smoothing, `GeometryResolver`, temporal/release/cooldown logic, transport validation, and observability. This is required before claiming desktop/iOS parity. The current iOS handover does not include the BlazeFace asset used by the desktop face guard.

### 1. Create the Xcode project and install libraries

On a Mac with Xcode, create an iOS Swift app target. The official MediaPipe Tasks iOS guide distributes `MediaPipeTasksVision` through CocoaPods, and ONNX Runtime supplies the `onnxruntime-objc` CocoaPod.

Example `Podfile`:

```ruby
target 'GestureApp' do
  use_frameworks!
  pod 'MediaPipeTasksVision'
  pod 'onnxruntime-objc'
end
```

Then run from the Xcode project directory:

```bash
pod install
open GestureApp.xcworkspace
```

Pin versions and commit `Podfile.lock` after validating a working combination. Use the generated `.xcworkspace`, not the original `.xcodeproj`, after installing pods. ONNX Runtime's Objective-C API can be exposed to Swift with a bridging header containing:

```objc
#import <onnxruntime.h>
```

Official references:

- [Google MediaPipe Hand Landmarker for iOS](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/ios)
- [ONNX Runtime Objective-C setup](https://onnxruntime.ai/docs/get-started/with-obj-c.html)
- [ONNX Runtime iOS deployment example](https://onnxruntime.ai/docs/tutorials/mobile/deploy-ios.html)

### 2. Add the runtime assets

Add these files to the app target with target membership enabled and confirm they appear under **Build Phases -> Copy Bundle Resources**:

```text
hand_landmarker.task
gesture_mlp_production.onnx
GestureFeatureExtractor.swift
```

Also keep `gesture_mlp_onnx_metadata.json` and `gesture_mobile_runtime_config.json` in the bundle if the app will validate the contract/read thresholds instead of duplicating them in Swift.

Run the supplied smoke test in a unit test or debug launch:

```swift
let result = GestureFeatureExtractor.runParitySelfTest()
precondition(result.passed, "76-D feature smoke test failed: \(result.maxFeatureDiff)")
```

This function compares only the first six features at a loose tolerance and returns a hard-coded sample label. It does not run ONNX. Add full 76-value golden fixtures and end-to-end model fixtures before treating parity as proven.

### 3. Add permissions

For the device's own camera, add `NSCameraUsageDescription` to `Info.plist` and request/check AVFoundation authorization.

For a direct NCM connection, add `NSLocalNetworkUsageDescription`, for example:

```xml
<key>NSLocalNetworkUsageDescription</key>
<string>This app connects to the local USB camera board for gesture recognition.</string>
```

Fixed-IP TCP and unicast UDP discovery do not use Bonjour, so `NSBonjourServices` is not needed unless the firmware/app later adopts Bonjour. The current discovery packet is unicast, so it does not require the multicast entitlement. Test local-network permission on a physical device, not only the simulator.

References:

- [Apple camera authorization](https://developer.apple.com/documentation/avfoundation/requesting-authorization-to-capture-and-save-media)
- [Apple local-network usage description](https://developer.apple.com/documentation/bundleresources/information-property-list/nslocalnetworkusagedescription)
- [Apple TN3179: local-network privacy](https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy)

### 4. Configure MediaPipe

Resolve `hand_landmarker.task` from `Bundle.main` and create a Hand Landmarker for `.liveStream` or `.video` mode. Match the desktop settings:

```text
numHands = 1
minHandDetectionConfidence = 0.55
minHandPresenceConfidence = 0.55
minTrackingConfidence = 0.60
```

Use monotonic, increasing millisecond timestamps. In live-stream mode MediaPipe may discard a new frame while busy; this is acceptable when the app always prefers the newest observation.

For the built-in iPhone/iPad camera:

- keep analysis pixels unmirrored (`isVideoMirrored = false` where supported);
- disable automatic mirroring before setting the flag when necessary;
- convert `CMSampleBuffer`/`CVPixelBuffer` to `MPImage` with the correct physical orientation;
- use normalized image landmarks 0 through 20, not world landmarks;
- do not perform another Left/Right label swap.

See [`AVCaptureConnection.isVideoMirrored`](https://developer.apple.com/documentation/avfoundation/avcaptureconnection/isvideomirrored).

### 5. Convert landmarks and run ONNX

From the first detected hand, convert the 21 normalized MediaPipe landmarks in their original index order:

```swift
let points = handLandmarkerResult.landmarks[0].map {
    SIMD2<Float>(Float($0.x), Float($0.y))
}
let features = try GestureFeatureExtractor.landmarksToFeature(points)
precondition(features.count == 76)
```

Create a contiguous `float32` tensor with shape `[1, 76]` and input name `landmark_features`. Query outputs by exact name:

```text
probabilities       float32 [1, 8]
known_gesture_mass  float32 [1, 1]
```

The correct name is `known_gesture_mass`, not `known_mass`. Do not assume output-array position, and do not add `no_gesture` to the probability vector. Validate that all values are finite and normalize the eight probabilities as the desktop runtime does.

`runtime_action` is not an ONNX output. The app derives it only after rejection, geometry, and temporal gates.

### 6. Reproduce the decision state machine

For a minimum compatible gate, in this order:

1. no usable hand -> reset candidate history and return `no_gesture`;
2. feature extraction failure -> reject;
3. apply the ported pose/geometry vetoes from `backend/geometry.py`;
4. require `known_gesture_mass >= 0.95`, except only explicitly ported/tested geometry-supported fallbacks;
5. require top probability `>= 0.80`;
6. require top-minus-second probability `>= 0.18`;
7. apply probability EMA with alpha `0.45`;
8. require the same EMA label for at least 3 frames and at least 0.18 seconds;
9. after firing, require 3 release frames and a 0.80-second cooldown before repeating;
10. map the final class index using the fixed table in this README.

Use a monotonic clock for frame intervals, holds, and cooldowns. A label change or capture gap must start a fresh hold so stale EMA cannot execute an action for a new pose.

The complete desktop resolver includes longer-hold geometry recovery and reviewed-negative veto behavior. Port the code/tests rather than approximating it if exact parity is required.

### 7. Enforce 10 FPS on iOS

Keep only the latest frame and admit at most one inference start every 100 ms. Make the timestamp reservation before MediaPipe/ONNX execution so failures cannot create a burst. Do not queue every camera frame; accumulated old frames cause latency and invalid temporal behavior.

```text
Camera/NCM producer -> atomic latest-frame slot -> 100 ms monotonic scheduler
                   -> MediaPipe -> 76-D -> ONNX -> decision state -> main-thread UI
```

Run model work on a serial background queue or actor. Publish only UI state on the main actor. Measure performance using a Release build on a physical device.

### 8. Use the USB-NCM board from iPhone/iPad

If the target device/accessory exposes the board as Ethernet:

1. connect the board through supported USB-C/Ethernet hardware;
2. configure the device-side Ethernet interface as `192.168.50.1` with subnet mask `255.255.255.252` when the OS/accessory setup permits it;
3. confirm the board answers at `192.168.50.2`;
4. optionally send ASCII `JL-CAMERA-DISCOVER` by unicast UDP to `192.168.50.2:5001`;
5. open an `NWConnection` TCP connection to `192.168.50.2:5000`;
6. implement the incremental JLIP parser below;
7. decode only validated `0x10` JPEG payloads and place the newest frame into the 10-FPS pipeline.

Use the canonical `/30` mask. Older handover wording that also suggests `/24` is not the release network contract.

JLIP header layout:

| Byte offset | Size | Field |
|---:|---:|---|
| 0 | 4 | ASCII magic `JLIP` |
| 4 | 1 | version (`1`) |
| 5 | 1 | packet type |
| 6 | 2 | header length (`24`), big-endian |
| 8 | 4 | sequence, big-endian |
| 12 | 4 | payload length, big-endian |
| 16 | 4 | payload CRC32, big-endian |
| 20 | 4 | timestamp milliseconds, big-endian |

Parser rules:

- append every `NWConnection.receive` chunk to a persistent buffer;
- search/resynchronize on the four magic bytes;
- wait until all 24 header bytes exist;
- reject a wrong version/header length or a payload above the configured maximum;
- wait until the full payload exists, even across many callbacks;
- calculate standard CRC32 over the payload and compare it with the header;
- track sequence gaps;
- reply to type `0x03` heartbeats with a valid empty JLIP type `0x03` packet, matching the supplied receiver;
- for type `0x10`, require JPEG start/end markers before decoding;
- handle multiple complete packets in one callback;
- reconnect with bounded delay and clear stale parser/session state after disconnect.

Never call `UIImage(data:)` on an arbitrary TCP receive chunk. TCP is a byte stream, not a JPEG message transport. The short `NWConnection` example in the dated handover README is schematic and is not a correct JLIP receiver by itself.

### 9. iOS acceptance tests

Before calling the port complete, test on a physical device:

- full 76-value feature fixtures and ONNX fixtures pass;
- ONNX input/output names, shapes, types, class order, and SHA-256 match metadata;
- all eight gestures work with unmirrored pixels and correct Left/Right semantics;
- `no_gesture` produces no action for empty frames, faces/ears, relaxed hands, and other fingers;
- low light, backgrounds, both hands, hand sizes, skin tones, distance, tilt, roll, partial hands, and motion are covered;
- repeated held gestures do not retrigger until release/cooldown rules pass;
- scheduling stays at or below 10 admitted starts/sec without a stale-frame queue;
- JLIP fragmentation, coalescing, CRC failure, oversized payload, reconnect, heartbeat, and sequence-gap cases pass;
- real NCM transport counters stay clean during a sustained run.

The offline model metrics do not prove live iOS or board-camera accuracy.

## Verification and troubleshooting

### Full automated verification

```powershell
.\verify_ncm_onnx_release.bat
```

This runs `pip check`, the backend pytest suite, ONNX/artifact preflight, TypeScript `--noEmit`, ESLint, and a production frontend build. It does not prove that the physical camera, USB link, lighting, gestures, or iOS port work.

### Run checks individually

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe .\scripts\onnx_preflight.py
.\node_modules\.bin\tsc.cmd --noEmit
.\node_modules\.bin\eslint.cmd . --ignore-pattern dist --ignore-pattern .next
pnpm build
```

### Common problems

| Symptom | What to check |
|---|---|
| Setup cannot create `.venv` | Install 64-bit Python 3.11/3.12 and disable the Microsoft Store alias if it masks the real interpreter |
| ONNX Runtime import/DLL error | Install the Microsoft Visual C++ 2015-2022 x64 Redistributable, reopen PowerShell, rerun setup |
| OpenBLAS allocation/thread errors | Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before Python; supplied scripts already do this |
| Frontend dependencies missing | Install Node >=22.13, enable Corepack, rerun setup |
| Port 3200 or 8200 is occupied | Stop the prior managed instance, inspect `.runtime/*.pid`, and identify the owning process before changing anything |
| `/readyz` returns 503 | Inspect `/api/health`, artifact integrity, detector readiness, and backend error log |
| UDP discovery has no reply | Confirm host `192.168.50.1/30`, board ARP, firmware UDP 5001, and firewall/local-network permission |
| TCP 5000 fails | Check ARP/neighbor state, firmware bind/listen/accept, cable/USB-NCM enumeration, and source adapter |
| Connected but no image | Check JLIP byte order/length, CRC32, JPEG markers, sequence/error counters, and firmware fragmentation |
| Preview works but no action | Inspect detection, pose veto, known mass, confidence, margin, stable frames, hold time, and rejection reason |
| Left and Right are inverted | Remove pixel/UI mirroring or an extra label swap; use the model's one semantic calibration |
| Collection root unavailable | Set `GESTURE_COLLECTION_DIR` to an existing writable folder before startup |
| iOS always rejects near zero mass | Verify orientation/mirroring, landmark order, full 76-D parity, float32 `[1,76]`, and named outputs before changing thresholds |
| iOS corrupts intermittent frames | Implement persistent JLIP reassembly; one `NWConnection.receive` callback is not one JPEG |

If ARP, discovery, and TCP all fail, instrument the firmware path in this order: USB NCM NTB parsing, ARP reply for `192.168.50.2`, UDP `recvfrom(5001)`, TCP `listen/accept(5000)`, JLIP header construction, fragmentation, CRC, and JPEG delivery. The Windows application cannot inspect firmware-internal NTB/RX counters.

## Creating a clean handover package

The safest way to transfer the project is:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\create_handover_zip.ps1
```

The script creates a timestamped ZIP in the parent directory. It excludes machine-local or sensitive/generated content such as `.venv`, `node_modules`, `.runtime`, `.next`, `dist`, caches, feedback, and TypeScript build state. The recipient extracts it, runs setup on their PC, verifies the release, and performs the physical NCM acceptance test.

Do not copy `.runtime/*.pid` files to another machine. The stop/start scripts trust those stored process IDs; a stale ID from a different computer could refer to an unrelated local process. The handover script prevents this by excluding `.runtime/`.

Before transfer:

- run `verify_ncm_onnx_release.bat`;
- review [HANDOVER_CHECKLIST.md](HANDOVER_CHECKLIST.md);
- do not include local adapter state/backups, participant data, feedback images, secrets, or credentials unless deliberately reviewed and approved;
- include the full `models/` release unit and the latest `handover_02/` iOS package;
- record the hardware/firmware version and physical board test result separately.

The software preflight and offline metrics are necessary checks, but acceptance still requires all eight commands and rejection cases on the actual target camera, lighting, mounting, and iOS/Windows hardware.
