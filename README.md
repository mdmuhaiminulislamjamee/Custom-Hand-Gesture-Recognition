# Custom Hand Gesture Recognition

## Demo Video Link: https://drive.google.com/drive/folders/15p14beDXWMqsz5UzX4Rb9WQASzxa1ay1

This repository is organized as a monorepo for multiple deployment editions of
the MediaPipe hand-gesture recognition system.

## Editions

| Folder | Runtime | Status |
|---|---|---|
| [`gesture-dashboard-onnx`](gesture-dashboard-onnx/) | MediaPipe + ONNX Runtime | Available |
| [`gesture-dashboard-joblib`](gesture-dashboard-joblib/) | MediaPipe + Python/Joblib | Available |
| [`gesture-dashboard-onnx-ncm-camera`](gesture-dashboard-onnx-ncm-camera/) | MediaPipe + ONNX Runtime + USB-NCM/JLIP board camera | Software verified; physical-board validation pending |

Each edition is self-contained and has its own setup instructions, dependencies,
runtime artifacts, tests, and documentation.

## Run the ONNX edition on Windows

```powershell
cd gesture-dashboard-onnx
.\setup_onnx_dashboard.bat
.\start_onnx_dashboard.bat
```

Then open <http://127.0.0.1:3100>.

See [`gesture-dashboard-onnx/README.md`](gesture-dashboard-onnx/README.md) for the
complete environment setup and operating instructions.

## Run the Joblib edition on Windows

```powershell
cd gesture-dashboard-joblib
.\setup_dashboard.bat
.\start_dashboard.bat
```

Then open <http://127.0.0.1:3000>.

The Joblib edition includes the trusted SVM, MLP, and OnlineMLP classifiers,
MediaPipe hand-landmark extraction, the exact 76-D feature pipeline, temporal EMA,
geometry resolvers, Follow Object, online learning, production qualification, and
the 10 FPS dashboard. It intentionally contains no ONNX or TFLite gesture
classifier.

See [`gesture-dashboard-joblib/README.md`](gesture-dashboard-joblib/README.md) for
complete setup, model-safety, testing, and production instructions.

## Run the ONNX USB-NCM board-camera edition

This edition requires 64-bit Windows 10/11, Python 3.11 or 3.12, Node.js 22.13+
with Corepack/pnpm, the Microsoft Visual C++ 2015–2022 x64 runtime, a working
USB-NCM driver, and compatible board firmware. Before starting, configure the
board's USB-NCM adapter as `192.168.50.1/30`; `check_ncm_link.bat` reports the
current link state but does not change the adapter configuration.

```powershell
cd gesture-dashboard-onnx-ncm-camera
.\setup_ncm_onnx_dashboard.bat
.\check_ncm_link.bat
.\start_ncm_onnx_dashboard.bat
```

Then open <http://127.0.0.1:3200>. This edition connects to the development board
at `192.168.50.2:5000` from the Windows NCM interface `192.168.50.1/30`; it does
not use browser webcam capture. The dashboard can open without the board, but
live inference requires a UDP discovery response on port `5001` and a valid
JLIP/TCP JPEG stream on port `5000`.

Run the software release checks with:

```powershell
.\verify_ncm_onnx_release.bat
```

See
[`gesture-dashboard-onnx-ncm-camera/README.md`](gesture-dashboard-onnx-ncm-camera/README.md)
for the JLIP protocol, firmware diagnostics, and operating instructions.
