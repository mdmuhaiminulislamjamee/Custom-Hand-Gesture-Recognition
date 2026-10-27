# Run the current Windows reference

Extract `desktop_reference.zip` into a writable folder. Work inside `desktop_reference`. Do not copy the sender's virtual environment or generated caches.

Install 64-bit Python 3.12 (3.11 is also supported by the setup script), Node.js 22.13+, pnpm/Corepack and the Visual C++ x64 runtime. Initial dependency installation needs internet access.

```powershell
.\setup_ncm_onnx_dashboard.bat
.\check_ncm_link.bat
.\run_live_camera.bat
```

Select 2 for USB-NCM. Host adapter: `192.168.50.1/30`; board: `192.168.50.2:5000`. Select the correct USB adapter rather than changing an unrelated network. Q/Esc closes the viewer. Webcam option 1 is a Windows developer convenience; native app inputs use NCM.

For the dashboard run `start_ncm_onnx_dashboard.bat` and open `http://127.0.0.1:3200`; the API is `http://127.0.0.1:8200`. Use `stop_ncm_onnx_dashboard.bat` to stop its servers. Only one reference application should own the board stream at a time. Loopback addresses on a phone refer to that phone, not this Windows backend.

The standalone viewer analyzes the full board frame. The dashboard uses its 0.92 center crop and serial 10-FPS policy. Select one for native comparison and match it exactly. Both share current model/gesture decisions. XYZ and apparent distance displays are diagnostics, not extra ONNX inputs or measured depth.

Open Palm accepts image XY −160°…−20°; Stop inverted accepts −145°…−35°. Both are inclusive. Outside the relevant band is no gesture. The image-plane angle differs from the displayed world-plane angles.

For software checks:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-handover-checks.txt
.\.venv\Scripts\python.exe -m scripts.onnx_preflight
.\.venv\Scripts\python.exe scripts\verify_backend_suite.py
```

The test runner isolates native-library memory by running one file per process. Results appear in `.runtime/backend_verification`. The included public/derived landmark qualification files support the tests; full retraining inputs and feedback photos are not shipped. Some historical training/release-builder commands reference datasets or earlier releases outside this delivery and are not part of the supported verification workflow.

The complete live backend and browser source, dependency locks, BAT/PowerShell launchers and current models are included. Native porting guidance is in the outer package; dated handover guides/scripts retained in this developer snapshot are historical.
