# Running the Windows reference

Extract `desktop_reference.zip` first. All commands below run from the extracted `desktop_reference` folder.

## Setup once

Install 64-bit Python 3.12 (verified host: 3.12.10), Node.js 22.13+, pnpm or Corepack, and the Visual C++ 2015–2022 x64 Redistributable. The setup script also supports Python 3.11. Add Python and Node to PATH and reopen the terminal.

~~~powershell
./setup_ncm_onnx_dashboard.bat
~~~

This creates `.venv`, installs `requirements.lock.txt` and the JavaScript lockfile, and checks ONNX/MediaPipe assets. Do not copy the sender's virtual environment. If pnpm/Corepack is unavailable, install pnpm, reopen PowerShell, and rerun setup. Check for successful completion before starting.

## Standalone live camera

~~~powershell
./run_live_camera.bat
~~~

Choose 1 for webcam, 2 for NCM. The menu defaults to webcam. It prints twelve commands and opens an OpenCV window. Q/Esc closes it. For a different webcam:

~~~powershell
./.venv/Scripts/python.exe scripts/run_live_camera.py --source webcam --webcam-id 1
~~~

The viewer displays gesture/action names; it does not send actions to another app or physical hardware. The XYZ triad, apparent hand distance and plane angles are display aids, not additional ONNX inputs. Distance is an apparent-palm-size estimate, not measured depth.

## USB-NCM board

1. Connect the board and identify its USB network adapter.
2. Configure that adapter as `192.168.50.1`, mask `255.255.255.252` (/30). Board: `192.168.50.2`. Do not change an unrelated adapter.
3. Run `check_ncm_link.bat` and check TCP 5000 connectivity.
4. Select NCM in the camera BAT or connect through the dashboard. Only one reference application should own the board stream at a time.

If TCP fails or no valid JPEG arrives, diagnose transport before changing the model. See the outer bundle's NCM_PROTOCOL.md.

## Dashboard

~~~powershell
./start_ncm_onnx_dashboard.bat
~~~

Open `http://127.0.0.1:3200`, run UDP discovery, then connect the board camera. API: `http://127.0.0.1:8200`, with `/readyz`, `/api/health` and `/docs`. Logs: `.runtime`. The frontend rebuilds at startup. Stop with:

~~~powershell
./stop_ncm_onnx_dashboard.bat
~~~

These are Windows loopback addresses. `127.0.0.1` on an iPhone refers to the phone and cannot reach this backend. The native iOS integration runs inference locally.

## Verification

Runtime readiness:

~~~powershell
./.venv/Scripts/python.exe -m scripts.onnx_preflight
~~~

Full software tests require the additional qualification dependencies:

~~~powershell
./.venv/Scripts/python.exe -m pip install -r requirements-handover-checks.txt
./verify_ncm_onnx_release.bat
~~~

Those extra packages serve test/training-helper imports, not live inference. Retain the included landmark-only input and twelve-class evaluation JSONs for the tests. This delivery is not a complete retraining dataset.

If TypeScript reports AppRoutes/LayoutRoutes errors from a generated Next.js validator, remove the generated `.next` cache in this extracted reference folder and rebuild. Do not copy caches between Next.js and Vinext builds. No generated cache is shipped.

Close other camera applications if camera 0 cannot open; install Visual C++ for ONNX DLL failures; free ports 3200/8200 or stop the previous reference; keep the complete `models` directory for manifest checks.
