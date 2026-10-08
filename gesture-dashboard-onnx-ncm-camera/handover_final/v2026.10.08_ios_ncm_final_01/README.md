# Final handover — twelve-gesture ONNX runtime

Release: **v2026.10.08_ios_ncm_final_01**, prepared 8 October 2026 (Asia/Dhaka).

Start here. This bundle contains the current production model and latest desktop recognition rules, including the 6 October Like/Down fixes. Send this entire bundle to the app team. Use one dated bundle throughout integration; older September/3 October handovers have different rules.

**iOS input contract:** the iOS app uses only the USB-NCM board camera. The iPhone/iPad built-in camera is not part of the integration. The native path is `NCM TCP/JLIP -> validated JPEG -> image preparation -> MediaPipe -> ONNX -> geometry/temporal gate -> app action`.

**Delivery status:** the Windows NCM reference is runnable. Native Swift integration sources, a Swift NCM/JLIP receiver, an ONNX wrapper, MediaPipe wiring, and comparison fixtures are supplied. This is an integration handover for an existing iOS app, not a signed iPhone app or finished Xcode project. Swift compilation, CocoaPods resolution, full NCM frame preprocessing/recovery integration, and physical iPhone/NCM acceptance remain to be completed on Apple hardware. See [IOS_INTEGRATION.md](IOS_INTEGRATION.md) for steps and responsibilities.

## 1. Check the delivery

Extract the outer ZIP into a writable local folder. In Terminal on the Mac:

~~~sh
cd /path/to/v2026.10.07_ios_final_01
python3 verify_package.py
~~~

On Windows use `py -3.12 verify_package.py`. Expected: JSON with a verified file count, model SHA-256, and `desktop_ios_assets_match: true`. Python 3.11 or 3.12 is sufficient for this checksum check; no extra packages are needed. Verify before installing or editing files inside the delivery folder.

The production model SHA-256 is:

~~~text
7434fffefd8d4cd3c6e82d8edacfd7c97b811beddea2bff3d506683f4dc0355e
~~~

The model has not been retrained for this handover. Recent improvements are in detection, geometry validation, and temporal behavior; copying only the ONNX file does not copy those improvements.

## 2. Run the same reference used by the BAT file

Extract `desktop_reference.zip` into a separate writable directory, then open its `desktop_reference` folder. Follow [WINDOWS_RUN.md](WINDOWS_RUN.md):

1. Install 64-bit Python 3.12, Node.js 22.13 or newer, pnpm/Corepack, and the Microsoft Visual C++ x64 runtime. Initial setup needs internet access.
2. Run `setup_ncm_onnx_dashboard.bat` once.
3. Run `run_live_camera.bat` and choose **2** for USB-NCM. Press **Q** or **Esc** inside the camera window to stop. Webcam option 1 exists only as a Windows developer convenience and is not the iOS input reference.
4. For the dashboard use `start_ncm_onnx_dashboard.bat` and open `http://127.0.0.1:3200`. Use `stop_ncm_onnx_dashboard.bat` to stop its servers.

For NCM, the Windows network interface is `192.168.50.1/30`, the board is `192.168.50.2`, TCP port `5000`; run `check_ncm_link.bat` first.

The standalone NCM BAT uses the full board frame and does not enforce the dashboard's 10-FPS scheduler. The NCM dashboard applies a 0.92 central square crop and limits inference starts to 10 FPS. The app team must choose and document which of these two NCM policies its product uses. The recommended production-equivalent target is the dashboard policy: newest valid board JPEG, 0.92 central crop, one serial inference stream at 10 FPS.

## 3. Integrate into the iOS app

Follow [IOS_INTEGRATION.md](IOS_INTEGRATION.md) in order: Xcode/CocoaPods setup, resource target membership, core Swift regression tests, ONNX parity, NCM/JLIP networking, JPEG-to-MediaPipe wiring, then full frame-pipeline acceptance. No direct iOS camera capture, Python server, BAT file, browser dashboard, or training dataset is required inside the iOS app.

| Folder/file | Purpose |
|---|---|
| `ios/models/` | ONNX, metadata/runtime JSON, MediaPipe hand task and BlazeFace model |
| `ios/swift/*.swift` | Feature extraction, current geometry, temporal hold, action latch, decision pipeline |
| `ios/swift/ios/` | NCM/JLIP receiver, ONNX adapter, landmark engine, MediaPipe starter, bridging header, Podfile and local-network permission template |
| `ios/swift/tests/` | 294 geometry/feature cases, temporal expectations, 26 ONNX cases, Mac test runner |
| `ios/reference/backend/` | Current Python source for the complete frame pipeline and JLIP transport |
| `desktop_reference.zip` | Runnable Windows source, pinned runtime requirements, launchers, qualification inputs and tests |
| `NCM_PROTOCOL.md` | Board protocol, reassembly, heartbeat and networking instructions |
| `VALIDATION.md`, `validation/` | Checks run here, logs, and remaining Apple/device checks |
| `RELEASE_MANIFEST.json`, `SHA256SUMS.json` | Source/model identity and hashes of delivered payload files |

## Gesture/action contract

The `probabilities` output has exactly 12 values in this order:

| Index | Gesture | Action |
|---:|---|---|
| 0 | left | Move Left |
| 1 | right | Move Right |
| 2 | up | Move Up |
| 3 | down | Move Down |
| 4 | open_palm | Enable / Disable Object Tracking |
| 5 | like | Play / Pause |
| 6 | dorsal | Return to Default Position |
| 7 | ok | Start / Stop Recording |
| 8 | fist | Mute |
| 9 | thumb_down | Volume Down |
| 10 | peace | Turn 90 Degrees |
| 11 | rock | Backup |

`no_gesture` means rejection/no action; it is not a thirteenth output. Dispatch actions only when `GesturePipelineResult.action` is non-nil. A high model score alone is not an action.

Analysis images remain unmirrored. The calibrated horizontal convention is **positive index x → left; negative index x → right**. Avoid a second swap in the app UI.

## Exclusions and scope

Local virtual environments, node_modules, feedback photos, credentials, live-session logs, and this machine's mutable online-adapter state are excluded. The standalone `run_live_camera.bat` uses the base classifier, so that reference remains reproducible. Dashboard Safe Learn personalization is separate; native online adaptation is not implemented in this Swift package.

Evaluation caches and the landmark-only qualification input in the desktop archive support reproducible software tests. They are not app resources. The complete retraining corpus is not included.

## Troubleshooting

- Hash failure: re-extract the same complete release; do not combine versions.
- Missing iOS model: add all five `ios/models` files to Copy Bundle Resources and check their URLs.
- iOS asks for camera permission: remove the direct-camera capture path and `NSCameraUsageDescription`; this release uses NCM JPEG frames only.
- Always `no_gesture`: run feature/ONNX parity first, then check orientation, handedness, face rejection, and world landmarks. Do not lower thresholds to hide wiring issues.
- Fist rejected: current live rules require 21 valid world landmarks and include view-specific rejection. Image x/y/z is not a substitute for world x/y/z.
- Repeated toggles: preserve one pipeline per stream and dispatch only non-nil actions.
- Growing latency: drop stale frames, avoid concurrent inference and unbounded queues.
- NCM cannot connect: check local-network permission and the accessory's route on iOS. Windows USB success does not establish iPhone compatibility.

Record physical-device results using [VALIDATION.md](VALIDATION.md).
