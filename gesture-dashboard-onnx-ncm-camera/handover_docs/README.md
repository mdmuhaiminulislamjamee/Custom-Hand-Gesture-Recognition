# Final handover — twelve-gesture ONNX runtime

Release: **v2026.10.08_ios_ncm_final_03**, prepared 8 October 2026 (Asia/Dhaka).

Start here. This bundle contains the current production model and latest desktop recognition rules, including the 6 October Like/Down fixes. Send this entire bundle to the app team. Use one dated bundle throughout integration; older September/3 October handovers have different rules.

**iOS input contract:** the iOS app uses only the USB-NCM board camera. The iPhone/iPad built-in camera is not part of the integration. The native path is `NCM TCP/JLIP -> validated JPEG -> image preparation -> MediaPipe -> ONNX -> geometry/temporal gate -> app action`.

**Delivery status:** the Windows NCM reference is runnable. Native Swift integration sources, a Swift NCM/JLIP receiver, an ONNX wrapper, MediaPipe wiring, and comparison fixtures are supplied. This is an integration handover for an existing iOS app, not a signed iPhone app or finished Xcode project. Swift compilation, CocoaPods resolution, full NCM frame preprocessing/recovery integration, and physical iPhone/NCM acceptance remain to be completed on Apple hardware. See [IOS_INTEGRATION.md](IOS_INTEGRATION.md) for steps and responsibilities.

## 1. Check the delivery

Extract the outer ZIP into a writable local folder. In Terminal on the Mac:

~~~sh
cd /path/to/v2026.10.08_ios_ncm_final_03
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

## 4. Folder and file guide

The app team should use this section to decide what belongs in the Xcode target and what is supplied only for reference or verification.

### Package root

| Folder/file | What it does | Include in the iOS app? |
|---|---|---|
| `README.md` | Starting point, package checks, integration summary, file guide and gesture contract | No; keep with the delivery |
| `IOS_INTEGRATION.md` | Step-by-step Xcode, CocoaPods, resource, NCM, MediaPipe, ONNX and acceptance instructions | No; follow during integration |
| `NCM_PROTOCOL.md` | Defines the board TCP/JLIP packet format, JPEG messages, CRC checks, heartbeat replies and reconnection rules | No; protocol reference |
| `MODEL_CARD.md` | Documents model purpose, inputs, 12 outputs, limits and validation scope | No; engineering reference |
| `WINDOWS_RUN.md` | Explains how to run the Windows NCM reference and dashboard | No |
| `VALIDATION.md` | Lists completed checks and the Apple hardware/device checks still required | No; use as acceptance checklist |
| `RELEASE_MANIFEST.json` | Records release identity, model hash, class order, actions and delivery scope | No; retain for traceability |
| `SHA256SUMS.json` | Contains the size and SHA-256 of every payload file so changes or corruption can be detected | No; retain for verification |
| `verify_package.py` | Verifies package hashes and, when dependencies are installed, ONNX output parity | No; run before integration |
| `validation/` | Contains logs from backend tests, ONNX checks, frontend checks and package validation | No; evidence only |
| `desktop_reference.zip` | Complete runnable Windows reference matching the BAT-file workflow | No; extract separately for comparison |
| `ios/` | Contains the resources, Swift code, tests and reference implementation for the iOS integration | Yes; use the required files described below |

### `ios/models/` — resources copied into the app bundle

Add all five files to the Xcode target's **Copy Bundle Resources** phase.

| File | What it does |
|---|---|
| `gesture_mlp_production.onnx` | Production neural-network classifier. It receives the 63-value normalized hand-landmark feature vector and returns 12 gesture probabilities |
| `gesture_mlp_onnx_metadata.json` | Records the model hash, tensor names/shapes and exact output-class order used to verify the ONNX contract |
| `gesture_mobile_runtime_config.json` | Supplies the gesture-to-action mapping and shared mobile runtime settings such as the 10-FPS target |
| `hand_landmarker.task` | MediaPipe hand-landmarker model used to find 21 image landmarks, world landmarks and handedness from each NCM JPEG |
| `blaze_face_short_range.tflite` | Face-detection model used by the reference rejection/safety logic; keep it with the delivered resources while the full frame-pipeline port is completed |

### `ios/swift/` — platform-independent gesture logic

Add these five Swift files to the application target.

| File | What it does |
|---|---|
| `GestureFeatureExtractor.swift` | Validates 21 landmarks, normalizes them and creates the 63-value feature vector in the exact order expected by the ONNX model; also defines class names and action mappings |
| `GestureGeometryResolver.swift` | Applies the current gesture-specific geometry rules and camera-angle recoveries, including the recent Like and Down behavior |
| `GestureTemporalGate.swift` | Smooths probability history, checks confidence/margin/vocabulary mass, and requires stable frames plus a minimum hold time before execution |
| `GestureActionLatch.swift` | Converts an accepted gesture into one app action and prevents repeated firing until release and cooldown conditions are met |
| `GestureDecisionPipeline.swift` | Orchestrates feature extraction, classification, geometry resolution, temporal gating and action latching; this is the main recognition pipeline called by the app |
| `README.md` | Explains the Swift runtime and its regression/parity test commands | No; developer guide |

### `ios/swift/ios/` — iOS and NCM integration

Add the `.swift` files to the application target. Use the header, Podfile and plist fragment as configuration templates.

| File | What it does |
|---|---|
| `GestureNCMClient.swift` | Connects to the board at `192.168.50.2:5000`, reassembles JLIP packets, validates CRC/JPEG data, replies to heartbeats and reconnects after network loss |
| `GestureHandLandmarker.swift` | Configures MediaPipe Tasks Vision and converts each decoded NCM JPEG image into hand landmarks, world landmarks and handedness |
| `GestureLandmarkEngine.swift` | Bridges landmark-detection results into the gesture decision pipeline and manages per-stream recognition state |
| `GestureONNXClassifier.swift` | Loads the bundled ONNX model with ONNX Runtime, checks its identity/contract and returns the 12 probability values |
| `GestureONNXParity.swift` | Runs the supplied ONNX regression vectors on Apple hardware to confirm that iOS inference matches the reference results | Test/qualification target only |
| `Gesture-Bridging-Header.h` | Exposes the ONNX Runtime Objective-C API to Swift; set it as the target's Objective-C bridging header |
| `Podfile.example` | Example CocoaPods configuration for `MediaPipeTasksVision` and `onnxruntime-objc`, with iOS 15 as the minimum target | Use as a template; merge into the app's Podfile |
| `Info.plist.fragment` | Supplies the local-network privacy description required for the NCM connection | Merge the key into the app's Info.plist |

### `ios/swift/tests/` — Mac/Xcode qualification files

| File | What it does | Include in the production app? |
|---|---|---|
| `main.swift` | Runs Swift feature, geometry, temporal and action-latch regression cases | No; test target/tool only |
| `parity_vectors.json` | Contains expected geometry outcomes and 26 ONNX input/output cases tied to the production model hash | No; test resource only |
| `verify_swift.sh` | Builds/runs the portable Swift checks and reports pass/fail on macOS | No; developer script |

### `ios/reference/` — behavior reference, not application code

| Folder/file | What it does |
|---|---|
| `backend/` | Python source and tests defining the current desktop frame pipeline, NCM transport, landmark handling, geometry and runtime behavior. Use it to resolve porting questions; do not bundle it in the iOS app |
| `backend/tests/` | Python regression coverage for camera stability, geometry, NCM packets, ONNX runtime and all 12 gestures. These files are verification evidence, not iOS resources |
| `protocol/NcmCameraTester.cpp` | C++ reference receiver showing the board's JLIP/NCM transport behavior. Use it when debugging packet compatibility; do not compile it into the iOS app |

### Minimum files required in the application target

The production target needs the five files in `ios/models/`, the five platform-independent Swift files in `ios/swift/`, and these four integration sources: `GestureNCMClient.swift`, `GestureHandLandmarker.swift`, `GestureLandmarkEngine.swift`, and `GestureONNXClassifier.swift`. It also needs the ONNX bridging header, the two CocoaPods dependencies from `Podfile.example`, and the local-network permission text from `Info.plist.fragment`.

`GestureONNXParity.swift`, `ios/swift/tests/`, `ios/reference/`, `validation/`, and `desktop_reference.zip` are used for testing, comparison and troubleshooting. They do not belong in the production application bundle.

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
