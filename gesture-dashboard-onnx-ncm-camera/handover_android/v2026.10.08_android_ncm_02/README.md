# Android NCM gesture handover

Release: **v2026.10.08_android_ncm_02**, prepared 8 October 2026 (Asia/Dhaka).

Send this entire bundle to the Android app team. It uses the same current 12-command ONNX model and NCM board-camera protocol as the iOS `v2026.10.08_ios_ncm_final_03` handover. The Android app's image path is `USB-NCM -> TCP/JLIP -> validated JPEG -> MediaPipe hand landmarks -> 76 features -> ONNX scores -> gesture decision -> app action`. The package supplies source through ONNX scores and the exact references and fixtures needed to finish the decision path. It is not a ready-to-install APK.

**Current status:** Windows NCM and host ONNX checks pass. Kotlin NCM, MediaPipe, feature, ONNX and diagnostic frame sources are supplied but have not been compiled or run on Android in this Windows workspace. Full Android geometry, face/recovery/stabilization, temporal and action dispatch are still app-team work. Android USB-NCM compatibility must be proven on the intended device. Do not treat a high raw model score as an action.

## Start and verify

Extract the outer ZIP. Run `python verify_package.py` from the extracted release folder with Python 3.11/3.12; it checks every delivered file and the model identity without extra dependencies. With the project's Python ONNX dependencies installed, `python verify_package.py --onnx` also runs 26 model vectors.

The production model is `gesture_mlp_production.onnx`, SHA-256 `7434fffefd8d4cd3c6e82d8edacfd7c97b811beddea2bff3d506683f4dc0355e`. It accepts float32 `[1,76]`, returns 12 probabilities and one known-gesture score. `no_gesture` is a rejection state, not a thirteenth probability.

Follow [ANDROID_INTEGRATION.md](ANDROID_INTEGRATION.md) to add files to an existing Android app, verify an Android USB-NCM route, run the debug parity helper, and finish the decision pipeline. Android SDK/API **24+** is the declared minimum from MediaPipe Tasks; actual target-device support still needs validation. The built-in Android camera is not used.

## Which file does what?

| Folder/file | Function | In production Android app? |
|---|---|---|
| `README.md` | Delivery status, verification, full file guide and gesture contract | No; read first |
| `ANDROID_INTEGRATION.md` | Gradle, manifest, assets, NCM route and diagnostic run steps, then remaining port tasks | No; follow during integration |
| `NCM_PROTOCOL.md` | JLIP header, CRC, JPEG, heartbeat and Android network routing contract | No; protocol reference |
| `MODEL_CARD.md` | Model purpose, input/output, training and validation limitations | No; engineering reference |
| `VALIDATION.md` | What was checked and the Android/device acceptance checklist | No; use for sign-off |
| `RELEASE_MANIFEST.json` | Release identity, hashes, class order and readiness flags | No; retain with delivery |
| `SHA256SUMS.json` | Hash and byte count for every delivered payload | No; retain with delivery |
| `verify_package.py` | Recomputes package hashes and optional ONNX parity | No; run before integration |
| `desktop_reference.zip` | Runnable Windows reference matching the BAT/NCM workflow | No; extract separately to compare behavior |
| `android/models/` | Five model/configuration assets described below | Yes; copy to `src/main/assets/` |
| `android/kotlin/` | Native transport and diagnostic inference sources described below | Yes, except parity helper may be debug-only |
| `android/AndroidManifest.fragment.xml` | Required TCP and network-state permission entries | Merge into existing app manifest |
| `android/tests/parity_vectors.json` | 294 feature/geometry fixtures and 26 ONNX cases | Debug/test assets only |
| `android/reference/backend/` | Current Python frame/gesture/JLIP rules and tests | No; porting reference |
| `android/reference/swift/` | Current iOS feature, geometry, temporal, latch and pipeline implementation | No; porting reference |
| `android/reference/protocol/NcmCameraTester.cpp` | Board JLIP receiver reference | No; protocol reference |

### Model assets

| File | Function |
|---|---|
| `gesture_mlp_production.onnx` | Production 12-gesture classifier |
| `gesture_mlp_onnx_metadata.json` | Model SHA, tensor contract and exact class order |
| `gesture_mobile_runtime_config.json` | Shared gesture/action map, crop, timing and detection policy |
| `hand_landmarker.task` | MediaPipe model for 21 hand landmarks, world landmarks and handedness |
| `blaze_face_short_range.tflite` | Face model supplied for the pending face-overlap rejection port |

### Kotlin sources

| File | Function |
|---|---|
| `GestureNcmClient.kt` | TCP connection, JLIP reassembly/CRC/JPEG validation, heartbeat reply and reconnect |
| `NcmNetworkSelector.kt` | Finds an OS-provided NCM/Ethernet `Network` at the reference host address |
| `GestureHandLandmarker.kt` | Runs MediaPipe Hand Landmarker in VIDEO mode on prepared frames |
| `GestureFeatureExtractor.kt` | Converts 21 normalized image landmarks into the model's 76 float features |
| `GestureOnnxClassifier.kt` | Checks model SHA/class order and invokes named ONNX inputs/outputs |
| `GestureFrameAnalyzer.kt` | Single-worker newest-frame diagnostic path, center crop and 10 FPS admission; outputs raw scores only |
| `GestureParity.kt` | Debug helper comparing 294 feature cases and 26 ONNX cases with the reference vectors |

## Gesture and action contract

| Index | Gesture | App action |
|---:|---|---|
| 0 | `left` | Move Left |
| 1 | `right` | Move Right |
| 2 | `up` | Move Up |
| 3 | `down` | Move Down |
| 4 | `open_palm` | Enable / Disable Object Tracking |
| 5 | `like` | Play / Pause |
| 6 | `dorsal` | Return to Default Position |
| 7 | `ok` | Start / Stop Recording |
| 8 | `fist` | Mute |
| 9 | `thumb_down` | Volume Down |
| 10 | `peace` | Turn 90 Degrees |
| 11 | `rock` | Backup |

The action names are contract values from `gesture_mobile_runtime_config.json`. Enable an action only after Android's complete geometry and temporal gate has passed the parity and device checks in [VALIDATION.md](VALIDATION.md).
