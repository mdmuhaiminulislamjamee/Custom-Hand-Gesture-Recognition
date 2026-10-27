# iOS integration

The only image source is the USB-NCM board camera. This is an integration delivery for an existing app, not a signed app or finished Xcode project. Native compilation, full decision-port parity and physical iOS/NCM acceptance remain to be performed by the app team.

## 1. Resources and dependencies

Copy `ios/models/` resources into the app bundle and confirm target membership. The ONNX model, metadata, runtime configuration and hand landmarker are required. The face asset is retained for compatibility with older bundles and is unused by the current hand-only policy.

Merge `ios/swift/ios/Podfile.example`, the ONNX bridging header and `Info.plist.fragment` into the existing app. The template declares iOS 15 and the `MediaPipeTasksVision` and `onnxruntime-objc` CocoaPods dependencies. Resolve on the integration Mac and retain tested versions in `Podfile.lock`; no Apple dependency lock or SDK build was qualified on Windows. Open the CocoaPods workspace. The plist template requests local-network access, not built-in camera access.

Add these current sources to the app target:

- `ios/swift/GestureFeatureExtractor.swift`: 76 features and the current 13-output taxonomy/action names.
- `ios/swift/GestureRuntimePolicy.swift`: source-pixel angle measurements and the two current angle ranges.
- `ios/swift/ios/GestureNCMClient.swift`: Network.framework TCP/JLIP receiver and reconnection.
- `ios/swift/ios/GestureHandLandmarker.swift`: diagnostic two-hand MediaPipe VIDEO-mode wiring.
- `ios/swift/ios/GestureONNXClassifier.swift`: identity/contract checks and named 13-output ONNX inference.
- `ios/swift/ios/GestureDiagnosticEngine.swift`: per-hand raw observations preserving image/world landmarks, handedness and prepared-frame dimensions.

Add `GestureONNXParity.swift` only to a debug/qualification target. Do not add `ios/reference/legacy_swift/` to the app target: those older decision sources do not implement today's runtime.

## 2. Run independent parity checks

On a Mac, from `ios/swift`, run `sh tests/verify_features.sh`. It builds the portable feature/angle test with `xcrun swiftc` and compares the supplied current vectors. This command is supplied but has not been run on the Windows preparation host.

Run `GestureONNXParity.run(classifier:vectorsURL:)` on a background queue using `ios/tests/parity_vectors.json`. The fixture model hash must match the bundled model, all probability lengths must be 13, and maximum absolute tensor error must be ≤0.0001. This validates model inference, not action logic.

## 3. Connect NCM and prepare frames

Follow [NCM_PROTOCOL.md](NCM_PROTOCOL.md). The board is `192.168.50.2:5000`; the reference host address is `192.168.50.1/30`. iOS must expose a usable network route through the intended phone/accessory/cable combination. A TCP client cannot supply missing OS USB-NCM support.

Use the receiver's newest valid JPEG. Decode upright, apply the documented crop/enhancement without mirroring, create a MediaPipe `MPImage`, and invoke the supplied detector on one serial background queue. Keep a monotonically increasing local millisecond detector timestamp and use the same clock in seconds for temporal decisions. Preserve all candidate hands and their matching world landmarks/handedness. Recreate/reset detector and decision state on a new stream as required by MediaPipe's timestamp contract.

Call `GestureDiagnosticEngine.observe(...)` for a candidate using normalized XY, matching world XYZ, confirmed handedness/confidence and **the actual prepared image width/height**. The returned probabilities and known mass are diagnostic values; they do not authorize an app action.

## 4. Finish and validate the action path

Implement [RUNTIME_PORTING.md](RUNTIME_PORTING.md) against `ios/reference/backend/` or the extracted desktop snapshot. In particular, preserve two-hand ownership, source-pixel Left/Right sectors, strict low-mass Peace, Open Palm −160°…−20°, Stop inverted −145°…−35°, immediate angle reset, Fist world-depth validation, recovery flags and the release/action latch.

Run every geometry/temporal/transition vector against the port before wiring product handlers. Then verify all 12 active commands on physical devices. `dorsal` never dispatches an action. Dispatch only the final accepted and latched command; raw argmax and the angle helper are insufficient.

The app team owns signing, UI, permission/lifecycle handling, full preprocessing/recovery/selection/stabilization, current geometry/temporal/action port, accessory support and product action handlers. Record the evidence in [VALIDATION.md](VALIDATION.md).
