# Android integration

This is an NCM/MediaPipe/ONNX integration delivery for an existing Android app. It is not an APK or complete Android Studio project. Kotlin compilation, full decision-port parity and physical Android/NCM acceptance have not been performed on the Windows preparation host.

## 1. Assets and build configuration

Copy the required assets from `android/models/` into `app/src/main/assets/`: ONNX, metadata, runtime JSON and `hand_landmarker.task`. The legacy face model is retained in the bundle but unused by the current hand-only policy.

Copy `android/kotlin/*.kt` into the app source tree and change the example package if needed. Merge `android/AndroidManifest.fragment.xml` and `android/app-build.gradle.kts.fragment` into the existing module. The supplied template declares minSdk 24 and dependencies on MediaPipe Tasks Vision and ONNX Runtime Android. Its dependency placeholders use `latest.release`; resolve and pin tested versions in the app's dependency lock/version catalog before qualification. These Android dependencies have not been built or resolved in this Windows delivery.

## 2. Qualify the NCM route

The intended board route is host `192.168.50.1/30` → board `192.168.50.2:5000`. Confirm OS USB-host/NCM support using the actual product device, firmware and cable. `NcmNetworkSelector` finds an OS-provided Android `Network`; `GestureNcmClient` binds its socket through that network. These classes do not implement USB-NCM drivers or create missing OS support. Follow [NCM_PROTOCOL.md](NCM_PROTOCOL.md) for JLIP, CRC, heartbeat, JPEG and reconnect behavior. Built-in Android camera capture is outside this delivery.

## 3. Run diagnostic inference

`GestureFrameAnalyzer` serializes newest-frame admission, applies the 0.92 center crop, runs two-hand MediaPipe VIDEO detection and classifies each candidate into 13 probabilities plus known mass. It emits diagnostic observations only. `candidates` preserves per-hand image/world landmarks, handedness/confidence and model outputs; `imageWidth`/`imageHeight` describe the actual prepared image. The top-level first-hand fields exist for diagnostics and do not implement gesture ownership.

Use callbacks for observation, missing-hand and error handling. Submit valid NCM frames with `submit(frame)`. Stop the client, clear pending frames and close/reset state on lifecycle or stream changes. Do not wire raw scores directly to tracking, media, recording or movement actions.

## 4. Run feature, angle and ONNX parity

For a debug build copy `android/tests/parity_vectors.json` into debug assets. On a background thread:

```kotlin
val report = GestureParity.run(context)
check(report.maximumError <= 0.0002f)
// Compare case counts with RELEASE_MANIFEST.json; do not hard-code an old count.
```

Feature/ONNX tolerance is 0.0001; angle tolerance is 0.0002°. The helper checks feature values, source-pixel angle ranges and model tensors. It does not yet test a complete native geometry/temporal/action port. Model identity and the complete 13-output order are checked on load.

## 5. Complete the current decision port

Port [RUNTIME_PORTING.md](RUNTIME_PORTING.md) using `android/reference/backend/` and the identical extracted desktop reference. The Swift files under `android/reference/legacy_swift/` are older implementation history, not a current decision oracle.

Required work includes recovery/hand qualification, ownership and stabilization, full current geometry, Fist world-depth rules, all recovery arguments, temporal hold and the action latch. Open Palm accepts only −160°…−20°; Stop inverted accepts −145°…−35°, inclusive. Out-of-range open hands become `no_gesture` immediately and returning needs a fresh hold. Keep Dorsal disabled and preserve Left/Right calibration and strict low-mass Peace recovery.

Compare every `expected` geometry/temporal field and all transition sequences in the supplied vectors, then perform the physical-device acceptance in [VALIDATION.md](VALIDATION.md). Record the tested Gradle/JDK/SDK/NDK/dependency versions, device/OS, accessory route and app action behavior.
