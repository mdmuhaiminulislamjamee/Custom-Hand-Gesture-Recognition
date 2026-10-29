# Boss static gesture handover - 29 October 2026

Read [README.md](README.md) for the step-by-step integration guide, why the
model alone is insufficient, and the responsibility of every main component.

Use this bundle for the latest static gesture integration. It supersedes
`static_gesture_runtime_20261029_candidate.zip` and
`static_gesture_runtime_20261029_overlay.zip` and the earlier
`Boss_Static_Gesture_Handover_20261029_final.zip`.

The `fix1` revision includes the Like/Thumb Down camera-direction correction
and removes the extra handedness swap that reversed Open Palm/Stop Inverted
in the shared tester/runtime. The Android tester is now version 2.0.1.

The delivery preserves the learned static gestures, copies the hand detector
from the supplied dynamic runtime v10.3, and supplies the feature extraction,
geometry, two-hand command controller and preview together. No retraining was
performed. The default is the qualified FP32 TFLite model.

## Start with the iOS integration

1. Read `static_gesture_runtime/IOS_INTEGRATION.md`.
2. Keep `static_gesture_runtime/flutter` and its bundled `third_party` directory
   together. Add the documented path dependencies to the existing Flutter app.
3. If the Boss app already runs the dynamic detector, reuse those raw hand
   results with `StaticDetectorAdapter` and `StaticGestureRuntimeApi`.
   Initialize the static classifier once per active session. Feed each fresh
   detector result, including empty results, and use monotonic timestamps.
4. Connect `onCommand` to the app's existing action handler. The example/tester
   displays commands; it does not implement the product's action transport.
5. Build a release version on a Mac and complete `DEVICE_ACCEPTANCE.md` on the
   supported physical iPhone and NCM board.

The runtime is Flutter/Dart source. A Swift-only app must embed a Flutter module
or port the contracts and pass the fixtures. An iOS binary/IPA is not included;
the iOS host must be built and signed by the app owner on a Mac.

## Included files

| Path | Use |
|---|---|
| `static_gesture_runtime/flutter/` | Main integration package, detector, models, controller, tests |
| `static_gesture_runtime/flutter/example/` | iOS NCM diagnostic example with matching two-hand overlay |
| `static_gesture_runtime/IOS_INTEGRATION.md` | Dependencies, existing-detector adapter, lifecycle, coordinates |
| `static_gesture_runtime/README.md` | Gesture/action mapping and runtime contract |
| `static_gesture_runtime/VALIDATION.md` | Model and geometry parity evidence and qualification limits |
| `static_gesture_runtime/validation/` | Conversion results and shared fixtures |
| `flutter_gesture_tester/` | Current Android Flutter source using the same static runtime |
| `flutter_gesture_tester/releases/GestureLab-v2.0.1-litert-ncm-arm64-release.apk` | Optional Android test app; Android 7/API 24+, 64-bit ARM |
| `RELEASE_NOTES.md` | Final delivery changes and retained behavior |
| `DEVICE_ACCEPTANCE.md` | iPhone/NCM checks and aggregate timing template |
| `release_manifest.json` | Model/APK identifiers, payload metadata and test evidence |
| `SHA256SUMS.json`, `verify_handover.py` | Complete payload integrity verification |

The Android APK is optimized in release mode and signed with an Android debug
certificate for testing. It cannot be installed on an iPhone. The source
contains no production signing keys or machine-specific SDK paths.
For Windows Android builds, extract into a short directory to avoid native
toolchain path-length limits.

## Verify the extracted delivery

Use Python 3.10+; the first check needs only the standard library:

```sh
cd Boss_Static_Gesture_Handover_20261029
python verify_handover.py
```

To execute native model fixtures as well, create an isolated Python environment
and follow `static_gesture_runtime/HOST_TESTING.md`. Flutter 3.47.8 / Dart 3.13.7
was used for the supplied checks; retain the lockfiles. On Mac, create the iOS
host as described in `static_gesture_runtime/flutter/example/README.md`.

## Model and gesture behavior

`gesture_static_float32.tflite` is the production integration default. It
preserves the original model's top labels and scalar gates on all 18,073
validation/test rows. This is conversion parity, not a measurement of new
camera accuracy or iPhone speed.

The INT8 experiment is retained with its failed qualification manifest for
traceability. The runtime refuses to load it. Do not change its manifest to
force it into production.

The twelve command labels and actions are listed in the runtime README.
`dorsal` remains rejection evidence, and `no_gesture` releases/rearms command
state. Both physical hands have independent state; at most one product command
is dispatched from a source frame. Boxes and landmarks display raw detections,
including rejected poses; a visible box does not imply an accepted command.

## NCM setup

The default is TCP `192.168.50.2:5000` with JLIP v1 framed JPEGs. Networking uses
the operating system's route table and does not require an app-level check for
exactly `192.168.50.1/30`. The host/port are configurable in the runtime.
Include `NSLocalNetworkUsageDescription` in the iOS host and grant local network
permission. The phone/board firmware must expose a usable network route.

Keep NCM images upright and unmirrored throughout detection, gesture geometry
and preview. Reset adapter and runtime after reconnect or orientation changes.
Newest pending frames replace stale ones; inference admissions are capped at
10 FPS. The selected detector backend and receiver-to-result timing are
observable. GPU is requested with CPU fallback; device selection and sustained
performance must be checked on the target iPhone.

This delivery has host validation and a verified Android build. Physical iOS
compilation, NCM routing and real camera performance remain device acceptance
steps. No zero-latency or identical end-to-end dynamic-model performance is
claimed.
