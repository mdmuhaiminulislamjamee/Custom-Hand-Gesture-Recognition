# Static Gesture Runtime - final app-team handover

Give the app team this entire `static_gesture_runtime` folder. It contains the
model AND the code needed to reproduce the delivered recognition behavior.
The Flutter tester app and its APK are not part of this package.

## What the app team should integrate

Add **`flutter/` as one Flutter path dependency** in the existing app. Keep its
`lib/`, `assets/`, `third_party/`, pubspec and lockfile together. The runtime
loads the correct model assets automatically. Do not select a loose TFLite file
and recreate feature extraction or hand selection separately.

The production gesture model is:

**`flutter/assets/models/gesture_static_float32.tflite`**

The included INT8 candidate is disabled. Keep the supplied FP32 default.

## Read in this order

1. [Component READMEs](component_readmes/README.md): short, separate instructions
   for each part. For NCM, follow 01 through 05 in order.
2. [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md): add the dependency, choose ONE
   camera-input route, connect `onCommand` to the app's actions, and manage lifecycle.
3. [MODEL_GUIDE.md](MODEL_GUIDE.md): model locations and which assets to retain.
4. [PERFORMANCE.md](PERFORMANCE.md): preserve the existing speed and behavior.
5. [DEVICE_ACCEPTANCE.md](DEVICE_ACCEPTANCE.md): check the integrated app on devices.

## What is already included

- All twelve static gestures, feature extraction, qualified FP32 classifier,
  geometry and confidence/known-mass checks.
- Bundled hand detector, 21-landmark model and existing face filtering.
- Stable two-hand identities and selected-hand ownership. A second hand merely
  appearing beside the selected hand does not take control. Switching to a
  closer hand, or the only valid peer gesture, requires confirmation.
- Per-hand gesture confirmation and duplicate-command prevention.
- NCM JPEG receiving, latest-frame scheduling, telemetry and the corrected
  preview that keeps each image and its matching overlay together during decode.
- Reusable phone-camera helpers, a small NCM integration example, validation
  fixtures, dependency lockfiles, licenses and portable integrity verification.

Further ear/ghost-detection tuning and movement suppression are deferred.
This delivery contains the existing filters and the implemented ownership/preview fixes.

## Folder map

| Location | What to do with it |
|---|---|
| `flutter/` | Integrate this complete package into the app |
| `component_readmes/` | Short, separate instructions for each integration part |
| `flutter/example/` | Follow this minimal NCM integration example; no tester APK is required |
| `integration/phone_camera/` | Optional reusable helpers when using the phone camera |
| `IOS_INTEGRATION.md` | Detailed coordinate, permission and iOS integration reference |
| `contracts/` | Exact detector settings and provenance |
| `validation/` | Automated checks and model/geometry fixtures |
| `reference/source_model/` | Matching source ONNX/configuration for reference; not loaded by the app |
| `reference/qualified_release/` | Original test evidence for the unchanged executable code |

## Verify the handover

From this folder, run Python 3.10 or later:

```sh
python verify_handover.py
```

The verifier checks every delivered file, all model assets, the preserved runtime
and the reusable phone helpers. No Python installation is required inside the app.

The runtime and its assets are byte-for-byte the tested 14 November 2026
ownership/preview release. This handover changes packaging and documentation,
not inference or timing. The app team must build/sign its iOS app on a Mac and
complete physical device acceptance. No IPA or native iOS build is claimed.
