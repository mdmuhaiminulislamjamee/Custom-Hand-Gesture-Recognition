# 01 - Add the runtime package and models

## What this part does

`flutter/` is the complete Flutter package. It contains the gesture classifier,
feature extraction, hand selection and command confirmation. Its assets contain
the production FP32 gesture model. `flutter/third_party/hand_detection/` contains
the hand detector, landmark and face-filter models. The runtime loads these
assets through its package configuration.

## What the app team integrates

1. Put the entire `static_gesture_runtime/` folder beside the Flutter app (or
   adjust the paths below). Preserve `flutter/lib`, `flutter/assets`,
   `flutter/third_party`, its pubspec and the supplied lockfiles together.
2. Add these entries to the app's `pubspec.yaml`, merging with its existing
   dependencies and hooks:

```yaml
dependencies:
  static_gesture_runtime:
    path: ../static_gesture_runtime/flutter
  hand_detection:
    path: ../static_gesture_runtime/flutter/third_party/hand_detection

dependency_overrides:
  hand_detection:
    path: ../static_gesture_runtime/flutter/third_party/hand_detection

hooks:
  user_defines:
    dartcv4:
      parallel_jobs: 1
      include_modules:
        - imgcodecs
        - imgproc
```

3. Run `flutter pub get` in the app. Keep native builds enabled on devices and
   review the app's resolved lockfile against
   `reference/qualified_release/dependency_versions.json`. Start with Flutter
   3.47.8 / Dart 3.13.7, the tested toolchain.
4. Use `StaticTfliteModel.float32` when creating the runtime. The INT8 model is
   present only as an unqualified candidate; do not enable it in production.

The app does not need to copy or convert the source ONNX model. A loose TFLite
file by itself does not include detection, geometry, ownership or confirmation.
See [MODEL_GUIDE.md](../../MODEL_GUIDE.md) for each asset's location.
