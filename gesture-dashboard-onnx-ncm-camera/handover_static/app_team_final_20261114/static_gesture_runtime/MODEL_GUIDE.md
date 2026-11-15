# Which model to use

The supplied runtime automatically loads the qualified production classifier:

```text
flutter/assets/models/gesture_static_float32.tflite
```

Use `StaticTfliteClassifier(model: StaticTfliteModel.float32)` or its default
constructor. Keep the matching `gesture_static_float32_manifest.json` beside it.

| Model | Location | Purpose |
|---|---|---|
| `gesture_static_float32.tflite` | `flutter/assets/models/` | Qualified static gesture classifier; use this |
| `gesture_static_int8.tflite` | `flutter/assets/models/` | Inspection candidate; runtime loading remains blocked |
| `hand_detection.tflite` | `flutter/third_party/hand_detection/assets/models/` | Detect palms/hands |
| `hand_landmark_full.tflite` | Same detector models folder | Produce the 21 hand landmarks; retain the full model |
| `blaze_face_short_range.tflite` | Same detector models folder | Existing face/false-hand filtering |
| `gesture_mlp_production.onnx` | `reference/source_model/` | Matching mobile classifier source snapshot; not used by the app |

All five TFLite assets and their current pubspec declarations are retained.
Do not remove the INT8 asset while keeping pubspec/model-selection code unchanged;
it is disabled by its qualification manifest, not by deleting its file.

## What the gesture model does

The gesture TFLite model accepts a prepared `float32 [1,76]` feature vector.
It returns thirteen gesture probabilities and a separate known-gesture-mass
score. It does not accept camera pixels. Use the supplied detector, adapter,
feature extractor, geometry and ownership/command controller together.
Normalization is already inside the model; do not normalize the 76 features again.

The app loads TFLite assets from the Flutter package. No ONNX runtime, Python,
training dataset or model conversion is required for integration.

## Model identity

Production FP32 SHA-256:

```text
cbd53a5c6874172ec8b16cb07c96c228a3063252f0156a7ee9232b4861d3f192
```

Matching source ONNX SHA-256:

```text
d8a37784b603fc9917ef5635f40c2b20b1032c7898a1fe98b3babc9b33c00044
```

The source snapshot belongs to this deployed mobile model. A newer desktop
model in another repository location must not be substituted into this package.
`python verify_handover.py` checks the model identities and all bundled files.
