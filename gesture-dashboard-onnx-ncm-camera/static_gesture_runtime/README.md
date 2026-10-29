# Static gestures for the iOS NCM runtime

This handover puts the existing twelve static commands into the same Flutter,
LiteRT and tracked MediaPipe detector setup supplied with the dynamic runtime.
The production ONNX model and Windows application remain available unchanged.
No training or Boss-app data is required to integrate this runtime.

The complete Boss handover is generated as
`handover_static/Boss_Static_Gesture_Handover_20261029_fix1.zip`. Start with
its `START_HERE.md`. It includes this latest runtime, the iOS example, current
Flutter tester source, the verified Android test APK and integrity checks.
Earlier `candidate` and `overlay` archives are superseded by that bundle.

The FP32 TFLite conversion preserves the current learned model: all 18,073
validation/test rows keep the same top label and scalar accept/reject result.
The largest probability difference is approximately 0.0000014. This measures
conversion parity, not camera accuracy or physical iPhone performance.

The Dart feature/geometry port also matches all 1,142 exported Python reference
cases in `validation/dart_geometry_parity.json`: feature vectors, adjusted
probabilities, final pose validity and geometry support. These are synthetic
and reconstructed landmark regressions, rather than 1,142 new camera recordings.
An additional 400 held-out landmark cases cover all twelve commands plus hard
negatives, with matching features and decisions. These cache cases do not have
original camera dimensions or metric world depth; their scope is recorded in
`validation/dart_cache_geometry_parity.json`. All 5,710 canonical known-mass
qualification cases also match the reference.

The INT8 dense-layer candidate is included for inspection, but failed the
predeclared 99.5% top-label-agreement gate on validation. Its manifest prevents
the runtime from loading it. FP32 is the recommended default. The quantized
candidate retains floating-point normalization and output calculations; it is
not described as a fully integer model.

## Which TFLite model to use when

| Model | Use case | Current selection |
|---|---|---|
| `flutter/assets/models/gesture_static_float32.tflite` | Static gestures on iOS/Android, with either phone or NCM camera input | Use for all current integrations; qualified default |
| `flutter/assets/models/gesture_static_int8.tflite` | Inspection and future quantization development | Failed qualification and blocked from loading; not a current app option |
| `flutter/third_party/hand_detection/assets/models/hand_detection.tflite` and `hand_landmark_full.tflite` | Detect hands and produce landmarks before classification | Use the bundled FP16 pair together, or reuse the compatible existing detector |

Use `StaticTfliteClassifier(model: StaticTfliteModel.float32)`; the constructor
default makes the same choice. A GPU/CPU detector choice does not change this
static classifier selection. The static classifier uses a persistent CPU
LiteRT worker. Retain each model's matching manifest and the complete runtime
code, including the thumb-direction and hand-surface fixes.

INT8 can replace FP32 only in a future release that passes qualification and
target-device acceptance with updated matching manifests. The supplied
inference plugin is `flutter_litert` 3.9.4. That dependency version, model
precision and the Android tester's 2.0.1 release number describe different things.

## Package contents

- `flutter/`: standalone path dependency `static_gesture_runtime`, model assets,
  the bundled detector, controller, tests and an NCM integration example.
- `flutter/lib/static_gesture_preview.dart`: reusable preview with the original
  dynamic detector's boxes, 21 landmark dots and skeletons for both hands.
  The example displays this overlay by default.
- `contracts/`: detector contract and exact upstream source/asset hashes.
- `validation/`: model conversion results and Python reference fixtures.
- `tools/verify_static_runtime.py`: portable model/hash verification.
- `SHA256SUMS.json`: payload hashes generated when the archive is assembled.

The upstream file is named `dynamic_gesture_runtime_api_v10_2_20261025.zip`,
although its actual contents identify v10.3. `detector_provenance.json` records
that discrepancy and the supplied archive hash.

## Gesture contract

| Label | Action |
|---|---|
| `left` | Move Left |
| `right` | Move Right |
| `up` | Move Up |
| `down` | Move Down |
| `open_palm` | Enable / Disable Object Tracking |
| `like` | Play / Pause |
| `ok` | Start / Stop Recording |
| `fist` | Mute |
| `thumb_down` | Volume Down |
| `peace` | Turn 90 Degrees |
| `rock` | Backup |
| `stop_inverted` | Return to main position |

The classifier still exposes thirteen probabilities. `dorsal` stays in slot six
as rejection evidence. `no_gesture` is the runtime rejection result. The source
MLP has fourteen internal outputs, including its trained unknown class; the
export preserves the thirteen conditional probabilities and the separate
`known_gesture_mass`. It does not discard the unknown-class evidence.

Input is `float32 [1,76]`. Output is `probabilities [1,13]` and
`known_gesture_mass [1,1]`. Normalization is embedded in the model, so the app
must not standardize the features again. The feature extractor operates on XY;
metric world XYZ is additionally required for live Fist geometry. Image Z is
not a substitute for metric world Z.

## Runtime behavior

NCM JPEGs enter a single latest-frame slot. New pending images replace older
pending images. Inference starts no faster than once every 100 ms. The detector
decodes a JPEG once, tracks up to two hands and uses the original supplied GPU
factory with explicit CPU fallback. Classifier inference uses a persistent CPU
LiteRT interpreter in a background isolate. Initialization is not repeated per
frame. The detector's selected backend is observable.

Each physical hand has independent confirmation and duplicate state. A strong
raw model prediction can confirm immediately at confidence 0.92 and margin
0.25, after known-mass and geometry checks. The ordinary path uses two matching
results among the latest three. A repeat of the same emitted command requires
confirmed `no_gesture`; there is no fixed 800 ms action cooldown. A missing hand
clears confirmation evidence after 350 ms without automatically rearming an
emitted command. An application reset after reconnect/orientation change clears
the session. At most one command is dispatched for a source frame.

Strict geometry recoveries keep their longer hold requirements. A high
confidence score does not override a forbidden surface, downward Fist or
open-hand angle. Open Palm retains the source-image XY band -160 to -20
degrees; Stop inverted retains -145 to -35 degrees.

Like requires an upward thumb above the folded fingertips; Thumb Down requires
a downward thumb below them. The 3-D thumb-only recovery uses camera-pixel
direction to correct either model label, including the reported side view with
a 98.3% Thumb Down score. Ambiguous sideways views remain rejected.

This is a static runtime: there is no artificial 500 ms sequence requirement.
An 8-step temporal model would need different inputs and newly trained weights.

## Integration and coordinates

See `flutter/example/README.md` and `IOS_INTEGRATION.md`. Keep the package with
its `third_party/hand_detection` directory. The example requires the iOS target
to be created on a Mac before it can be built in Xcode.

All detector images and classifier landmarks must use the same upright,
unmirrored NCM coordinate system. The NCM board timestamp is diagnostic only:
local monotonic receive time drives scheduling and confirmation. It does not
measure exposure-to-receive latency because the board and phone clocks are not
synchronized.

The adapter converts detector handedness to physical identity once. Surface
geometry uses that same identity directly: a visible palm is Open Palm, and
the back of an upright open hand is Stop inverted. A second swap previously
reversed these commands in both the tester and runtime; this release removes it.
When `modelInputMirrored` is true, the adapter also unmirrors image/world X
before geometry. Callers must not add a second swap. The existing semantic mapping
for directions is retained: positive index screen X resolves to `left`.
Do not compensate by mirroring pixels or exchanging command labels in the app.

## Rebuild from the original repository

Use an isolated Python environment with `requirements-static-conversion.txt`.
The conversion uses an explicit FlatBuffer graph of standard TFLite operations;
it does not require TensorFlow, Flex operators or a training run.

```powershell
python scripts/prepare_static_runtime_vendor.py
python scripts/convert_static_tflite.py
python scripts/export_mobile_parity.py static_gesture_runtime/validation/python_reference.json
cd static_gesture_runtime/flutter
flutter pub get
flutter analyze
flutter test
```

Run `scripts/package_static_runtime.py` from the repository root after all
checks. Verify the extracted archive with:

```powershell
python -m pip install -r tools/requirements-verification.txt
python tools/verify_static_runtime.py --package-root .
```

## Qualification boundary

The included report states the checks actually run. A Windows host cannot
compile an iOS application or establish NCM/iPhone latency. The app team must
run the example in a release build on the target iPhone, verify NCM networking,
both physical hands, all twelve commands, neutral release and repeat behavior,
and record detector/processing latency. No images or private Boss-app recordings
need to be shared.

The new two-hand controller intentionally differs from the previous selected-
owner desktop controller. Model conversion parity and geometry fixture parity
do not mean the two controllers have identical event timing.

The preview pairs each completed JPEG with its own raw detector results.
Detected hands remain visible when geometry rejects a gesture; a drawn box
does not mean a product command was accepted. The overlay uses upright,
unmirrored source coordinates and preserves aspect ratio. No additional
detector/classifier invocation is needed to draw it.
