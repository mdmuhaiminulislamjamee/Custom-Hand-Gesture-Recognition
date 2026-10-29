# How to use the static gesture handover

Use `Boss_Static_Gesture_Handover_20261029_fix1.zip` as a complete Flutter
integration package. Extract it and start from the folder
`Boss_Static_Gesture_Handover_20261029/`.

**The `.tflite` model alone is insufficient to reproduce this handover's
recognition and command behavior.** Integrate the supplied runtime code with
its model assets and detector, or reuse the app's existing compatible detector
through the supplied adapter.

## Why the model alone is insufficient

The static model does not accept a camera image or a raw `21 x 3` landmark
array. It accepts a `float32 [1, 76]` feature vector built by
`feature_extractor.dart`. It returns thirteen class probabilities and a
separate `known_gesture_mass` score. These are inputs to the runtime's decision,
rather than permission to send a product command.

The complete path is:

```text
Phone camera or NCM JPEG
  -> hand detector: 21 image landmarks, world landmarks and handedness
  -> detector adapter: consistent coordinates and physical Left/Right
  -> feature extractor: the model's exact 76 input values
  -> FP32 TFLite classifier: probabilities + known-gesture mass
  -> geometry: validate shape, surface, direction and rejection
  -> command controller: confirm, suppress duplicates and manage release
  -> onCommand callback
  -> the consuming app's existing action/transport handler
```

For example, the learned model can strongly predict Thumb Down while the
visible thumb points upward. The corrected `geometry_policy.dart` resolves
that supported pose to Like. The Open Palm/Stop Inverted correction also lives
in the adapter/geometry coordinate handling. Copying only the model file does
not include either correction.

The model's normalization is embedded in the TFLite graph. Use the supplied
feature extractor and do not standardize its output a second time.

## Which TFLite model to use, and when

**Use `gesture_static_float32.tflite` for every current static integration:**
the Boss iOS app, Android tester, phone camera and NCM camera. The camera source
does not require a different static classifier.

| Model / version | When to use it | Status in this handover |
|---|---|---|
| `gesture_static_float32.tflite` — FP32, 78,960 bytes | All current static gesture recognition; use with the complete runtime | Qualified default; used by the supplied tester |
| `gesture_static_int8.tflite` — INT8 dense-layer candidate, 23,240 bytes | Inspect quantization results or develop a future improved candidate | Failed qualification; blocked from loading. Do not select it for the current app |
| `hand_detection.tflite` + `hand_landmark_full.tflite` — FP16 detector weights | Convert camera images into hand landmarks before static classification | Bundled detector pair; use together, or reuse the compatible detector already running in the app |

The two `gesture_static_*` files and their manifests are in
`static_gesture_runtime/flutter/assets/models/`. The detector pair is in
`static_gesture_runtime/flutter/third_party/hand_detection/assets/models/`.
Detector FP16 and classifier FP32 describe different stages; they are used
together. Selecting GPU or CPU for detection does not require exchanging the
static FP32 model for INT8. The supplied static classifier uses its persistent
CPU LiteRT worker; the detector requests GPU with CPU fallback.

Select the current classifier explicitly with:

```dart
StaticTfliteClassifier(model: StaticTfliteModel.float32)
```

`StaticTfliteClassifier()` makes the same choice. The INT8 candidate is smaller,
but missed the required 99.5% validation label agreement and increased unknown
false acceptance beyond its allowed limit. It also retains floating-point
normalization/output operations. A future INT8 release must pass model
qualification and target-device acceptance with matching manifests before it
can replace FP32; changing its qualification flag is not validation.

Keep the classifier's matching manifest beside its model. Keep both detector
assets with the bundled detector package. The `fix1` handover includes the
thumb/surface corrections in runtime code, so retain that code with the model.

The inference dependency version is **`flutter_litert` 3.9.4**, as recorded in
the supplied pubspec/lockfiles. **GestureLab 2.0.1** identifies the Android
tester release; **FP32/FP16/INT8** identify model precision and are separate
from the dependency/app version numbers.

## What each part of the ZIP does

| Path inside the extracted folder | Purpose | How to use it |
|---|---|---|
| `static_gesture_runtime/flutter/` | Complete reusable static recognition package | Integrate this package into the Flutter app |
| `static_gesture_runtime/flutter/third_party/hand_detection/` | Bundled MediaPipe detector code and palm/landmark models from the supplied dynamic runtime | Keep it with the runtime; reuse this detector if the app already runs it |
| `static_gesture_runtime/flutter/assets/models/gesture_static_float32.tflite` | Qualified default static classifier | Loaded by `StaticTfliteClassifier` |
| `static_gesture_runtime/flutter/assets/models/gesture_static_float32_manifest.json` | Model hash, input/output contract and qualification information | Retain it beside the model; the loader validates the contract |
| `static_gesture_runtime/flutter/assets/models/gesture_static_int8*` | Unqualified quantization experiment | Retained for inspection; the runtime blocks it from loading |
| `static_gesture_runtime/flutter/example/` | NCM integration example with a two-hand preview | Use as a reference or create its iOS host on a Mac |
| `flutter_gesture_tester/` | Android diagnostic app source using the same static runtime | Use to inspect phone-camera/NCM recognition |
| `flutter_gesture_tester/releases/GestureLab-v2.0.1-litert-ncm-arm64-release.apk` | Ready-built Android tester | Install on a compatible ARM64 Android API 24+ device; it displays commands |
| `static_gesture_runtime/IOS_INTEGRATION.md` | Detailed app integration and coordinate contract | Read when connecting the runtime to the existing iOS app |
| `static_gesture_runtime/validation/` and `flutter/test/` within the runtime | Model/geometry fixtures, reports and automated tests | Use to check a port or a later change |
| `contracts/detector_provenance.json` within the runtime | Provenance and hashes of the supplied detector | Identifies the exact detector delivery |
| `HOST_CHECKS.json`, `static_gesture_runtime/VALIDATION.md` | What was checked and what remains to be measured on devices | Review before product acceptance |
| `DEVICE_ACCEPTANCE.md` | Physical iPhone/NCM acceptance procedure | Complete on each supported device/setup |
| `release_manifest.json`, `SHA256SUMS.json`, `verify_handover.py` | Release identity and payload integrity | Verify the extracted delivery before integration |

The Android APK is a test build signed with the Android debug certificate.
The iOS delivery is source; the owning app team builds and signs its iOS host
on a Mac. The example and tester do not implement the Boss app's action
transport.

## Which runtime file does what

Paths in this table are relative to `static_gesture_runtime/flutter/lib/`.

| File | Responsibility |
|---|---|
| `static_gesture_runtime_api.dart` | Public entry point for an app that already has detector results |
| `src/runtime_api.dart` | Runs feature extraction, classification, geometry and per-hand command state; exposes `processFrame`, `reset`, `close` and `onCommand` |
| `src/detector_adapter.dart` | Converts bundled detector results into runtime frames; normalizes image/crop-world coordinates, mirror parity and physical hand identity |
| `src/feature_extractor.dart` | Produces the exact 76-value input expected by the model |
| `src/tflite_classifier.dart` | Loads the qualified model and keeps a persistent LiteRT worker/interpreter |
| `src/model_manifest.dart` | Checks the model manifest, hashes, features, labels and qualification |
| `src/geometry_policy.dart` | Validates shape, palm/back surface, thumb direction, orientation and rejection; contains the recent gesture corrections |
| `src/command_controller.dart` | Confirms gestures, suppresses held-command repeats and manages release separately for each hand |
| `src/frame_scheduler.dart` | Keeps one active frame and at most one pending frame; replaces stale pending work |
| `ncm_static_gesture_runtime.dart` | Optional complete NCM front end: detector, adapter and runtime; admits inference at most every 100 ms |
| `src/ncm_client.dart` | NCM TCP/JLIP parsing, JPEG framing, heartbeat handling and reconnect |
| `detector/hand_detector_factory.dart` | Creates the bundled detector with GPU preference and CPU fallback |
| `static_gesture_preview.dart` | Displays the paired JPEG and original detector boxes/landmarks; drawing a hand does not mean a command was accepted |
| `src/telemetry.dart` | Reports processing counts and timing for diagnostics |

## Step 1: extract and verify

Keep the extracted directory structure together. On Windows, use a short path
for Android builds to avoid native toolchain path-length limits.

With Python 3.10+ installed, run from the extracted root:

```sh
python verify_handover.py
```

The result must contain `"passed": true`. This checks payload hashes, the
detector provenance and model consistency between the runtime and APK. For
native model and Flutter checks, follow
[`static_gesture_runtime/HOST_TESTING.md`](static_gesture_runtime/HOST_TESTING.md).

## Step 2: choose the integration path

| The consuming app already has... | Use |
|---|---|
| A running compatible dynamic hand detector returning `List<Hand>` | `StaticDetectorAdapter` + `StaticGestureRuntimeApi`; share the existing camera/detector |
| An NCM stream but no hand detection pipeline | `NcmStaticGestureRuntime`, which supplies the NCM client and detector |
| A native Swift-only implementation | Embed a Flutter module or port the complete feature/geometry/controller contracts and pass the shared fixtures |
| Only a need to inspect recognition on Android | Install the supplied tester APK |

When the app already runs the dynamic detector, pass its existing hand results
to the static runtime. Do not start another detector for the same image.

## Step 3: add the package to the Flutter app

For an app stored beside `static_gesture_runtime/`, add the following to the
app's `pubspec.yaml`, adjusting paths for its actual folder layout:

```yaml
dependencies:
  static_gesture_runtime:
    path: ../static_gesture_runtime/flutter
  hand_detection:
    path: ../static_gesture_runtime/flutter/third_party/hand_detection

dependency_overrides:
  hand_detection:
    path: ../static_gesture_runtime/flutter/third_party/hand_detection
```

Keep the supplied `third_party` directory and model assets with the package.
The override ensures an existing app does not resolve a different detector.
Use the qualified Flutter 3.47.8 / Dart 3.13.7 toolchain and retain the
lockfiles. Merge the native OpenCV hook settings documented in
[`IOS_INTEGRATION.md`](static_gesture_runtime/IOS_INTEGRATION.md), then run
`flutter pub get` in the consuming app.

## Step 4A: reuse the existing detector

Import the public API:

```dart
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';
```

Create these once for the active camera session, and initialize the runtime
inside the app's asynchronous startup method:

```dart
final clock = Stopwatch()..start();
final adapter = StaticDetectorAdapter();
final runtime = StaticGestureRuntimeApi(
  classifier: StaticTfliteClassifier(), // qualified FP32 default
  onCommand: (event) {
    // Connect this event to the app's existing command handler.
    // event.label identifies the gesture; event.action describes its action.
    print('${event.hand.name}: ${event.label} -> ${event.action}');
  },
);
await runtime.initialize();
```

Inside the existing detector callback, submit each fresh result:

```dart
final frame = adapter.frame(
  timestampMs: clock.elapsedMicroseconds / 1000.0,
  width: uprightWidth,
  height: uprightHeight,
  detections: rawDetectorHands,
  modelInputMirrored: false, // ordinary upright, unmirrored NCM input
);
final result = await runtime.processFrame(frame);
```

`rawDetectorHands` is the bundled detector's `List<Hand>`. `uprightWidth` and
`uprightHeight` are the dimensions used by those detections. Timestamps must
increase monotonically. Submit empty detections too; do not reuse old
landmarks when detection misses. A superseded frame can return `null`.

Set `modelInputMirrored` from the image actually fed to the detector, rather
than whether the UI preview is mirrored. The adapter corrects handedness and
unmirrors image/world X once. Do not swap Left/Right or exchange Open Palm and
Stop Inverted labels again in the app.

Supply the detector's metric world landmarks. Normalized image Z is not a
substitute; live Fist validation requires world depth. Use the adapter to
handle the bundled detector's crop-space world coordinates.

On reconnect or a coordinate/orientation change, call `adapter.reset()` and
`runtime.reset()` before submitting new frames. During disposal, await
`runtime.close()`.

## Step 4B: let the package own the NCM connection

Use this alternative when the app does not already own a detector/stream:

```dart
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';
import 'package:static_gesture_runtime/ncm_static_gesture_runtime.dart';

final runtime = StaticGestureRuntimeApi(
  classifier: StaticTfliteClassifier(),
  onCommand: (event) {
    // Route accepted commands to the existing product handler here.
    print('${event.label} -> ${event.action}');
  },
);
final engine = NcmStaticGestureRuntime(
  runtime: runtime,
  host: '192.168.50.2',
  port: 5000,
  onPreview: (frame) {
    // Save this paired frame for StaticNcmPreview in the widget tree.
  },
);
await engine.start();
```

Use `await engine.stop()` when leaving the active session and
`await engine.close()` when disposing the engine. The example shows preview
updates, error handling and lifecycle cleanup.

On iOS, add `NSLocalNetworkUsageDescription` to the host app's Info.plist and
grant local-network permission. The board/phone must provide a working NCM
network route. Add camera permission separately if the app also uses the
phone camera. Follow
[`flutter/example/README.md`](static_gesture_runtime/flutter/example/README.md)
for creating and running the iOS example on a Mac.

## Step 5: connect accepted commands to product actions

Dispatch from `onCommand`. Raw model scores and the preview overlay are
diagnostics; triggering actions from them bypasses geometry, rejection and
duplicate suppression.

| Accepted label | Established action |
|---|---|
| `left` / `right` | Move Left / Move Right |
| `up` / `down` | Move Up / Move Down |
| `open_palm` | Enable / Disable Object Tracking |
| `like` | Play / Pause |
| `ok` | Start / Stop Recording |
| `fist` | Mute |
| `thumb_down` | Volume Down |
| `peace` | Turn 90 Degrees |
| `rock` | Backup |
| `stop_inverted` | Return to main position |

`no_gesture` produces no action. The retired `dorsal` output is retained only
as rejection evidence. Both physical hands have independent state, and the
runtime dispatches at most one command per source frame. Holding an emitted
gesture does not repeatedly trigger it; a confirmed neutral/no-gesture release
rearms it.

## Step 6: check the integration on the target device

Complete [`DEVICE_ACCEPTANCE.md`](DEVICE_ACCEPTANCE.md). In particular, check
both physical hands, upright coordinates, reconnects and these corrected poses:

- Palm facing the camera: **Open Palm**.
- Back of the upright open hand facing the camera: **Stop Inverted**.
- Extended upward thumb above folded fingertips: **Like**.
- Extended downward thumb below folded fingertips: **Thumb Down**.

The handover passed 48 runtime tests, 2 tester tests, 1,142 canonical geometry
cases and 400 additional held-out landmark cases. These are host/landmark
checks. Physical iPhone/NCM accuracy, iOS compilation and sustained performance
remain checks for the consuming app team.

## Common integration mistakes

| Symptom | Check |
|---|---|
| Only copying the model gives different results | Integrate the adapter, exact feature extractor, geometry and command controller as well |
| Open Palm and Stop Inverted are reversed | Use the current adapter/geometry together and remove any extra hand/label swap in the caller |
| Upward thumb still resolves to Thumb Down | Confirm the app uses this handover's `geometry_policy.dart`, rather than only its model |
| Fist is always rejected | Confirm valid metric world landmarks reach the runtime |
| A held command fires repeatedly | Dispatch through `onCommand`, rather than per-frame scores |
| Preview has a hand box but no command | Inspect geometry/known-mass diagnostics; a detector result can be rejected |
| NCM has no frames | Check the OS network route, board TCP service, permissions and JLIP connection diagnostics |

For release changes see [`RELEASE_NOTES.md`](RELEASE_NOTES.md). For model and
test evidence see [`static_gesture_runtime/VALIDATION.md`](static_gesture_runtime/VALIDATION.md).
