# Integrating the static runtime in an existing iOS app

Add the delivered package as a Flutter path dependency:

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

Adjust the relative paths for the consuming application's directory structure.
Keep the override when the app already depends on a different `hand_detection`
release. Otherwise pub may choose a different detector from the supplied source.
Retain the resolved lockfile and record any dependency changes made in the app.

The supplied lockfiles were resolved with Flutter 3.47.8 / Dart 3.13.7. Their
transitive dependencies require at least Flutter 3.44 / Dart 3.13; the broader
minimums in the original detector pubspec are not the effective minimums for
this resolved bundle. Use the qualified toolchain first, then qualify any SDK
or plugin upgrades separately in the consuming app.

The package uses `flutter_litert` 3.9.4 and the full detector assets supplied with
the dynamic archive. The consuming app must initialize the runtime once and
retain it for the active camera session. Call reset after NCM reconnect,
orientation changes or a capture timestamp reset. Await close during disposal
so pending inference cannot dispatch after the view closes.

For an existing dynamic landmark pipeline, use `StaticGestureRuntimeApi` and
submit its landmarks directly. This is the lowest-overhead integration: one
camera, one detector and two gesture classifiers if both vocabularies are used.
The package's NCM wrapper is an alternative when the app does not already own
the NCM stream. Opening two parallel camera/detector pipelines is unnecessary.

For the provided NCM wrapper, display the original detector overlay with the
reusable `StaticNcmPreview` widget:

```dart
import 'package:static_gesture_runtime/ncm_static_gesture_runtime.dart';
import 'package:static_gesture_runtime/static_gesture_preview.dart';

// In State: NcmStaticPreviewFrame? preview;
final engine = NcmStaticGestureRuntime(
  runtime: runtime,
  onPreview: (frame) => setState(() => preview = frame),
);
// In build, place within a width-constrained layout:
// if (preview != null) StaticNcmPreview(frame: preview!)
```

`onPreview` provides the source JPEG, its dimensions, up to two raw detector
`Hand` results and the matching gesture result. The widget reuses the dynamic
package's original `MultiOverlayPainter` for boxes, 21 points and skeletons.
It waits for that JPEG to decode before drawing, preserving unmirrored image
alignment. Clear the saved preview on disconnect/errors and when stopping,
as shown in the example. The existing two-argument `onFrame` callback remains
available for integrations without an overlay.

When the dynamic pipeline already returns the bundled detector's `List<Hand>`,
reuse one persistent adapter rather than manually normalizing or swapping
handedness:

```dart
final adapter = StaticDetectorAdapter(); // exported by the public runtime API
final frame = adapter.frame(
  timestampMs: monotonicTimestampMs,
  width: uprightWidth,
  height: uprightHeight,
  detections: rawDetectorHands,
  modelInputMirrored: false, // upright, unmirrored NCM JPEG
);
await runtime.processFrame(frame);
```

Call `adapter.reset()` with `runtime.reset()` after reconnect or a coordinate
system change. The adapter converts pixel landmarks and crop-space metric
world landmarks together. `modelInputMirrored` controls handedness mapping and
unmirrors image/world X into the runtime's canonical coordinates. Raw detections
remain available for the original preview. Keep NCM JPEG input upright and
unmirrored with the default `false`. BGRA is a pixel format, not a mirror or
rotation convention.

Landmark-only integration:

```dart
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';

final runtime = StaticGestureRuntimeApi(
  classifier: StaticTfliteClassifier(), // qualified FP32 default
  onCommand: (event) {
    // Route event.label to the existing product command handler.
    // event.action contains the established display action.
  },
);
await runtime.initialize();

await runtime.processFrame(StaticGestureFrame(
  timestampMs: monotonicTimestampMs,
  frameWidth: uprightWidth,
  frameHeight: uprightHeight,
  hands: [
    StaticGestureHand(
      hand: StaticPhysicalHand.right,
      handednessConfidence: handednessConfidence,
      normalizedLandmarks: normalizedXYZ, // x/W, y/H, z/W
      worldLandmarks: metricWorldXYZ,     // untouched metric coordinates
    ),
  ],
));
```

Submit a frame for every fresh detector result, including `hands: []`. Do not
repeat the last landmarks during a miss. The runtime accepts at most one
physical left and one physical right observation. Duplicate/uncertain identity
should be resolved by the detector adapter before submission.

The app owns iOS network permissions and the board's network route. Include an
appropriate `NSLocalNetworkUsageDescription` in the application's Info.plist.
An NCM-only integration does not require built-in camera access; add
`NSCameraUsageDescription` only if the app uses that camera separately.

Build the consuming Flutter app on a Mac with its normal CocoaPods or Swift
Package Manager setup and a compatible iOS deployment target. The supplied
example instructions explain creating a temporary host app. iOS compilation
and device execution are required before product acceptance; they cannot be
performed from the Windows workspace used to author this package.

During the first device run, keep command dispatch in the app's diagnostic mode
and record selected detector backend, received JPEGs, dropped pending frames,
processing FPS, and p50/p95 detector/runtime times. A native Swift-only app must
embed the Flutter module or port the contracts and pass the shared fixtures.
The `.tflite` file alone does not include feature extraction or pose geometry.

For the supplied standalone example, the app-level OpenCV hook limits native
compilation to the modules used by the detector. If integrating into an
existing app, merge the following into that app's pubspec and retain any extra
OpenCV modules the app itself uses:

```yaml
hooks:
  user_defines:
    dartcv4:
      parallel_jobs: 1
      include_modules:
        - imgcodecs
        - imgproc
```

Use normal native builds for device testing. `skip_build: true` is only a
temporary host-unit-test setting, as described in `HOST_TESTING.md`.
