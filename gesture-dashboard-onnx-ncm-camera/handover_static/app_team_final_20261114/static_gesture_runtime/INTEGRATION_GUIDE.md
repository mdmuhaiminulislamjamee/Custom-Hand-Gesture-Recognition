# Integrate into the existing app

## 1. Add the complete Flutter package

Keep these folders beside each other, or adjust the relative paths below:

```text
workspace/
  my_app/
  static_gesture_runtime/
    flutter/
    ...guides, verification and reference files...
```

Merge into `my_app/pubspec.yaml`:

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

Retain any other OpenCV modules required by your app. Native builds must remain
enabled for device use. The package pubspecs already declare model assets; do
not move/rename them or add another copy under the app's assets.

Use Flutter 3.47.8 / Dart 3.13.7 for the initial integration, matching the tested
lockfiles. The resolved dependencies require at least Flutter 3.44 / Dart 3.13;
the broader minimum in a vendor pubspec is not the effective SDK minimum.
The supplied code uses `flutter_litert` 3.9.4. Keep the supplied package/example
lockfiles and compare the consuming app's resolution against
`reference/qualified_release/dependency_versions.json`. Dependency lockfiles
of path dependencies do not pin the consuming app's entire resolution: review
the app's own lockfile after `flutter pub get`, and qualify any differences.
Do not replace an existing app's entire lockfile without merging its requirements.

If using the optional phone helpers, also add `camera: 0.12.0+2` as a direct
app dependency. Their original host dependency resolution is retained as
`reference/qualified_release/phone_camera_host.pubspec.lock` for comparison.

## 2. Choose exactly one input route

| Existing app setup | Route |
|---|---|
| Already runs the compatible hand detector | A: share its fresh `List<Hand>` results |
| Wants the runtime to own the NCM connection and detector | B: use `NcmStaticGestureRuntime` |
| Uses the built-in phone camera and needs a detector front end | C: use the reusable phone helpers |

All three routes use the same feature extractor, FP32 model, geometry, hand
ownership and command confirmation. Do not run multiple routes for the same
camera feed. A native Swift-only app must embed Flutter or port and requalify
the complete contracts; a TFLite file alone cannot reproduce the behavior.

## Route A: share the existing detector

Import the public API and create these objects once per active session:

```dart
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';

final clock = Stopwatch()..start();
final adapter = StaticDetectorAdapter();
final runtime = StaticGestureRuntimeApi(
  classifier: StaticTfliteClassifier(model: StaticTfliteModel.float32),
  onCommand: (event) {
    // Connect event.label to your existing action/transport handler.
    // event.action is its descriptive action; event.hand is the selected hand.
  },
);
await runtime.initialize();
```

For each fresh detector result, including an empty result:

```dart
final result = await runtime.processFrame(adapter.frame(
  timestampMs: clock.elapsedMicroseconds / 1000.0,
  width: uprightWidth,
  height: uprightHeight,
  detections: acceptedDetectorHands,
  modelInputMirrored: false, // upright, unmirrored NCM image
));
```

The width/height and landmarks must describe the same upright image. Pass the
bundled detector's fresh `List<Hand>` with its metric world landmarks. The
adapter performs image/crop-world normalization and physical handedness once.
Set mirror parity from the actual model input, not the mirrored UI preview.
Do not manually exchange Left/Right or Open Palm/Stop Inverted again.
Image Z cannot replace metric world landmarks for Fist validation.

Share the bundled detector with its existing pixel/face filters. Landmark-only
input cannot reproduce pixel-level face rejection if an older external
detector supplies false candidates. Keep one detector and one frame admission
loop; do not start the NCM wrapper or phone engine in addition to it.

A superseded/invalidated call can return `null`; ignore it. Do not redispatch
the previous result or repeat old landmarks during a detection miss. Submit
`detections: []` so stale hands lose eligibility.

On reconnect, orientation/mirror change or timestamp restart, invalidate old
host callbacks, clear the preview, then call `adapter.reset()` and
`runtime.reset()`. Await `runtime.close()` during disposal.

## Route B: let the package own the NCM stream

Use the existing [flutter/example/](flutter/example/) for the complete minimal
widget/lifecycle example. It uses the same runtime files as production.

```dart
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';
import 'package:static_gesture_runtime/ncm_static_gesture_runtime.dart';
import 'package:static_gesture_runtime/static_gesture_preview.dart';

final runtime = StaticGestureRuntimeApi(
  classifier: StaticTfliteClassifier(model: StaticTfliteModel.float32),
  onCommand: (event) {
    // Call the product's existing handler for event.label here.
  },
);
final engine = NcmStaticGestureRuntime(
  runtime: runtime,
  host: '192.168.50.2',
  port: 5000,
  onPreview: (frame) {
    // Save frame, then rebuild only the necessary UI.
    // Display StaticNcmPreview(frame: frame) in a bounded-width layout.
    // Use frame.result.selectedHand for the displayed gesture.
  },
  onState: (state) {
    // If state != 'connected', clear the saved preview and gesture display.
  },
  onError: (error) {
    // Clear the saved preview/result; show the connection/detector error.
  },
  onTelemetry: (values) {
    // Record aggregate processing times/FPS when measuring performance.
  },
);
await engine.start(); // initializes the persistent runtime and detector
```

Use `await engine.stop()` for pause/background and clear the saved UI result.
Use `await engine.start()` for resume after the stop completes. Serialize these
operations so background/resume events cannot start competing sessions.
Dispose with `await engine.close()`; it owns the supplied runtime's lifecycle.

`onPreview` contains the paired JPEG and its original detector overlay.
`StaticNcmPreview` retains the last decoded image and its own landmarks during
the next decode, then swaps them together. Genuine no-hand frames remove the
overlay when their matching picture arrives. Do not implement a separate
overlay that blanks during every JPEG decode.

If no preview is required, use `onFrame` for results instead of `onPreview`.
The NCM wrapper implements latest-frame scheduling, JPEG parsing, heartbeat,
reconnect and persistent detection. It does not create an OS USB Ethernet route.
The device must provide a route to the board. The tester's Android-specific raw
USB fallback is not part of this runtime package; a host needing such a route
must supply its existing transport through `socketConnector`/`closeTransport`
or establish a standard OS network route.

## Route C: built-in phone camera

Use [integration/phone_camera/README.md](integration/phone_camera/README.md).
It explains the two reusable helpers, callback admission, rotation/mirror
handling, errors and lifecycle. These helpers are unchanged from the tested
release and have no tester UI dependency. Copy them into the host app's service
folder and keep one engine plus one `PhoneFrameGate` per active camera session.
The app owns camera permissions, selection and `CameraController` lifecycle.

## 3. Display the selected hand and dispatch confirmed commands

The runtime returns `StaticGestureFrameResult` with:

- `selectedHand`: the only hand that supplies the public gesture/commands.
  It can be null while choosing or while the owner is missing.
- `switchPending`: handoff confirmation is in progress; commands are paused.
- `selectionReason`: diagnostic ownership status.
- `hands`: per-hand diagnostics; these are not independent action events.
- `command`: the confirmed event for that frame, if any.

For a live gesture display, find the entry in `result.hands` whose `hand`
equals `result.selectedHand`. Do not sort both hands by confidence or use the
first detection as the controlling hand. A null selection means no selected
gesture for that frame.

Connect product actions through **`onCommand` only**. Do not additionally
execute `result.command`, or execute `geometry.label` on every frame: that
would duplicate commands or bypass confirmation. A visible bounding box,
accepted geometry candidate or high model probability alone is not a command.
`event.label` is the stable key for your existing product action handler:

| Label | Intended action |
|---|---|
| `left`, `right`, `up`, `down` | Move Left, Right, Up, Down |
| `open_palm` | Enable / Disable Object Tracking |
| `like` | Play / Pause |
| `ok` | Start / Stop Recording |
| `fist` | Mute |
| `thumb_down` | Volume Down |
| `peace` | Turn 90 Degrees |
| `rock` | Backup |
| `stop_inverted` | Return to main position |

`no_gesture` and the retained `dorsal` model output never dispatch actions.
The product supplies its action/transport implementation; this package does
not operate motors or other product functions on its own.

The first qualified hand keeps control when a second appears beside it.
A consistently closer hand, or the sole valid peer gesture, needs at least
three observations spanning 250 ms before handoff. A transfer clears gesture
histories/latches and requires fresh evidence; no command is sent on the
transfer frame. A missing owner gets 350 ms to return. Simultaneous equal
hands wait for a clear choice. Single-hand confirmation retains its existing
strong/ordinary rules. See [PERFORMANCE.md](PERFORMANCE.md) for timing.

## 4. Platform setup and acceptance

On iOS, build/sign on macOS with Xcode. Resolve the native dependencies using
the app's normal CocoaPods/Swift Package Manager setup, starting at iOS 15 or
the higher target required by the resolved dependencies. Add
`NSLocalNetworkUsageDescription` for NCM. Add `NSCameraUsageDescription` and
grant permission when using the phone camera. No microphone capture is needed;
create the camera with audio disabled. See [IOS_INTEGRATION.md](IOS_INTEGRATION.md).

On Android, keep the app's network permission (`android.permission.INTERNET`)
for NCM. For a phone-camera route, configure/request `android.permission.CAMERA`.
The qualified Android build used ARM64 and minimum API 24. Other targets need
native dependency/device qualification.

Run `python verify_handover.py` before integration. After dependency resolution,
run the host app's analysis/tests, build in release mode and complete
[DEVICE_ACCEPTANCE.md](DEVICE_ACCEPTANCE.md) on the supported devices.
Test every gesture with both hands, ownership transfer, empty frames, preview
alignment/flicker, reconnects, background/resume and sustained speed.
Additional movement suppression and ear/ghost tuning remain deferred.
