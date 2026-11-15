# Optional built-in phone camera integration

This folder contains only two reusable services, copied unchanged from the
qualified release. It contains no tester screens, Android app or APK:

- `phone_static_gesture_engine.dart`: persistent detector/classifier/adapter
  front end; returns the shared `StaticGestureFrameResult`.
- `phone_frame_gate.dart`: admits callbacks at up to 10 FPS and drops callbacks
  while inference is busy. It prevents an unbounded camera-frame queue.

If the host already runs a compatible detector, use Route A in
[../../INTEGRATION_GUIDE.md](../../INTEGRATION_GUIDE.md) instead of starting
this engine. These helpers are an alternative input route.

## Add to the app

Copy both Dart files into the same app service directory. Add the dependencies
from the main guide, including direct `camera: 0.12.0+2` and `hand_detection`.
The helpers' imports use only the public packages, not the tester app.

Initialize one engine and gate outside the camera callback:

```dart
final gate = PhoneFrameGate();
var acceptingFrames = true;
final engine = PhoneStaticGestureEngine(
  onCommand: (event) {
    if (!acceptingFrames) return;
    // Dispatch event.label through the app's existing action handler.
  },
);
await engine.initialize();
```

The app owns the `CameraController`. Use audio disabled and the source formats
from `preferredImageFormats` in
`package:static_gesture_runtime/detector/camera_frame_adapter.dart`:
BGRA8888 on iOS; NV21 or the supported YUV420 fallback on Android. Preserve the
current input resolution initially. Higher resolution can increase latency.

Inside the camera image callback, use the gate BEFORE invoking the engine:

```dart
if (!acceptingFrames) return;
final ticket = gate.begin();
if (ticket == null) return;
try {
  final frame = await engine.analyze(
    image,
    rotationDegrees: rotationDegrees,
    modelInputMirrored: modelInputMirrored,
  );
  if (!acceptingFrames || !gate.isCurrent(ticket)) return;
  // Display frame.result.selectedHand; dispatch only onCommand above.
  // frame.detections supplies the raw hand overlay; frame.width/height
  // are the upright dimensions to use for drawing it.
} catch (error) {
  if (acceptingFrames && gate.isCurrent(ticket)) {
    await engine.invalidateFrame();
    // Clear the saved result/overlay and surface the error in the host UI.
  }
} finally {
  gate.finish();
}
```

`image` is the current `CameraImage`. `rotationDegrees` must describe the
sensor buffer's rotation into upright source coordinates; use the camera
description's sensor orientation and current device orientation:

```dart
const orientationDegrees = <DeviceOrientation, int>{
  DeviceOrientation.portraitUp: 0,
  DeviceOrientation.landscapeLeft: 90,
  DeviceOrientation.portraitDown: 180,
  DeviceOrientation.landscapeRight: 270,
};
final device = orientationDegrees[controller.value.deviceOrientation] ?? 0;
final sensor = controller.description.sensorOrientation;
final front = controller.description.lensDirection == CameraLensDirection.front;
final rotationDegrees = front
    ? (sensor + device) % 360
    : (sensor - device + 360) % 360;
final modelInputMirrored = isDetectorInputMirrored(
  isIOS: Platform.isIOS,
  lensDirection: controller.description.lensDirection,
);
```

Imports for this calculation are `dart:io`, `package:camera/camera.dart`,
`package:flutter/services.dart` and the runtime's `camera_frame_adapter.dart`.
The supplied AVFoundation front-camera buffers are mirrored; Android CameraX
analysis buffers are not. UI preview mirroring is a separate display transform.
Do not apply a second reflection to model landmarks or swap labels manually.

## Pause, switch camera, resume and close

Serialize camera operations in the host app. On pause/background or before
switching cameras, set `acceptingFrames = false`, call `gate.invalidate()`,
await `engine.invalidateFrame()`, stop the image stream and clear the saved
result/overlay. Dispose/recreate the camera controller when required by the
camera plugin. Do not start a second detector while the first is in flight.

Before resuming a fresh stream or changing orientation/mirror conventions,
await `engine.reset()`, then admit callbacks again. Keep an explicit user pause
state when handling app background/resume. The engine resets timestamps and
both-hand ownership/evidence; the gate drops busy callbacks until prior work exits.

When the view/session is permanently disposed, stop callbacks, invalidate the
gate, clear the UI and await `engine.close()` plus camera-controller disposal.
Make late host callbacks check their session generation/mounted state as well.

For performance comparisons, use release builds and the same resolution,
delegate, orientation, preview and dependency versions. The copied helpers
add no new behavior relative to the qualified release. Further incidental-motion
and ear/ghost-filter tuning is not part of this package.
