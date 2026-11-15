# 06 - Use the phone camera instead of NCM (optional)

## What this part does

`integration/phone_camera/phone_static_gesture_engine.dart` is a reusable
detector/classifier front end for the phone camera.
`phone_frame_gate.dart` admits up to 10 processed frames per second and drops
callbacks while inference is busy. They were copied unchanged from the tested
release. This is an **alternative** input route, not something to run beside
the NCM engine on the same camera feed.

## What the app team integrates

1. Add the complete `flutter/` runtime package as in
   [01 - Runtime package and models](../01_runtime_package/README.md).
2. Copy both helper Dart files into one service directory in the host app and
   add `camera: 0.12.0+2` as a direct app dependency.
3. Own the `CameraController`, camera permission and image stream in the host
   app. Create one `PhoneStaticGestureEngine` and one `PhoneFrameGate` per
   active session. Call the gate before sending a frame to `engine.analyze`.
4. Pass the correct sensor rotation and actual model-input mirror state.
   Display `frame.result.selectedHand`; handle actions only through
   `onCommand`.
5. On pause, camera switch, orientation change or disposal, invalidate old
   callbacks, clear the display and reset/close the engine as described in the
   [phone-camera integration README](../../integration/phone_camera/README.md).

Use the [phone-camera integration README](../../integration/phone_camera/README.md)
for the complete callback, image-format and lifecycle code. The NCM connection
and `StaticNcmPreview` do not apply to this route.
