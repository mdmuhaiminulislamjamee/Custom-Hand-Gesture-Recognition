# Swift integration sources

Use the final guide in [IOS_INTEGRATION.md](../../IOS_INTEGRATION.md) and the dated bundle produced by `scripts/create_final_handover.py`. These sources now include the 6 October Like/Down changes and current Peace, Dorsal, Fist, OK, low-mass recovery and tracking-gap rules.

The five root Swift files implement the 76-feature extractor and post-landmark decisions. `ios/` contains the ONNX adapter, landmark engine, MediaPipe starter and Xcode templates. `tests/` contains Python reference expectations and a Mac compiler/test runner.

Run on a Mac:

~~~sh
sh tests/verify_swift.sh
~~~

The iOS image source is the NCM board only. Use GestureNCMClient to receive validated JPEG frames; do not create an AVCaptureSession or request built-in-camera permission. Call GestureLandmarkEngine with accepted/stabilized image landmarks, matching world landmarks, handedness and a strictly increasing timestamp in seconds. Live Fist requires valid world landmarks. Keep one instance per stream; dispatch only non-nil actions.

The native frame preparation, detector recovery, face guard, smoothing, camera/NCM transport and product action routing require integration described in the guide. Swift/Xcode and physical-device validation have not been run on the Windows preparation host. The feature extractor's older standalone pose helpers are not the authoritative live decision path; use GestureGeometryResolver through GestureDecisionPipeline.
