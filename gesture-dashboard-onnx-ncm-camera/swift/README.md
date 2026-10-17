# Swift integration sources

Use the final guide in [IOS_INTEGRATION.md](../handover_docs/IOS_INTEGRATION.md) and the dated bundle produced by `scripts/create_final_handover.py`. Dorsal is retired: the root Swift action map and decision gates reject it as `no_gesture`. Its legacy ONNX tensor slot remains reserved. Dated handover archives are unchanged. Full parity with the latest Windows geometry and a Mac/device acceptance run are required before a new native release.

The five root Swift files implement the 76-feature extractor and post-landmark decisions. `ios/` contains the ONNX adapter, landmark engine, MediaPipe starter and Xcode templates. `tests/` contains Python reference expectations and a Mac compiler/test runner.

Run on a Mac:

~~~sh
sh tests/verify_swift.sh
~~~

The iOS image source is the NCM board only. Use GestureNCMClient to receive validated JPEG frames; do not create an AVCaptureSession or request built-in-camera permission. Call GestureLandmarkEngine with accepted/stabilized image landmarks, matching world landmarks, handedness and a strictly increasing timestamp in seconds. Live Fist requires valid world landmarks. Keep one instance per stream; dispatch only non-nil actions.

The native frame preparation, detector recovery, face guard, smoothing, camera/NCM transport and product action routing require integration described in the guide. Swift/Xcode and physical-device validation have not been run on the Windows preparation host. The feature extractor's older standalone pose helpers are not the authoritative live decision path; use GestureGeometryResolver through GestureDecisionPipeline.
