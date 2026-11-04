# Native false-hand source update - 31 October 2026

This update adds candidate filtering to the native Android and iOS diagnostic
detectors. Keep the owning app's current classifier and qualified decision
pipeline. No classifier weights have changed. The bundled face model SHA-256 is
`b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f`.

Android: copy `GestureFalseHandGuard.kt` and the updated `GestureHandLandmarker.kt`
with the existing handover sources. Add `blaze_face_short_range.tflite` to app
assets. `detect` now returns `GestureFilteredHandResult`; its `landmarks()`,
`worldLandmarks()` and `handedness()` contain accepted candidates only. The
supplied `GestureFrameAnalyzer` therefore filters before ONNX prediction.
Use the same filtered candidates for preview. `rejections` records raw indices
and reasons for diagnostics; those indices must not be treated as hands.

iOS: add `GestureFalseHandPolicy.swift` and `ios/GestureFaceGuard.swift` to the
app target, with the updated `ios/GestureHandLandmarker.swift`. Add the face
model beside `hand_landmarker.task` in app resources. `detect` now returns
`GestureFilteredHandResult.hands`; each accepted item supplies normalized
`landmarks`, metric `worldLandmarks`, `handedness`, and `handednessConfidence`.
Pass these accepted items to `GestureDiagnosticEngine` or the app's current
decision pipeline. Pass missing-hand evidence and clear previews when empty.

Each detector belongs to one serial stream. Face checks use the same prepared
pixels and capture timestamps as hands, refresh every 200 ms, and expire after
500 ms. Recreate the VIDEO hand detector after stream/orientation/timestamp
reset. A face-check failure throws: clear pending command evidence and preview;
do not retry that frame without the guard. Initialization requires the model.

The policy rejects compact candidates mostly inside the face/ear region,
insufficient source-pixel detail (18/40/12 pixel palm/span/short side),
incomplete candidates, and collapsed finger chains. Coherent 3-D joints permit
edge-on hands. A larger foreground hand or one fingertip crossing a face is
allowed. Small real hands directly over the face can be rejected. Plausible
background hallucinations can still pass, and stricter thresholds can reduce
range in dim scenes. This is a runtime mitigation, not detector retraining.

`false_hand_policy.json` contains 42 synthetic Python expectations covering
mirrors, scales, fingertip overlap, collapse, and 3-D rescue. The native sources
remain integration components, not complete native apps or action pipelines.
The Flutter APK is a separate Android tester; it does not verify these native
integrations. Compile the iOS integration with Xcode and test both platforms on
the intended phone/camera/NCM hardware before product release.

Host dependency check: the Android sources were compiled against cached
MediaPipe Tasks Vision 1.1.1. Pin that version initially and qualify any changes
with the owning app. On a Mac, run
`sh swift/tests/verify_false_hands.sh false_hand_policy.json` from this update's
root to compare the pure Swift policy with all 42 Python expectations.
Run `python verify_update.py` to check this source update's integrity.
