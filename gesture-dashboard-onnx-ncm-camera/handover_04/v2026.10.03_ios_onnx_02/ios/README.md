# iOS integration contract

The five Swift sources provide feature extraction, geometry resolution,
temporal gating, action latching, and a stream decision pipeline. Pair them with the ONNX model, metadata,
runtime JSON, and hand-landmarker asset from the same dated release package.

## Command order

The `probabilities` tensor contains twelve values in this exact order:

| Index | Command | Action |
|---:|---|---|
| 0 | `left` | Move Left |
| 1 | `right` | Move Right |
| 2 | `up` | Move Up |
| 3 | `down` | Move Down |
| 4 | `open_palm` | Enable / Disable Object Tracking |
| 5 | `like` | Play / Pause |
| 6 | `dorsal` | Return to Default Position |
| 7 | `ok` | Start / Stop Recording |
| 8 | `fist` | Mute |
| 9 | `thumb_down` | Volume Down |
| 10 | `peace` | Turn 90 Degrees |
| 11 | `rock` | Backup |

`no_gesture` is a rejection sentinel, not a thirteenth probability. Validate the
tensor width and the class order from `gesture_mlp_onnx_metadata.json` before
enabling actions.

## Decision pipeline

Feature extraction and the ONNX result are only part of the decision. Create
one `GestureDecisionPipeline` per camera stream. After reading the ONNX outputs
by name, pass the values and the matching image landmarks to that pipeline:

```swift
let points: [SIMD2<Float>] = handLandmarkerResult.landmarks[0].map {
    SIMD2<Float>(Float($0.x), Float($0.y))
}
let features = try GestureFeatureExtractor.landmarksToFeature(points)
// Run ONNX with a float32 [1, 76] tensor built from features.
let decision = pipeline.process(
    probabilities: probabilities,          // named output [1, 12]
    knownGestureMass: knownGestureMass,    // named output [1, 1]
    landmarks: points,
    handedness: detectedHandedness,
    handednessConfidence: detectedHandednessScore,
    timestamp: frameTimestamp
)
if let action = decision.action {
    dispatch(action)
}
```

The Open Palm and Fist direction gates allow the wrist-to-palm axis to point
up or sideways in the image. Downward poses are rejected. Fist and Thumbs Down
checks are neutral to horizontal mirroring, so they cover either hand.

The supplied Swift resolver, temporal gate, and action latch implement the
core decision path. A full iOS app must also integrate camera or JLIP capture,
MediaPipe detection, ONNX Runtime, detector recovery, face guard, landmark
smoothing, frame scheduling, and the action destination. Compare its decisions
with the Python reference on captured frames before claiming parity.

## Camera coordinates

- Supply all 21 MediaPipe image landmarks in canonical index order.
- Use normalized image coordinates with `(0, 0)` at the top left.
- Keep analysis pixels unmirrored. On a front camera, explicitly disable video
  mirroring where supported.
- Apply the physical device orientation when creating the MediaPipe image; do
  not rotate the landmarks a second time.
- The same contract applies to frames from the device webcam and decoded NCM
  JPEG frames.

## Release checks

Before shipping on a physical iPhone or iPad:

1. Check that the ONNX input is `landmark_features`, `float32`, `[1, 76]`.
2. Query `probabilities` and `known_gesture_mass` by name.
3. Verify `probabilities` is `[1, 12]` and its order equals `classNames`.
4. Run `GestureFeatureExtractor.runParitySelfTest()`.
5. Exercise both hands for Fist and Thumbs Down at multiple distances, rolls,
   and camera angles.
6. Confirm Open Palm fires while the palm axis points up, left, right, or obliquely sideways, and rejects downward poses.
7. Confirm every rejected pose produces no action and clears the hold state.

Offline metrics do not replace these physical-device checks.
