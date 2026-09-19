# iOS integration contract

`GestureFeatureExtractor.swift` is the Swift counterpart of the desktop
76-feature pipeline. Pair it only with the ONNX model, metadata, runtime JSON,
and hand-landmarker asset from the same dated release package.

## Command order

The `probabilities` tensor contains ten values in this exact order:

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

`no_gesture` is a rejection sentinel, not an eleventh probability. Validate the
tensor width and the class order from `gesture_mlp_onnx_metadata.json` before
enabling actions.

## Required pose gate

Feature extraction and the ONNX result are only part of the decision. For the
three newly constrained poses, call the supplied gate before temporal
confirmation:

```swift
let points: [SIMD2<Float>] = handLandmarkerResult.landmarks[0].map {
    SIMD2<Float>(Float($0.x), Float($0.y))
}
let features = try GestureFeatureExtractor.landmarksToFeature(points)

// Run ONNX with `features`, then use its top class and probability.
let poseAllowed = try GestureFeatureExtractor.runtimePoseAllowed(
    predictedGesture: topClass,
    modelConfidence: topProbability,
    landmarksXY: points
)
guard poseAllowed else {
    // Reset the candidate hold and publish no_gesture / Wait / No Action.
    return
}
```

The Open Palm command is allowed only when the wrist-to-palm axis points up in
the image. Left-, right-, and downward-facing open palms must be rejected. The
Fist and Thumbs Down checks use vertical and extension geometry, so horizontal
mirroring produces the same result for left and right hands.

This helper covers the new cross-platform constraints. A production-equivalent
iOS app must also port all detector recovery, handedness/surface validation,
directional resolution, confidence/margin/known-mass gates, probability EMA,
hold/release state, cooldown, and reviewed-negative logic described by the
runtime JSON and `backend/geometry.py`.

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
3. Verify `probabilities` is `[1, 10]` and its order equals `classNames`.
4. Run `GestureFeatureExtractor.runParitySelfTest()`.
5. Exercise both hands for Fist and Thumbs Down at multiple distances, rolls,
   and camera angles.
6. Confirm Open Palm fires only while the palm axis points upward.
7. Confirm every rejected pose produces no action and clears the hold state.

Offline metrics do not replace these physical-device checks.
