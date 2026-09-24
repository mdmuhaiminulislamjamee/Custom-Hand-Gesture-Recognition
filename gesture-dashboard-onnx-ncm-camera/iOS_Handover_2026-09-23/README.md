# iOS hand gesture runtime handover

Package date: **2026-09-23**

This folder contains the complete current `models` directory and the runtime
geometry needed to reproduce the ten gestures on iOS. The ONNX model itself is
unchanged. The camera-angle correction for the reported poses is implemented by
`swift/GestureGeometryResolver.swift` as a post-model landmark rule.

## What the iOS developer must add

Add these four source files to the application target:

1. `swift/GestureFeatureExtractor.swift`
2. `swift/GestureGeometryResolver.swift`
3. `swift/GestureTemporalGate.swift`
4. `swift/GestureActionLatch.swift`

Add these runtime resources to the target and confirm they appear under **Build
Phases > Copy Bundle Resources**:

1. `models/gesture_mlp_production.onnx`
2. `models/gesture_mlp_onnx_metadata.json`
3. `models/gesture_mobile_runtime_config.json`
4. `models/hand_landmarker.task`
5. `models/blaze_face_short_range.tflite`

The three `gesture_online_*_cache.npz` files and
`gesture_artifact_manifest.json` are included because this handover contains the
entire model folder. They are validation and desktop online-learning artifacts;
an iOS release does not need to copy the `.npz` files into the app bundle.

Use MediaPipe Tasks Vision for the 21 hand landmarks and ONNX Runtime for the
classifier. Keep both dependencies on versions supported by the application's
current iOS deployment target.

## Fixed ten-command contract

The ONNX input is `landmark_features`, type `float32`, shape `[1, 76]`.

Read both outputs by name:

- `probabilities`: `float32`, shape `[1, 10]`
- `known_gesture_mass`: `float32`, shape `[1, 1]`

The probability order must remain exactly:

| Index | Gesture | App action |
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

`no_gesture` is a rejection result. It is not an eleventh ONNX probability.

## Required processing order

For every accepted camera frame:

1. Run MediaPipe Hand Landmarker and obtain all 21 image landmarks.
2. Convert them to `[SIMD2<Float>]` without changing landmark order.
3. Build the 76 values with `GestureFeatureExtractor.landmarksToFeature`.
4. Run `gesture_mlp_production.onnx`.
5. Pass the ten probabilities and the same 21 landmarks to
   `GestureGeometryResolver.resolve`.
6. Pass that resolution and `known_gesture_mass` to one session-local
   `GestureTemporalGate`.
7. Pass the decision through `GestureActionLatch` and dispatch only the action
   string it returns.

```swift
import simd

// Create once for each camera/session.
let temporalGate = GestureTemporalGate()
let actionLatch = GestureActionLatch()

func processHand(
    mediaPipeLandmarks: [NormalizedLandmark],
    handedness: String?,
    handednessScore: Float?,
    timestampSeconds: TimeInterval
) throws {
    let points = mediaPipeLandmarks.map {
        SIMD2<Float>(Float($0.x), Float($0.y))
    }

    let features = try GestureFeatureExtractor.landmarksToFeature(points)

    // Implement this adapter with ONNX Runtime. It must return the two named
    // outputs and must not reorder the ten probabilities.
    let output = try runGestureONNX(inputName: "landmark_features", values: features)

    let geometry = try GestureGeometryResolver.resolve(
        probabilities: output.probabilities,
        landmarks: points,
        handedness: handedness,
        handednessConfidence: handednessScore
    )

    let decision = temporalGate.update(
        geometry,
        knownGestureMass: output.knownGestureMass,
        timestamp: timestampSeconds
    )

    if let action = actionLatch.update(decision, timestamp: timestampSeconds) {
        dispatchGestureAction(action)
    }
}
```

Do not use `GestureFeatureExtractor.runtimePrediction` as the final decision.
That older convenience method does not contain the complete geometry resolver.

## Camera and MediaPipe settings

- Analyze **unmirrored** camera pixels. The preview may be mirrored for the
  user, but the pixel buffer sent to MediaPipe and ONNX must stay unmirrored.
- MediaPipe coordinates must use `(0, 0)` at the image's top-left and positive
  y downward.
- Apply the device orientation once when creating the MediaPipe image. Do not
  rotate the returned landmarks again.
- Request one hand.
- Use VIDEO running mode for the normal stream.
- Start with detection `0.55`, presence `0.55`, and tracking `0.60`.
- The desktop recovery pass uses detection `0.20`, presence `0.30`, and requires
  two confirming frames. Mirror that behavior if distant-hand recovery is
  needed in the iOS camera pipeline.
- Limit analysis to 10 FPS and a maximum image dimension of 960 pixels.
- Clear `GestureTemporalGate` when the hand is missing, tracking is reacquired,
  or landmarks are rejected.

The horizontal command mapping is calibrated for this unmirrored input:
positive index-finger x means `left`; negative x means `right`.

## The new Down correction

The supplied model hash is:

```text
f598a0eb676f6bf7a42c1a46a4ff31615f6b8fc36d730edd1cfe8391d5bceecd
```

The new behavior is in the Swift and Python geometry files, not inside this
unchanged ONNX file. `GestureGeometryResolver` contains four tightly bounded
edge-on Down recoveries. The newest one is `reversePalmEdgeDown`; it recognizes
the reported view where three fingertips form a short row and either outer edge
finger is reconstructed much farther downward.

This recovery can replace an incorrect ONNX top class with `down`. When
`known_gesture_mass` is below `0.70`, `GestureTemporalGate` requires at least a
`0.40` second hold before it emits `Move Down`. Do not bypass the resolver or
the temporal gate.

## Runtime thresholds

The Swift gate uses the values in `gesture_mobile_runtime_config.json`:

| Setting | Value |
|---|---:|
| Probability EMA alpha | `0.45` |
| Confidence floor | `0.70` |
| Probability margin floor | `0.08` |
| Known gesture mass floor | `0.70` |
| Geometry recovery mass floor | `0.15` |
| Stable frames | `3` |
| Normal minimum hold | `0.18 s` |
| Low-mass directional hold | `0.40 s` |
| Action cooldown | `0.80 s` |
| Release frames before rearming | `3` |

`GestureActionLatch` keeps the action layer edge-triggered: one held pose does
not repeatedly fire the same command. It rearms after three release frames and
enforces the 0.80 second cooldown.

## Required startup checks

Run these once in a debug build before testing actions:

```swift
let featureCheck = GestureFeatureExtractor.runParitySelfTest()
precondition(featureCheck.passed, "76-feature parity failed")

precondition(
    GestureGeometryResolver.runDownRecoverySelfTest(),
    "Down camera-angle geometry regression failed"
)
```

Also check at startup that:

- the ONNX input and output names match this README;
- the probability tensor contains exactly ten values;
- metadata `output_class_order` equals
  `GestureFeatureExtractor.classNames`;
- the ONNX SHA-256 matches `SHA256SUMS.csv`.

## Physical-device acceptance

Before shipping, test the following on the actual iPhone/iPad camera used by the
app:

- each of the ten gestures with both hands;
- the supplied Down pose at several distances and wrist rolls;
- both outer-edge landmark assignments covered by the Down self-test;
- low light and bright backlight;
- brief hand loss and reacquisition;
- an upward Open Palm accepts, while left-, right-, and downward-facing Open
  Palms reject;
- held gestures produce one action and rearm only after release.

No landmark model can guarantee every person, distance, camera, or lighting
condition. The included regression tests verify the reported Down landmark
layouts and the desktop suite; physical-device acceptance verifies the iOS
camera and orientation integration.

## Reference files

`reference_python/geometry.py` is the authoritative desktop geometry source and
contains the same `reverse_palm_edge_down` rule. `reference_python/runtime.py`
contains the matching temporal gate. The remaining Python files document
detector recovery, smoothing, camera behavior, configuration, and action flow.
They are reference material and are not compiled into the iOS application.

Use `SHA256SUMS.csv` to verify that no file changed during transfer. On the
iOS developer's Mac, run this command from the handover folder:

```bash
python3 verify_package.py
```
