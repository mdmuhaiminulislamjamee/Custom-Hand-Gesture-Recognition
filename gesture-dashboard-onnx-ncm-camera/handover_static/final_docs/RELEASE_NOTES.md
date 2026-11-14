# Two-hand stability - 14 November 2026

- Retained both established, spatially separated hand identities during low
  Left/Right confidence observations. The existing spatial match now runs
  before rejecting weak labels. Fresh image/world landmarks are still required.
- Kept the .75 confidence requirement for new physical identities. Existing
  overlap abstention, 350 ms track expiry, reset behavior and single-hand
  uncertainty handling are unchanged. A matched track supplies its previously
  confirmed side/confidence; no geometry or classifier threshold is lowered.
- Preserved every mobile model asset, manifest, feature/geometry rule, command
  controller, detector setting and dependency from the feedback-refresh bundle.
  The matching `source_model/` snapshot is retained even where the desktop
  checkout has a newer ONNX model. No training or conversion was performed.
- Added regressions for weak/flipped handedness, fresh coordinates, mirrored
  inputs, invalid confidence, expiry and reset. Included GestureLab 2.0.12
  source and a newly compiled ARM64 diagnostic APK using the same models.
- Kept previous archives immutable. `HOST_CHECKS.json` and the runtime's
  `validation/two_hand_stability_release.json` bind the new source, APK and
  unchanged assets. Physical phone/NCM improvement still needs device testing.

## Previous model release: feedback refresh - 11 November 2026

- Used all 990 new valid reviewed records absent from the previous training
  snapshot: 480 Down, 240 Thumb Down, 150 Stop Inverted, 90 Like and 30 rejection
  examples. One existing duplicate ID was deduplicated for fitting; 11 older
  rows without finite 21x2 landmarks were excluded. Forty-two new JPEGs had been
  deleted, so those records train from their saved landmarks only.
- Fitted output corrections on the released 76/128/64/14 MLP. The installed
  ONNX is `d8a37784b603fc9917ef5635f40c2b20b1032c7898a1fe98b3babc9b33c00044`.
  Normalization, hidden layers, unaffected output columns, 17 operations and the
  77,797-byte ONNX size are retained. Runtime rules and scheduling are retained.
- Larger joint, coordinate and hidden-layer fits were retained as experiments
  and rejected when preservation gates failed. The accepted conservative step
  improves accepted correctness on the 990 new training rows from 628 to 630.
  This is a training-fit result, not independent camera accuracy.
- Every scalar and geometry-resolved per-class metric on the 9,042-row fixed
  regression test split is preserved. Scalar macro F1 remains 0.9910724152 and
  accepted precision 0.9958095238. The 9,031-row validation split and earlier
  reviewed corrections were also checked. These reused benchmarks support
  regression acceptance, not a new blind generalization estimate.
- A deterministic 48-image, 12-frame hold replay retains 38 correct final
  classifications and improves observed expected actions from 37 to 38.
  The sample includes existing Left/Right and new positive gesture captures.
  Saved training images cannot establish new-person or physical-camera accuracy.
- Retained .75 confidence, .10 margin, .70 known mass, bounded directional
  continuation, the metric-world Right correction, existing ghost/face/body
  rejection, independent hand state and all twelve command mappings.
- Retained persistent inference, 100 ms minimum frame admission, latest pending
  frame replacement, immediate strong confirmation, ordinary two-of-three
  confirmation and no fixed action cooldown. No extra inference pass is used.
- Regenerated FP32/INT8 assets, model fixtures and integrity records from the
  selected ONNX. FP32 remains the qualified default; INT8 stays blocked pending
  precision-specific geometry/controller and target-device acceptance.
- Included GestureLab 2.0.11 source, current USB-NCM/preview behavior and a newly
  built ARM64 diagnostic release APK. Existing archives remain immutable.

Fresh evidence is in `static_gesture_runtime/validation/feedback_refresh_release.json`,
`validation/conversion_report.json`, `validation/onnx_latency.json`,
`validation/latency_comparison.json` and root `HOST_CHECKS.json`.
The six existing fist-orientation/hand-ownership backend failures were reproduced
before training and remain explicit. Ghost detection work is deferred.

Same-host classifier timing excludes camera capture, transport, detector,
rendering and confirmation. Physical iPhone/NCM speed and accuracy, iOS builds
and signing remain device acceptance work. The Android APK uses release Dart,
disabled shrinking and a debug signing key for test distribution.
