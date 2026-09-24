# Ten-Gesture Runtime Artifacts

Current qualified release: **TenGestureBalancedMLP (2026-09-16)**.

This directory is the atomic runtime bundle:

- `gesture_mlp_production.onnx` — 76-feature, ten-output classifier plus
  `known_gesture_mass`;
- `gesture_mlp_onnx_metadata.json` — hash, tensor/class contract, data audit,
  qualification, per-class evidence, and independent confirmation evidence;
- `gesture_mobile_runtime_config.json` — action mapping, pose policy, rejection
  thresholds, 10 FPS controls, and Safe Learn settings;
- `gesture_online_*_cache.npz` — ten-command replay/validation/test anchors;
- `hand_landmarker.task` and `blaze_face_short_range.tflite` — detector assets;
- `ten_gesture_*` CSV files — readable release metrics and confusion matrix;
- `gesture_artifact_manifest.json` — sizes and SHA-256 hashes for every trusted
  runtime artifact.

Canonical probability order:

```text
left, right, up, down, open_palm, like, dorsal, ok, fist, thumb_down
```

`fist` maps to **Mute** and `thumb_down` maps to **Volume Down**.
`no_gesture` is the internal rejection/feedback label, not an eleventh output.
Open Palm is eligible only when the palm axis points upward; left-, right-, and
downward Open Palm poses are intentionally rejected. Fist and Thumbs Down are
handedness-neutral.
Raised, camera-facing Fists also use a compact curl-back resolver for angles
where perspective hides the finger bend. That resolver cannot override an
existing Like or Thumbs Down prediction.

## Qualification summary

Training is balanced at 4,096 rows for each of 11 internal classes (45,056
rows total). HaGRID participant groups are disjoint across training,
validation, and test. The release also passed a 4,000-row confirmation
cohort from 2,600 additional HaGRID participants with zero participant overlap.

Key public-test measurements:

- raw known-class accuracy: `0.99618`;
- raw known-class macro F1: `0.99626`;
- runtime macro F1 including rejection: `0.99035`;
- accepted precision: `0.99769`;
- unknown false-acceptance rate: `0.01000`;
- Fist F1 / recall: `0.99117` / `0.98250`;
- Thumbs Down F1 / recall: `0.98485` / `0.97500`;
- Like F1: `0.98737`;
- classifier-only p95 latency: below `0.05 ms` on the qualification machine.

On the independent new-participant confirmation cohort, present-class macro F1
was `0.97587`, accepted precision was `0.99630`, unknown/non-up-palm false
acceptance was `0.00250`, Fist recall was `0.98250`, and Thumbs Down recall was
`0.96000`. Thumbs Down F1 (`0.97710`) slightly exceeded Like F1 (`0.97567`).

The confirmation cohort was excluded from fitting and scalar-threshold
selection, but an earlier failure on it informed later training and geometry
revisions. It is therefore release-regression evidence, not an untouched
statistical estimate. These are landmark-level offline measurements, not a
promise for every person, camera, angle, or lighting condition. Webcam, NCM
hardware, and physical iOS acceptance remain required.

## Runtime policy

The selected scalar gates are confidence `0.70`, probability margin `0.08`,
known mass `0.70`, and geometry-recovery mass `0.15`. The special, tightly
bounded Down camera-angle fallback remains available below that recovery floor;
other geometry paths cannot override essentially zero model evidence.

Pixels and landmarks stay unmirrored. Positive index-finger screen x maps to
Left and negative x maps to Right under the NCM calibration.

## Releasing another model

Train and qualify without touching production, then deploy only if every gate
passes:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_ten_gesture
.\.venv\Scripts\python.exe -m scripts.qualify_ten_gesture
.\.venv\Scripts\python.exe -m scripts.qualify_ten_gesture --deploy
.\.venv\Scripts\python.exe -m scripts.onnx_preflight
```

The artifact manifest is an integrity check, not an accuracy test. For a future
release, create and reserve a new participant-disjoint final cohort before
qualification; do not reuse this release's test caches as fresh evidence.
Historical eight-command CSVs are preserved under
`artifacts/v18_20/baseline/release_evidence/`, outside this runtime bundle.
