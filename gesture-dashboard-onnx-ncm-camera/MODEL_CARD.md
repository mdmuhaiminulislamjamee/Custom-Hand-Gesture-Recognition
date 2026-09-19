# Model Card — Ten-Gesture ONNX Release

Release: **TenGestureBalancedMLP, 2026-09-16**

## Purpose and scope

This local model recognizes ten static hand commands from MediaPipe hand
landmarks for the Windows dashboard, standalone Webcam/NCM viewer, and iOS
integration package. It is not a biometric system or a safety-critical control.
Physical controls and product-level fail-safe behavior remain necessary.

## Contract

- Input: `landmark_features`, float32 `[N, 76]`.
- Outputs: `probabilities` float32 `[N, 10]`, `known_gesture_mass` float32
  `[N, 1]`, and an internal argmax label.
- Order: `left`, `right`, `up`, `down`, `open_palm`, `like`, `dorsal`, `ok`,
  `fist`, `thumb_down`.
- Actions: Move Left, Move Right, Move Up, Move Down, tracking toggle,
  Play/Pause, default position, recording toggle, Mute, and Volume Down.
- `no_gesture` is an internal negative/rejection class and is not exposed as a
  command probability.
- ONNX opset: 16; CPU provider; one intra-op and one inter-op thread.

The image is never horizontally mirrored. Under the target NCM calibration,
positive index-direction x resolves to Left and negative x resolves to Right.

## Data and training

The internal training taxonomy has ten commands plus `no_gesture`. Every
internal class is balanced to 4,096 feature rows. Public landmark sources use
HaGRID v2 annotations, including `fist` and `dislike` (mapped to
`thumb_down`), plus the existing participant-labelled dorsal-hand source.
Training includes both natural handednesses, bounded training-only rotation,
scale/noise augmentation, expanded Rock/Call/Peace/neutral negatives, and
explicit left/right/down Open-Palm negatives.

Participant IDs are disjoint across the public training, validation, and test
splits. Exact feature duplicates are removed with evaluation precedence. The
legacy replay cache is training-only and has incomplete participant provenance;
this limitation is recorded in model metadata.

The final confirmation set contains 4,000 rows from 2,600 additional HaGRID
participants with zero overlap against training, validation, or test. It was
not used for model fitting or scalar threshold selection. It was, however, a
fail-closed development/release gate: an earlier candidate's failure on this
cohort informed subsequent training and geometry revisions. Treat the final
numbers as regression evidence, not as an untouched statistical estimate.

HaGRID carries its publisher's custom CC BY-SA terms. The dorsal source is
CC BY-NC-SA; review dataset licensing before commercial redistribution.

## Pose and rejection policy

An ONNX argmax does not directly dispatch an action. The runtime applies pose
validation, known-mass/confidence/margin checks, temporal EMA, stable-frame and
hold requirements, release frames, and cooldown.

- Open Palm requires an upward wrist-to-MCP palm axis. Sideways/downward Open
  Palm is rejected and cannot fall through to Dorsal when the live surface gate
  identifies a palmar hand.
- Fist and Thumbs Down work for either hand.
- A compact curl-back resolver recovers raised, camera-facing Fists when
  perspective hides the finger bend; it cannot override Like or Thumbs Down.
- Thumbs Down accepts foreshortened camera views while still requiring a
  downward thumb relationship and model evidence.
- Geometry recovery normally requires at least `0.15` known mass. Only the
  tightly bounded Down silhouettes retained from real NCM regressions can use
  the special lower-mass path.

Selected gates: known mass `0.70`, confidence `0.70`, probability margin
`0.08`, three stable frames, minimum hold `0.18 s`, and cooldown `0.80 s`.

## Measured offline performance

On the participant-disjoint public test split, used as a release-regression
gate after candidate iteration:

| Metric | Value |
|---|---:|
| Raw known accuracy | 0.99618 |
| Raw known macro F1 | 0.99626 |
| Runtime macro F1 including rejection | 0.99035 |
| Accepted precision | 0.99769 |
| Unknown false acceptance | 0.01000 |
| Minimum command F1 | 0.98485 |
| Fist precision / recall / F1 | 1.00000 / 0.98250 / 0.99117 |
| Thumbs Down precision / recall / F1 | 0.99490 / 0.97500 / 0.98485 |
| Like F1 | 0.98737 |

On the additional new-participant confirmation cohort:

| Metric | Value |
|---|---:|
| Present-class macro F1 | 0.97587 |
| Accepted precision | 0.99630 |
| Unknown/non-up-palm false acceptance | 0.00250 |
| Fist recall / F1 | 0.98250 / 0.98992 |
| Thumbs Down recall / F1 | 0.96000 / 0.97710 |
| Like recall / F1 | 0.95250 / 0.97567 |
| Upward Open Palm recall | 1.00000 |

ONNX/source prediction agreement is `1.0`; maximum absolute probability error
is below `1e-6`. Classifier-only p95 latency was below `0.05 ms` on the
qualification machine. Full image decoding, MediaPipe, transport, and UI time
are not included in that latency number.

## Limitations and required acceptance

Offline landmark evaluation cannot guarantee performance for every hand,
disability, demographic, lens, background, occlusion, blur, lighting level, or
camera angle. Direction variants include rotations of source landmarks rather
than independent horizontal captures. Dorsal/palmar separation depends on
MediaPipe handedness at runtime because 2-D landmarks alone are ambiguous.

Before product acceptance, test both hands for Fist and Thumbs Down and all
angles expected in use on both Webcam and NCM. Explicitly test Open Palm upward
versus left/right/down, empty scenes, partial hands, low light, Like versus
Thumbs Down, Down versus Dorsal, and action debounce. Record a live confusion
matrix, false actions per minute, latency, landmark jitter, FPS/budget results,
and NCM transport errors. Repeat equivalent tests on physical iOS hardware.
