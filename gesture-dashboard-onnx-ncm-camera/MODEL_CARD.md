# Model card: twelve-command ONNX release

The active model is `TwelveGestureBalancedMLP`, a 76-feature MediaPipe hand-landmark classifier for the Windows dashboard and iOS handover. It emits twelve command probabilities and a separate `known_gesture_mass` score. `no_gesture` is a rejection label with no action. The retired thumb-and-little-finger call sign is trained as `no_gesture`.

## Contract

- Input: `landmark_features`, `float32 [N, 76]`.
- Outputs: `probabilities`, `float32 [N, 12]`; `known_gesture_mass`, `float32 [N, 1]`; and internal argmax `label`.
- Order: `left`, `right`, `up`, `down`, `open_palm`, `like`, `dorsal`, `ok`, `fist`, `thumb_down`, `peace`, `rock`.
- ONNX opset 16; CPU execution provider. The model hash and feature order are in `models/gesture_mlp_onnx_metadata.json`.
- Actions in that order: Move Left, Move Right, Move Up, Move Down, tracking toggle, Play/Pause, default position, recording toggle, Mute, Volume Down, Turn 90 Degrees, Backup.

The analysis image and landmarks remain unmirrored. Under the NCM calibration, positive index screen x resolves to Left and negative x resolves to Right.

## Training and holdouts

The model is trained on participant-labeled HaGRID landmarks and a dorsal-hand source. The thumb-and-little-finger sign is a labeled negative; Peace and Rock are commands. Participant groups are held out before augmentation. Exact feature duplicates are removed across splits. The new-command validation and test rows originate from the publisher's training partition but are participant-disjoint from the fitting rows. Synthetic rotation is a landmark transform, not a new camera view.

The training and evaluation record is `artifacts/twelve_gesture/evaluation.json`; geometry-level results are in `artifacts/twelve_gesture/runtime_comparison.json`. The retired ten-command material is a historical baseline. HaGRID and the dorsal-hand source have separate attribution and redistribution terms; review them before commercial redistribution.

## Rejection and pose policy

ONNX argmax alone never fires an action. The runtime checks hand validity, geometry, known mass, confidence, probability margin, temporal stability, hold duration, release frames, and cooldown. Selected scalar gates are known mass `0.70`, confidence `0.70`, margin `0.08`, geometry recovery mass `0.15`, three stable frames, minimum hold `0.18 s`, and action cooldown `0.80 s`.

A Fist requires closed-hand geometry regardless of model confidence. Either hand can make a Fist across the upward XY semicircle, with up to 20 degrees of tolerance below either horizontal boundary. The desktop resolver also accepts the supplied reverse-facing XY -130, YZ -143, ZX -132 fist within a 12-degree 3-D cone, and recovers the supplied side-view XY -5, YZ +117, ZX +80 fist within an 18-degree cone when an invalid Thumbs Down label is predicted. Both views allow horizontal mirroring. The existing compact side-view recovery around XY +8, YZ +68, ZX +72 remains available. Recovery requires curled fingers and a tucked thumb; a true Thumbs Down keeps its label. Other downward Fists and reverse-facing directions outside the reported cone are rejected. If the live camera supplies no usable world landmarks, Fist resolves to No Gesture because its facing direction cannot be verified. Partly bent open hands are negative examples to collect and challenge. Open Palm also accepts upward and sideways axes and rejects downward axes. Thumbs Down requires a downward thumb relationship.

The later camera corrections override general Fist eligibility: the three closed-hand views at XY/YZ/ZX `-64/-161/+126`, `-68/+161/+50`, and `-20/-150/+102` are No Gesture within six degrees of their 3-D palm axes. This veto applies to both model-predicted Fist and geometry recovery, and clears a held Fist immediately. The closed side view at `-3/-99/+107` can recover an invalid Thumbs Down label as Fist within ten degrees, subject to the same tucked-thumb and closed-finger checks. All four rules include horizontal mirrors. These regressions use reconstructed axes and synthetic hand shapes; the screenshot pixels have not been replayed through MediaPipe.

The desktop resolver also recovers the reported Peace side views from an incorrect Up prediction when two independently tracked, straight index/middle fingers extend well beyond the ring/little fingertips. This includes nearly touching fingers. World landmarks, when supplied, must corroborate the two-finger geometry. Dorsal uses the displayed signed XY palm-axis range from 0 through 150 degrees clockwise from screen-right; world XY is authoritative when available. A verified dorsal surface with four straight, parallel, close fingers can recover even at known mass 0.05, after a 0.45-second hold. Unknown/palmar surfaces do not receive this low-mass exception. Regression fixtures approximate the six screenshot skeletons; they are not raw camera landmark captures or screenshot pixel replays. These desktop changes are not included in the immutable iOS handover snapshots.

## Offline measurements

On the participant-disjoint held-out test, the ONNX graph agreed with the source model on all validation argmax predictions. Maximum absolute probability error was below `2.3e-6`. Test macro F1 including rejection was `0.99108`; accepted precision was `0.99500`; unknown false acceptance was `0.01765`. Among 291 retired-sign negatives, two passed the raw model thresholds (`0.00687` false acceptance). The ten original commands had a correct-accept rate of `0.99154`. Peace recall was `0.98662` and Rock recall was `0.99677` before post-model geometry. Post-geometry Peace recall was `0.96321`. These are landmark-level results. See the evaluation JSON for the confusion matrix and per-class scores.

## Acceptance status

Offline qualification and artifact checks have passed. Live low-resolution webcam, USB-NCM, and physical iOS testing remain pending. The two screenshots supplied for the Fist distinction do not provide reusable landmark vectors or a camera test set, so the exact pictured cases require a live or captured-image regression before product sign-off. Measure all twelve classes, both hands, left/right/up Fists, downward and partly open negatives, the retired call sign, low light, varied distance, and false actions per minute. Compile and run the Swift package on a physical iOS device before shipping it.
