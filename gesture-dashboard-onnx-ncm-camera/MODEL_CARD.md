# Model card: twelve-command ONNX release

The active model is `TwelveGestureBalancedMLP`, a 76-feature MediaPipe hand-landmark classifier for the Windows dashboard and iOS handover. It emits twelve command probabilities and a separate `known_gesture_mass` score. `no_gesture` is a rejection label with no action. The retired thumb-and-little-finger call sign is trained as `no_gesture`.

The Windows runtime uses hand landmarks only for eleven active commands. Dorsal is retired and is always `no_gesture`; its output slot is reserved to preserve the existing ONNX tensor contract. Body/face models and arm/waist presentation requirements are disabled. Full-frame search and bounded local contrast/crop recovery support difficult clothing backgrounds and uneven lighting. Upright acquisition crops are revisited from a cold start; successful crops retain fresh VIDEO tracking while the full frame is scanned each frame. Geometry, source-pixel and temporal gates remain active. A resting hand that exactly matches a command cannot be distinguished by body posture in this policy. This changes runtime behavior, not model weights; clothing-specific live accuracy remains unmeasured.

Video Lab evaluates the current hand-only pipeline on reviewed clips. Model Training creates isolated twelve-class ONNX candidates with validation-only epoch selection and explicit frozen-test evaluation. Context negatives remain outside the 76-feature classifier. Existing immutable mobile handovers describe their packaged version; mobile ports need the current hand-only behavior before claiming parity.

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

A Fist requires closed-hand geometry regardless of model confidence. Both hands accept the complete upper screen arc: horizontal left, up-left, up, up-right and horizontal right. Any downward tilt below either horizontal boundary, including straight down, resolves to `no_gesture`. World-landmark wrist-to-knuckle depth is required for live Fist recognition. A 30-degree pitch toward the camera is accepted throughout the upper arc, including the floating-point boundary at normalized depth `-0.5`. Deeper forward tilts retain the established confident dorsal-surface recovery; an unverified reverse-facing view remains rejected. Compact, tucked-thumb fists confirmed by closed 3-D finger joints can recover an incorrect classifier label without a special screen-angle cone. Genuine Down and Thumbs Down shapes retain their separate checks. Crossing below horizontal clears a held Fist immediately and returning requires a fresh hold.

Closed-hand regressions compare the reported XY/YZ/ZX views `-86/+177/+51` (Fist), `-83/-148/+170` (No Gesture), and `-75/-146/+159` (dorsal Fist). These regressions use reconstructed axes and synthetic hand shapes; the screenshot pixels have not been replayed through MediaPipe.

The desktop resolver also recovers the reported Peace side views from an incorrect Up prediction when two independently tracked, straight index/middle fingers extend well beyond the ring/little fingertips. This includes nearly touching fingers. World landmarks, when supplied, must corroborate the two-finger geometry. Every former Dorsal pose is a hard negative at every screen angle and known-gesture mass. Neither geometry recovery, body position, foreground clearance nor a long hold can authorize it. Dorsal-shaped hands are rejected before any alternate command can execute. Regression fixtures approximate the screenshot skeletons; they are not raw camera landmark captures or screenshot pixel replays. These desktop changes are not included in the immutable iOS handover snapshots.

The NCM view at XY/YZ/ZX `+25/-75/+151` shows one long screen-right index, three folded fingers, and a thumb arched on the opposite side of the index. The desktop resolver treats this strict silhouette as Left even when ONNX reports approximately `0.01` known-gesture mass; horizontal mirroring maps it to Right. This recovery requires mass at least `0.005` and a 0.50-second stable hold. The regression reconstructs visible landmarks and HUD angles; it is not a replay of the camera image.

The reported close webcam view with an arched thumb and approximately `0.15`
known mass uses the same held recovery. A shorter projected index requires
three tightly curled, shorter fingers and a separately visible arched thumb.
Tests trace the screenshot overlay at multiple image aspect ratios and verify
that an additional extended finger cannot enter this recovery.

## Offline measurements

On the participant-disjoint held-out test, the ONNX graph agreed with the source model on all validation argmax predictions. Maximum absolute probability error was below `2.3e-6`. Test macro F1 including rejection was `0.99108`; accepted precision was `0.99500`; unknown false acceptance was `0.01765`. Among 291 retired-sign negatives, two passed the raw model thresholds (`0.00687` false acceptance). The ten original commands had a correct-accept rate of `0.99154`. Peace recall was `0.98662` and Rock recall was `0.99677` before post-model geometry. Post-geometry Peace recall was `0.96321`. These are landmark-level results. See the evaluation JSON for the confusion matrix and per-class scores.

## Acceptance status

Offline qualification and artifact checks have passed. Live low-resolution webcam, USB-NCM, and physical iOS testing remain pending. The two screenshots supplied for the Fist distinction do not provide reusable landmark vectors or a camera test set, so the exact pictured cases require a live or captured-image regression before product sign-off. Measure all twelve classes, both hands, left/right/up Fists, downward and partly open negatives, the retired call sign, low light, varied distance, and false actions per minute. Compile and run the Swift package on a physical iOS device before shipping it.

Current policy does not change the shipped ONNX weights. Collection, reviewed legacy Dorsal ingestion, online feedback and Model Lab training map Dorsal to rejection; a future retrained candidate must be qualified independently. Historical handover archives retain their original versions.
