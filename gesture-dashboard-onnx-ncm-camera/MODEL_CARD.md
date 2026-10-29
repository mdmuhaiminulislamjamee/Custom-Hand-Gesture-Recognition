# Model card: Stop inverted and feedback ONNX release

The active Windows model is `StopInvertedDirectionalFeedbackMLP`, a 76-feature MediaPipe hand-landmark classifier. It emits thirteen probabilities, including the reserved retired Dorsal output, and a separate `known_gesture_mass` score. There are twelve active commands. HaGRID `stop_inverted` maps to **Return to main position**. `no_gesture` is an internal rejection class with no action. The retired thumb-and-little-finger call sign is trained as `no_gesture`.

The Windows runtime uses hand landmarks only. The old Dorsal output remains disabled; a separately trained Stop inverted output permits the requested upright back-of-hand command. Body/face models and arm/waist presentation requirements are disabled. Geometry, source-pixel and temporal gates remain active. A resting hand that exactly matches a command cannot be distinguished by body posture. The weights were retrained for this release; clothing-specific live accuracy remains unmeasured.

Video Lab evaluates the current hand-only pipeline on reviewed clips. Model Training includes Stop inverted and the reviewed feedback, with participant-isolated public sources. Context negatives remain outside the 76-feature classifier. Existing immutable mobile handovers describe their packaged version and have not been upgraded to this model.

## Contract

- Input: `landmark_features`, `float32 [N, 76]`.
- Outputs: `probabilities`, `float32 [N, 13]`; `known_gesture_mass`, `float32 [N, 1]`; and internal argmax `label`.
- Order: `left`, `right`, `up`, `down`, `open_palm`, `like`, `dorsal`, `ok`, `fist`, `thumb_down`, `peace`, `rock`, `stop_inverted`.
- ONNX opset 16; CPU execution provider. The model hash and feature order are in `models/gesture_mlp_onnx_metadata.json`.
- Actions in that order: Move Left, Move Right, Move Up, Move Down, tracking toggle, Play/Pause, no action (reserved Dorsal), recording toggle, Mute, Volume Down, Turn 90 Degrees, Backup, Return to main position.

The analysis image and landmarks remain unmirrored. Under the NCM calibration, positive index screen x resolves to Left and negative x resolves to Right.

## Training and holdouts

The model retains the earlier participant-labeled HaGRID and dorsal-hand sources and adds 4,200 publisher-annotated Stop-inverted hands: 3,000 from the official training source, 600 validation and 600 test. Whole participants follow the existing evaluation ownership; training-partition participants reserved for validation/test stay held out. Mirrors, finger splay, and rotations augment training only. The trained classifier used known-source Palm provenance to correct horizontal negatives and a 10° guard band around horizontal endpoints, with lower rotations training rejection. The current runtime accepts Open Palm within XY −160° to −20° and Stop inverted within −145° to −35°; future training uses each command's band with a synthetic guard band. Synthetic transformations are not independent camera viewpoints.

The desktop runtime checks thumb direction in source-image pixels. A thumb pointing above the folded fingertips can resolve to Like even when ONNX favors Thumb Down; a thumb pointing below them can resolve to Thumb Down. Ambiguous sideways views are rejected. The model weights are unchanged by this runtime rule.

All 103 saved feedback corrections (30 Left, 48 Right, 12 Peace and 13 no_gesture) enter training with their reviewed actual labels. They have no participant/session metadata and are excluded from evaluation. Training fit on those corrections is not a generalization measurement. The current training and evaluation record is `artifacts/directional_feedback_20261027/evaluation.json`, with source provenance retained in `data/stop_inverted/sources.json`. The previous model/config/metadata/caches are preserved under `artifacts/directional_feedback_20261027/previous_release/`. Older release records remain historical evidence. HaGRID and the dorsal-hand source retain their attribution and redistribution terms.

The October 27 directional update adds 8,192 training variants of the reviewed Left/Right shapes. Mirroring swaps their labels. Rotations cover up to 38 degrees either side of horizontal, with mild affine variation and square, 4:3, 16:9 and portrait aspects. The runtime now measures direction sectors in source-image pixels, avoiding false Up/Down labels caused by rectangular normalized coordinates. All 78 reviewed directional landmark records classify and execute correctly, all 78 saved JPEGs execute the correct action when redetected and replayed as static holds, and 3,120 separate grid variants of those records pass. The 13 reviewed no-gesture landmark records remain rejected. These are feedback training checks, not independent new camera viewpoints or proof of all lighting and depth conditions. Original feedback and detailed per-frame reports remain local and are excluded from Git.

On the untouched public test split (9,042 rows), raw classifier macro-F1 is 0.990889, accepted precision is 0.994949 and unknown false acceptance is 0.011123. Left recall is 0.994092 and Right recall is 0.992614. The post-geometry report is recorded separately; its macro-F1 is 0.903042 and it lacks live depth and camera metadata. Participant overlap between train/validation/test is zero; model epoch selection uses validation only.

Training also varies normalized image aspect ratios for open-hand commands and genuine rejection examples. Training-only One examples include varied free-thumb positions while preserving the index direction and folded remaining fingers, retaining the earlier expanded-thumb Left/Right behavior. These transformations use training participants and do not copy the reported-camera test skeletons into fitting data.

## Rejection and pose policy

ONNX argmax alone never fires an action. The runtime checks hand validity, geometry, known mass, confidence, probability margin, temporal stability, hold duration, release frames, and cooldown. Selected scalar gates are known mass `0.70`, confidence `0.70`, margin `0.08`, geometry recovery mass `0.15`, three stable frames, minimum hold `0.18 s`, and action cooldown `0.80 s`.

Stop inverted requires four extended non-thumb fingers and confident dorsal-surface winding from detector handedness. Finger spacing is unrestricted. Both Palm and Stop inverted use the same wrist-to-knuckle source-image XY angle: 0° is screen-right, −90° is upward and −180° is screen-left. Open Palm accepts −160° to −20°, inclusive (up ±70°). Stop inverted keeps −145° to −35°, inclusive (up ±55°). Outside its command's band the open hand is no_gesture, even if the classifier reports a different command. Crossing either limit immediately clears the hold; returning needs a fresh hold. Source dimensions avoid aspect-ratio distortion. This image-plane angle is separate from the HUD's world-plane angle. The old Dorsal output cannot recover as this new command. Existing classifier measurements describe the trained model; the narrower open-hand runtime policies are checked separately by angle-boundary regressions.

A Fist requires closed-hand geometry regardless of model confidence. Both hands accept the complete upper screen arc: horizontal left, up-left, up, up-right and horizontal right. Any downward tilt below either horizontal boundary, including straight down, resolves to `no_gesture`. World-landmark wrist-to-knuckle depth is required for live Fist recognition. A 30-degree pitch toward the camera is accepted throughout the upper arc, including the floating-point boundary at normalized depth `-0.5`. Deeper forward tilts retain the established confident dorsal-surface recovery; an unverified reverse-facing view remains rejected. Compact, tucked-thumb fists confirmed by closed 3-D finger joints can recover an incorrect classifier label without a special screen-angle cone. Genuine Down and Thumbs Down shapes retain their separate checks. Crossing below horizontal clears a held Fist immediately and returning requires a fresh hold.

Closed-hand regressions compare the reported XY/YZ/ZX views `-86/+177/+51` (Fist), `-83/-148/+170` (No Gesture), and `-75/-146/+159` (dorsal Fist). These regressions use reconstructed axes and synthetic hand shapes; the screenshot pixels have not been replayed through MediaPipe.

The desktop resolver also recovers reported Peace side views from an incorrect Up prediction when two tracked straight index/middle fingers extend well beyond ring/little fingertips, including nearly touching fingers. World landmarks, when supplied, must corroborate the shape. The retired Dorsal output and lower extended open-hand poses remain rejection evidence. Regression fixtures approximate screenshot skeletons; they are not raw camera captures or screenshot pixel replays. Immutable mobile handovers retain their earlier behavior.

The NCM view at XY/YZ/ZX `+25/-75/+151` shows one long screen-right index, three folded fingers, and a thumb arched on the opposite side of the index. The desktop resolver treats this strict silhouette as Left even when ONNX reports approximately `0.01` known-gesture mass; horizontal mirroring maps it to Right. This recovery requires mass at least `0.005` and a 0.50-second stable hold. The regression reconstructs visible landmarks and HUD angles; it is not a replay of the camera image.

The reported close webcam view with an arched thumb and approximately `0.15`
known mass uses the same held recovery. A shorter projected index requires
three tightly curled, shorter fingers and a separately visible arched thumb.
Tests trace the screenshot overlay at multiple image aspect ratios and verify
that an additional extended finger cannot enter this recovery.

The three further reported side views at approximately `0.14/0.01/0.01`
known mass also use this recovery. When the thumb projects nearly parallel to
a shorter index, all three other fingertips must return behind their PIP joints
along the index direction, remain close to their own knuckles, and lie across
the index from the separated thumb. These checks handle an apparently straight
folded pinky without accepting a second extended finger. The `0.005` mass floor
and 0.50-second hold are unchanged. Tests use approximate overlay traces and
the installed ONNX classifier, including candidate selection and action output;
the original image detection and world depth are not replayed.

## Offline measurements

Current numerical parity, raw classifier metrics, post-geometry metrics, per-class reports and feedback training fit are recorded in `artifacts/directional_feedback_20261027/evaluation.json`. Raw classifier scores and post-geometry replay scores must be reported separately; neither measures live image detection accuracy. Feedback training fit is not an independent generalization result. The held-out source images are not decoded and re-detected by these landmark-level evaluations.

## Acceptance status

Offline qualification and artifact checks have passed. Live low-resolution webcam, USB-NCM, and physical iOS testing remain pending. The two screenshots supplied for the Fist distinction do not provide reusable landmark vectors or a camera test set, so the exact pictured cases require a live or captured-image regression before product sign-off. Measure all twelve classes, both hands, left/right/up Fists, downward and partly open negatives, the retired call sign, low light, varied distance, and false actions per minute. Compile and run the Swift package on a physical iOS device before shipping it.

This release replaces the active ONNX weights and extends the output contract by one class. The feedback CSV remains unchanged. An adapter saved against older weights is preserved by the runtime's existing compatibility backup mechanism; the new base model includes the reviewed corrections. Historical mobile handover archives retain their original models and require an explicit future upgrade.
