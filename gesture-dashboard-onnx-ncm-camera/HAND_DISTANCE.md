# Hand distance and range testing

## Automatic selection of the gesture hand

Both `run_live_camera.bat` and the live dashboard now use the same automatic
two-hand selection pipeline. Restart a running camera process. Restart the
dashboard using `start_ncm_onnx_dashboard.bat` to rebuild and load the changes.
No tapping, calibration, or preferred left/right setting is needed for selection.

The preview outlines both observations and highlights the selected hand. The
dashboard's **Automatic hand selection** card and the BAT camera's bottom banner
explain which hand is used or why actions are paused. Existing metre estimates
are approximate. When both hands have coherent world landmarks, selection compares
their image-to-world projected scale. It falls back to apparent palm size when
that comparison is unavailable. Neither cue measures camera depth.

Every frame includes a full-frame detection pass. A bounded second pass can
recover a difficult hand from a rotated/cropped view. Each hand is associated
with a persistent track using palm position, shape, size, and a soft handedness
cue. Each track has its own confidence, geometry, vocabulary, and temporal
checks. Classifier history and landmark smoothing reset on a change of hand.
The dashboard NCM path now analyzes the full image, matching the BAT viewer.
Previews use a fitted full image and both skeletons follow the same coordinates.
Recalibrate any saved range calibration that was made with the old central crop.

- Both clearly visible physical hands are compared before gesture qualification,
  including idle hands and real hands blocked by a context rule. Their gesture
  labels do not affect the comparison. Face hallucinations and incomplete/tiny
  clouds do not count as another reliable physical hand.
- With two hands present, coherent image/world landmarks can identify
  a hand that projects at least **1.10 times** larger, even when palm rotation
  makes its four image palm segments look similar to the other hand's. This
  comparison applies to every gesture label and either hand. If coherent world
  landmarks are unavailable, the closer palm must appear at least **1.15 times**
  larger with consistent ratios across four palm segments.
  Only that nearer hand is passed to gesture recognition; a farther gesture
  cannot beat a nearer idle or blocked hand. Choice must persist for at least
  **3 observations and 0.25 seconds**. The gesture has its own hold.
- Similar projected scales or a small, uncertain difference keep an already
  selected hand active while its track is clear. Without an existing owner,
  neither hand is selected until one moves clearly forward. Uncertain track
  assignment still pauses actions.
- When the other hand appears clearly in front, the selected hand stays visible
  while the challenger is checked. New actions pause during this check. The
  challenger takes ownership only after **3 observations and 0.25 seconds** of
  a sustained lead; a one-frame size spike cannot switch hands. Only the
  selected hand can show a recognized gesture; inactive previews remain
  `no_gesture`.
- An invalid but physically tracked foreground hand remains selected but does
  not produce an action. If it disappears for **0.80 seconds**, selection resets
  even when another hand remains visible. That hand must complete a fresh
  gesture hold before it can act. No output uses stale landmarks, and a
  high-confidence opposite handedness cannot inherit the owner's track.
- Track position continues updating during body rejections. A unique,
  confidently identified returning hand can recover its owner ID after a large
  movement. That recovery resets smoothing and gesture confirmation, so stale
  commands cannot execute at the new position. Ambiguous or opposite hands
  cannot use this recovery.
- Selection uses the same temporal gap policy as gesture confirmation. Valid
  observations arriving 350–499 ms apart no longer reset the selection hold on
  every frame.
- An intermediate projected scale difference (**1.05–1.10 times**) or a
  similarly small, inconsistent palm-size difference keeps a clearly tracked
  owner. A larger but inconsistent difference or uncertain track assignment
  pauses actions without transferring ownership.
- If two hands were recently tracked and a frame finds only one, the detector
  attempts its bounded second pass on that frame. Recovery still requires
  fresh landmarks; missing hands cannot produce a gesture action.
- Live selection requires at least **18 palm pixels**, **40 hand-span pixels**,
  and **12 pixels along the short side** of the hand. Dorsal additionally requires
  **24 palm pixels**. These checks use analysis-image pixels before recovery crop
  magnification. Tiny observations can remain visible in gray without triggering
  an action. The older 8/24/10 detector floors below still apply to collecting raw
  observations; they are no longer sufficient for live action eligibility.

Defaults are in `backend/hand_selection.py` (`SelectionPolicy`). They are starting
values for real camera trials, not a measured accuracy guarantee. Palm rotation,
different hand anatomy, inaccurate world landmarks, occlusion, and lighting can
still bias the comparison. A single RGB camera cannot resolve every near-equal
or fully occluded two-hand arrangement.
The body-position restriction applies only to Dorsal on a relaxed hanging arm
below the waist or a folded resting arm with a dropped hand. Face overlap does
not pause live gestures. These are
RGB estimates rather than measured camera depth or
cloth/skin contact.

Suggested live checks, in both the BAT camera and dashboard:

1. Hold each gesture with either hand alone; confirm the selected hand and action.
2. Move the other hand in front after selection. The old gesture should stop,
   then the new foreground hand should be selected. A nearer idle hand should
   prevent a farther gesture from taking control.
3. Start with both hands at similar apparent sizes, including identical gestures
   or one idle hand. Expect no gesture from either hand.
4. Bring one hand clearly forward and hold. Expect selection after confirmation.
5. Swap screen positions without overlapping, then overlap/cross the hands. Expect
   stable identities while separated and no action during uncertain overlap.
6. Move the selected hand back to the other hand's apparent size. Expect it to
   remain selected while both hands are clearly tracked. Cross or overlap the
   hands so identity becomes uncertain; expect actions to pause.
7. Hide the selected hand while the peer stays visible. Expect a brief pause,
   then a fresh selection and gesture hold after 0.80 seconds. If the selected
   hand remains physically tracked but its gesture is invalid, expect it to
   remain selected without an action.
8. Check Dorsal with a straight or slightly bent hanging arm below the waist,
   and with a folded arm resting beside a desk while the hand hangs downward;
   expect no gesture. Raise the arm and hold a deliberate Dorsal; expect
   recognition. Check both arms and verify that other classes remain eligible.

Record false actions, missed intended gestures, and time to selection. Automated
tests cover selection and state transitions; the actual camera trial is still
needed to establish suitable size and apparent-closeness thresholds.

## Calibrated distance and range measurements

Open **Hand distance & detection range** in the Live or Feedback tab, then connect
the board camera. The compact heading keeps the latest distance visible when the
controls are collapsed.

1. Measure from the camera lens to your palm with a tape. Enter that distance in
   metres under **Known calibration distance**. Use a clearly visible, steady
   palm facing the camera, near the centre of the guide.
2. Hold still for at least one second and click **Calibrate steady palm**. Ten
   consistent observations of the same hand and image dimensions are required.
3. Move back while keeping the palm orientation similar. The panel shows an
   approximate distance, the spread of four palm-segment estimates, and hand size
   in analysis pixels. Recalibrate for another person, the other hand, a different
   camera, or a changed zoom/crop. Rotating the palm can bias all four estimates.
4. At each test position, measure the distance independently, enter it under
   **Measured test distance**, select the gesture, and run the 10-second test.
   Keep the tab visible and hold the pose throughout. Repeat all ten gestures,
   neutral poses, transitions, both hands, and representative lighting conditions.
5. **Export CSV** saves the observations. Copy `gesture-distance-tests.csv` into
   the project directory to review distance-grouped results in `v18_20.ipynb`.

The trial counts hand tracking and correct *confirmed* gestures separately. It
samples at 10 Hz, counts missing/stale frames as unavailable, and includes the
initial confirmation period. Measurements are held in the current browser page
until exported or reloaded. A trial's measured distance remains available when
landmarks disappear; the current estimated distance does not. The last seen
estimate is explicitly labelled with its age.

This is a calibrated apparent-size estimate based on the pinhole camera relation
`distance = reference_distance × reference_palm_size / current_palm_size`.
It is not a depth sensor, and the segment spread is not a statistical confidence
interval. MediaPipe's world landmarks are relative to the hand, so their Z value
must not be presented as camera-to-hand distance. See the
[MediaPipe hand landmark guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
and [OpenCV camera model](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

The dashboard's live **XY**, **YZ**, and **XZ** values are signed projection
angles of the wrist-to-palm axis derived from MediaPipe world landmarks. They
help compare pose orientation across captures; they are not absolute Euler
angles, camera distance, or a calibrated measurement of the user's arm.

## Detection changes and evidence

### Dorsal on a relaxed hanging or folded arm

The BAT viewer and dashboard match each hand to confident MediaPipe body
landmarks: right shoulder/elbow/wrist **12/14/16** with hip **24**, or left
shoulder/elbow/wrist **11/13/15** with hip **23**. After classification, only
**Dorsal** with a below-waist hand and a relaxed hanging arm is `no_gesture`.
The elbow's interior angle must be at least **135 degrees**, and both upper arm
and forearm must point down along the torso. Shoulder, elbow, wrist and hip
visibility/presence must each be at least **0.65**. The palm centre must be at or
below the hip's waist plane; wrist matching allows a small tolerance for a
sloping torso and mildly bent hanging arm.

For a seated desk view with the waist cropped, a second narrow check uses a
confident shoulder, elbow and wrist. It suppresses Dorsal when the upper arm
drops, the elbow folds outward, the forearm returns across the shoulder line,
and the fingers hang downward. A bent arm by itself does not trigger this rule.

A raised arm or a bent arm outside the folded resting pattern remains eligible
for Dorsal. There is no horizontal trouser margin and no
body-position restriction on Fist, Up, Down, Left, Right or the other classes.
The same rule checks the final smoothed prediction; crop recovery cannot bypass
a confirmed relaxed Dorsal rejection. Ordinary gesture holds still apply.

Per-hand `body_context` includes `below_waist`, `relaxed`, `hanging_arm`,
`folded_resting_arm`,
`arm_confident`, `arm_landmark_indices`, `waist_landmark_index`,
`elbow_angle_deg`, `upper_arm_down_alignment`, `forearm_down_alignment`,
`wrist_height` and `palm_height`. Missing or weak body evidence does not impose
a body-position veto.

Dorsal, Open Palm and Fist no longer have fixed screen-angle rejection ranges.
They still require the intended finger shape and palm/back-of-hand evidence.
The defining directions of Up, Down, Left, Right, Like and Thumb Down remain
part of recognizing those gestures.

Live gestures are not rejected because of face position or contact. The Face
Landmarker model has been removed from the runtime and installer. No face
inference or contact hold runs in the live hand detector. The old face-contact
helper is retained only for offline regression/reference work.

Dorsal requires four straight, together, approximately parallel fingers, with
the thumb beside the index finger and the back of the hand visible. The thumb
gap to the index-finger segment must be at most 0.50 of wrist-to-middle-MCP
length. This shape rule applies at every screen rotation and location,
including in front of the face. A loose spread-thumb hand does not qualify.
This is an explicit gesture definition, not an attempt to infer intent.

Pose Lite also checks the folded resting arm when the hip is out of frame.
Other gestures have no body-position veto. Missing body landmarks cannot
establish either relaxed posture, so keep the shoulder, elbow and wrist visible
when validating this rule.
A confirmed resting posture survives a brief missing body estimate for up to
650 ms, only while the same hand stays in place at a similar size. Movement,
an explicit raised/bent-arm estimate or ambiguous hand matching clears it.
Reused posture evidence never extends its own timeout.

Regression checks cover the traced chin-rest screenshot, both hands, rotated
and scaled deliberate Dorsal, relaxed arms, and above-waist/bent-arm positives.
Screenshot traces and synthetic posture tests are not measured camera accuracy.

Horizontal Left/Right accepts an expanded thumb. A strict long straight index,
separate extended thumb and three deeply curled fingers additionally supports
the reported low-mass L-shaped pose in `backend/geometry.py`. This narrow
recovery still requires at least 0.01 vocabulary mass, a 500 ms hold and no
online negative evidence. Other Left/Right shapes retain the usual mass gates;
Peace/Rock/open fingers and diagonal directions cannot use this recovery.

This uses [MediaPipe Pose Landmarker Lite](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker),
with the official version-1 model pinned by SHA-256 in `backend/body_guard.py`.
The model is included in `models/pose_landmarker_lite.task`; setup can restore it
using `scripts/download_pose_landmarker.py`. A missing or corrupt model prevents
startup instead of silently disabling the check.

Pose inference uses a maximum dimension of 512 pixels, runs at most once every
150 ms when a usable hand is present, and retains a missed body for at most
400 ms. The current hand is checked on every frame, including recovered views.
Pose history is isolated by camera session. Per-frame diagnostics include
`quality.detector.body_guard` and each rejected hand's `body_context` evidence.

This is a posture heuristic, not detection of cloth contact or measured depth.
The head and knees may be cropped. A reliable hip is required for the
below-waist case; the folded-arm case can work without it. If the body cannot
be located, the body-position rule leaves the hand eligible; the other gesture
and selection checks still apply. A clearly enlarged foreground hand is excluded
from body matching. To intentionally perform Dorsal, raise the arm away from
the hanging or folded resting posture.

Restart the BAT viewer or dashboard backend to load this change. During live
trials, compare both relaxed hands beside the trousers, one raised gesture with
the other hand relaxed, a clear listed gesture beside the face, unknown poses
beside the face, and an intentional dorsal pose above the waist.
Repeat with the head cropped, different clothing and lighting. Live accuracy
and the extra pose inference cost have not yet been measured on these cases.

The old NCM path reduced the central camera region to 320 × 320 pixels and
rejected hands below 1.8% of that region's area. The updated path preserves native
detail up to a maximum dimension of 960 pixels. On the current 640 × 480 camera,
the previous central 92% region was analysed at 442 × 442 pixels. The automatic
two-hand path now uses the full 640 × 480 image. This does not create detail
that the camera did not capture.

The detector can recover through a rotated or overlapping cropped view and local
contrast correction. Coordinates are mapped back to the original unmirrored
camera view before classification. A second pass is attempted only while the
first pass remains within its 55 ms allowance; there are at most two passes per
frame. The normal detection thresholds and command confidence/hold rules remain
in place. Recovery searches use lower candidate thresholds, then the existing
classifier and geometry rules qualify candidates. A rejected finger arrangement
triggers reacquisition rather than indefinite tracking of the wrong arrangement.

The fixed area cutoff is replaced with a minimum of 8 palm pixels, 24 pixels
along the hand's long side, and 10 along its short side. These are information
floors, **not accuracy guarantees**. Crop magnification cannot bypass them.

The captured Left development case improved from 0/50 to 44/50 frames with usable
landmarks; 32/50 frames confirmed Left after the required holds. Six frames still
lost the hand. No other command was confirmed. Pipeline p95 was 74.4 ms, with one
startup frame over 100 ms in that replay. See
`artifacts/v18_20/left_range_regression.json`. This is a replay of the case used to
develop the fix, not an independent live accuracy or distance measurement.

No maximum reliable distance in metres has been measured yet. Establish it using
the range test on your actual camera: require consistently high tracking and
correct-confirmation rates across repeated trials, and check false actions on
neutral hands too. Results depend on source resolution, focus, motion blur,
lighting, hand angle, and the individual. Older public-landmark classification
scores do not measure these image-detection failure modes.
