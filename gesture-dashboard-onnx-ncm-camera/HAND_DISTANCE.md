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

Every frame includes a full-frame detection pass. Up to two bounded recovery passes can
recover difficult hands from rotated/cropped views. Each hand is associated
with a persistent track using palm position, shape, size, and a soft handedness
cue. Three confident observations establish the hand's left/right identity;
the identity stays with its continuously tracked physical hand. Clearly separated
two-hand tracks survive duplicate or swapped detector labels and detection-order
changes. Ambiguous overlap still pauses actions; a lone opposite hand cannot
inherit a missing owner's track. Each track has its own confidence, geometry, vocabulary, and temporal
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
  challenger takes ownership only after raw and filtered distance evidence agree
  for **3 observations and 0.25 seconds**. Apparent palm and coherent projected
  world scales use a time-adjusted EMA (0.50 per 100 ms). Borderline noisy bursts
  cannot complete a switch using a hold accumulated before filtered evidence
  agrees. Only the
  selected hand can show a recognized gesture; inactive previews remain
  `no_gesture`.
- An invalid but physically tracked foreground hand remains selected but does
  not produce an action. If it disappears for **0.80 seconds**, selection resets
  even when another hand remains visible. That hand must complete a fresh
  gesture hold before it can act. No output uses stale landmarks, and a
  high-confidence opposite handedness cannot inherit the owner's track.
- Track position continues updating during geometry rejections. A unique,
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
  pauses actions without transferring ownership. When the owner's current
  landmarks and identity remain clear, it stays visible and selected during
  this pause. Missing or ambiguously overlapping hands do not use that path.
- If a frame finds only one hand, the detector searches for its peer on both
  sides of that hand, including when the person is off-centre. Recovery still requires
  fresh landmarks; missing hands cannot produce a gesture action.
- Live selection requires at least **18 palm pixels**, **40 hand-span pixels**,
  and **12 pixels along the short side** of the hand. These checks use analysis-image pixels before recovery crop
  magnification. Tiny observations can remain visible in gray without triggering
  an action. The older 8/24/10 detector floors below still apply to collecting raw
  observations; they are no longer sufficient for live action eligibility.

Defaults are in `backend/hand_selection.py` (`SelectionPolicy`). They are starting
values for real camera trials, not a measured accuracy guarantee. Palm rotation,
different hand anatomy, inaccurate world landmarks, occlusion, and lighting can
still bias the comparison. A single RGB camera cannot resolve every near-equal
or fully occluded two-hand arrangement.
Live inference uses hand landmarks only. Body, arm and waist position, face overlap, and clothing are not gesture vetoes. Hand shape, source detail and stability still determine recognition.

Both hand overlays use motion-adaptive smoothing local to their track. The
selected hand's accepted landmark filter supplies its recognized overlay.
Raw landmarks and current finger poses still drive classification; the display
filter does not authorize commands or invent hands during a missed detection.
Dorsal is a hard negative regardless of foreground clearance, hand count, distance
or wrist angle. It cannot activate a command.

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
8. Show all commands in front of plain, dark and patterned clothing, at chest
   and lap level and with the body cropped. Complete valid hands remain eligible.
   Check that low-detail, incomplete and unknown hand shapes remain rejected.
   A resting hand that matches a command can qualify under the hand-only policy.

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

### Hand-only detection and clothing recovery

Body and face inference are disabled. All former Dorsal poses are `no_gesture`,
including deliberately presented hands and views with the body cropped.
Geometry and source-pixel checks still reject incomplete or ambiguous hands.
Legacy body helpers and models remain available for offline reference only.

Cold-start recovery repeatedly checks an upright central crop while continuing
the other search views. A productive view retains fresh VIDEO tracking in the
bounded recovery passes while every frame still starts with a full-frame scan.
Two crop streams retain independent VIDEO tracking histories, so finding one
hand does not reset the other's tracker. Recent
hand locations guide padded crops for up to 650 ms, with
missing hands prioritized over already visible hands. These local retries
alternate with a progressing global rotation/crop search. Original and enhanced
contrast are both searched; enhancement can reduce detector recall on some palms.
Every frame still
starts with full-frame detection; cached coordinates never count as detected
landmarks and never execute a command. Recovery state belongs to each camera
session and expires when observations are too old or image size changes.

Automated regressions cover removal of body vetoes for all twelve commands,
contrast enhancement, recovery after a tracking miss, missing second hands,
search progression, history expiry, session separation and the frame budget.
These checks do not measure live detection accuracy against clothing. Validate
on recorded deployment-camera footage across backgrounds, lighting and motion.
Fully hidden fingers, severe blur and absent image detail remain limits.

The old NCM path reduced the central camera region to 320 × 320 pixels and
rejected hands below 1.8% of that region's area. The updated path preserves native
detail up to a maximum dimension of 960 pixels. On the current 640 × 480 camera,
the previous central 92% region was analysed at 442 × 442 pixels. The automatic
two-hand path now uses the full 640 × 480 image. This does not create detail
that the camera did not capture.

The detector can recover through a rotated or overlapping cropped view and local
contrast correction. Coordinates are mapped back to the original unmirrored
camera view before classification. Recovery starts within 55 ms; a further peer
pass may start within 85 ms. There are at most three passes per frame, and at
most two resident recovery VIDEO streams. Extra recovery can reduce frame rate.
The webcam runner requests 1280 × 720 capture, with driver fallback, to retain
more distant finger detail. Source-pixel eligibility checks still apply.
The normal detection thresholds and command confidence/hold rules remain
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
