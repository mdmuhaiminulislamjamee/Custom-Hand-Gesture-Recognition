# Hand distance and range testing

## Automatic selection of the gesture hand

Both `run_live_camera.bat` and the live dashboard now use the same automatic
two-hand selection pipeline. Restart a running camera process. Restart the
dashboard using `start_ncm_onnx_dashboard.bat` to rebuild and load the changes.
No tapping, calibration, or preferred left/right setting is needed for selection.

The preview outlines both observations and highlights the selected hand. The
dashboard's **Automatic hand selection** card and the BAT camera's bottom banner
explain which hand is used or why actions are paused. Existing metre estimates
are approximate; selection compares apparent palm size and does not use them as
measured camera depth.

Every frame includes a full-frame detection pass. A bounded second pass can
recover a difficult hand from a rotated/cropped view. Each hand is associated
with a persistent track using palm position, shape, size, and a soft handedness
cue. Each track has its own confidence, geometry, vocabulary, and temporal
checks. Classifier history and landmark smoothing reset on a change of hand.
The dashboard NCM path now analyzes the full image, matching the BAT viewer.
Previews use a fitted full image and both skeletons follow the same coordinates.
Recalibrate any saved range calibration that was made with the old central crop.

- If only one hand qualifies, it can control the app even if an idle hand is larger.
- If both qualify initially, one palm must appear at least **1.35 times** larger,
  with reasonably consistent ratios across palm segments. The choice must persist
  for at least **3 observations and 0.25 seconds**, in addition to gesture holds.
- Once selected, a hand remains selected while its gesture is valid. Switching to
  another valid hand needs a stronger **1.60 times** palm-size ratio and the same
  choice confirmation. Actions pause while that switch is being confirmed.
- At startup, similarly sized qualifying hands cause a pause. Bring the intended
  hand forward or relax the other. Strong overlap or uncertain track assignment
  also pauses actions. The system cannot recover fingers hidden by the other hand.
- A missing active hand has a **0.60-second** grace period before another hand may
  take over. No action uses stale landmarks during that period. A visible hand
  that releases its gesture gives up ownership after **0.35 seconds**.
- Live selection requires at least **18 palm pixels**, **40 hand-span pixels**,
  and **12 pixels along the short side** of the hand. Dorsal additionally requires
  **24 palm pixels**. These checks use analysis-image pixels before recovery crop
  magnification. Tiny observations can remain visible in gray without triggering
  an action. The older 8/24/10 detector floors below still apply to collecting raw
  observations; they are no longer sufficient for live action eligibility.

Defaults are in `backend/hand_selection.py` (`SelectionPolicy`). They are starting
values for real camera trials, not a measured accuracy guarantee. Palm rotation,
different hand anatomy, occlusion, and lighting can still bias apparent size.
Clothing proximity is not used as a universal veto; a valid foreground hand may
be in front of the user's shirt. Face proximity is recorded for diagnostics and
the normal gesture qualification rules still apply there.

Suggested live checks, in both the BAT camera and dashboard:

1. Hold each gesture with either hand alone; confirm the selected hand and action.
2. Add a larger idle hand. The gesturing hand should keep control.
3. Start with both hands making gestures at similar apparent sizes. Expect a pause.
4. Bring one hand clearly forward and hold. Expect selection after confirmation.
5. Swap screen positions without overlapping, then overlap/cross the hands. Expect
   stable identities while separated and no action during uncertain overlap.
6. Briefly hide the selected hand. Expect a pause without immediate takeover.
7. Stand farther away with relaxed hands beside clothing. Check that tiny dorsal
   candidates do not trigger. Then show an intentional gesture and move closer.

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

### Relaxed hands beside clothing

The BAT viewer and dashboard live inference match each hand to confident
shoulder, wrist and hip landmarks. After gesture classification, a **dorsal**
hand below the waist is treated as `no_gesture`. A dorsal hand hanging down
beside the hip or upper thigh is also marked **resting / no gesture**. Other
listed gestures at that position remain eligible. A blocked dorsal hand cannot
win automatic selection, and a new temporal hold is required after it moves
into an eligible position.

A face box is now recorded as scene context, with no automatic proximity veto.
The previous veto rejected a clear open palm at the user's ear before its pose
could be classified. A hand beside the face now goes through the same twelve
gesture geometry, known-mass, confidence and hold rules as other hands. These
rules reject many unknown poses, but no closed-set classifier can guarantee
that every unlisted hand shape will be rejected. Collecting representative
face/ear hard negatives and evaluating them remains important.

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
The head, elbows and knees may be cropped for the below-waist rule, but reliable
shoulder, wrist and hip landmarks are required. A hanging-arm label additionally
needs the elbow and downward fingers. If the body cannot be located, this rule
cannot identify a below-waist hand; the other gesture and selection checks still
apply. A clearly enlarged foreground hand is excluded by the body matching rule.
To intentionally perform dorsal, raise the hand above the waist.

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
