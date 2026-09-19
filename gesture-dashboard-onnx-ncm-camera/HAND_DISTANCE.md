# Hand distance and range testing

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

The old NCM path reduced the central camera region to 320 × 320 pixels and
rejected hands below 1.8% of that region's area. The updated path preserves native
detail up to a maximum dimension of 960 pixels. On the current 640 × 480 camera,
the central 92% region is analysed at 442 × 442 pixels. This does not create detail
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
