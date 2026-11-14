# Current Boss handover

Give the Boss app developer this complete archive:

[Boss_Static_Gesture_Handover_20261114_two_hand_stability.zip](Boss_Static_Gesture_Handover_20261114_two_hand_stability.zip)

Its root `README.md` explains how to integrate the full pipeline, why the
model alone is insufficient, and which file does what. `START_HERE.md` covers
the release summary. The bundle contains the latest Flutter/iOS static runtime,
models, existing dynamic-detector integration, two-hand preview, validation and
device acceptance. The current Android tester source and verified test APK are
included as optional diagnostics. The iOS app must be built/signed on a Mac and
qualified with the supported iPhone/NCM board.

Verify the adjacent archive `.sha256` before transfer, and run the extracted
`verify_handover.py` with Python 3.10+ to check the complete payload.

This revision carries the handedness stability fix into the shared Flutter
runtime and GestureLab 2.0.12 APK. Established, spatially separated hands retain
their identities during temporary Left/Right confidence drops. New identities
still require .75 confidence, current landmarks remain required, and existing
overlap/reset/expiry rules remain in force.

The previous mobile handover's model binaries, matching ONNX/configuration
snapshot, features, geometry, scalar gates, detector configuration and command
timing are preserved. `static_gesture_runtime/validation/two_hand_stability_release.json`
records the exact comparison with the previous archive. No training or extra
inference pass is included. Physical device acceptance remains pending.

The older archives remain unchanged for history. This delivery supersedes the
11 November `feedback_refresh` bundle. [START_HERE.md](final_docs/START_HERE.md) and
[RELEASE_NOTES.md](final_docs/RELEASE_NOTES.md) describe integration and changes.
