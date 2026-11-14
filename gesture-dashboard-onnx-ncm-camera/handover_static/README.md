# Current Boss handover

Give the Boss app developer this complete archive:

[Boss_Static_Gesture_Handover_20261114_ownership_preview.zip](Boss_Static_Gesture_Handover_20261114_ownership_preview.zip)

Its root `README.md` explains how to integrate the full pipeline, why the
model alone is insufficient, and which file does what. `START_HERE.md` covers
the release summary. The bundle contains the latest Flutter/iOS static runtime,
models, existing dynamic-detector integration, two-hand preview, validation and
device acceptance. The current Android tester source and verified test APK are
included as optional diagnostics. The iOS app must be built/signed on a Mac and
qualified with the supported iPhone/NCM board.

Verify the adjacent archive `.sha256` before transfer, and run the extracted
`verify_handover.py` with Python 3.10+ to check the complete payload.

This revision includes the shared active-hand selector and GestureLab 2.0.13
APK for phone/NCM cameras. The first qualified hand stays active when another
appears beside it. A clearly closer challenger, or the only valid peer gesture,
must confirm across three observations and at least 250 ms before handoff.
The selected hand supplies the displayed gesture and commands. The earlier
weak-handedness tracking fix is retained.

The NCM preview keeps the previous decoded picture and its own overlay visible
until the next JPEG and matching landmarks are ready. This removes repeated
decode-time skeleton/box blinking without pairing new points to an old picture.

The previous mobile handover's model binaries, matching ONNX/configuration
snapshot, features, geometry, scalar gates, detector configuration, single-hand
confirmation and FPS scheduling are preserved. Switching adds a confirmation
hold and rebuilds gesture evidence after transfer. The new
`static_gesture_runtime/validation/ownership_preview_release.json` records the
comparison with the previous archive. No training or extra inference pass is
included. Ear/ghost and incidental-motion tuning remain deferred. Physical
device acceptance remains pending.

The older archives remain unchanged for history. This delivery supersedes the
14 November `two_hand_stability` bundle. [START_HERE.md](final_docs/START_HERE.md) and
[RELEASE_NOTES.md](final_docs/RELEASE_NOTES.md) describe integration and changes.
