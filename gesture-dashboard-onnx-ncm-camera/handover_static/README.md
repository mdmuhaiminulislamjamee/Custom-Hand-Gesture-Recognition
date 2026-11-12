# Current Boss handover

Give the Boss app developer this complete archive:

[Boss_Static_Gesture_Handover_20261111_feedback_refresh.zip](Boss_Static_Gesture_Handover_20261111_feedback_refresh.zip)

Its root `README.md` explains how to integrate the full pipeline, why the
model alone is insufficient, and which file does what. `START_HERE.md` covers
the release summary. The bundle contains the latest Flutter/iOS static runtime,
models, existing dynamic-detector integration, two-hand preview, validation and
device acceptance. The current Android tester source and verified test APK are
included as optional diagnostics. The iOS app must be built/signed on a Mac and
qualified with the supported iPhone/NCM board.

Verify the adjacent archive `.sha256` before transfer, and run the extracted
`verify_handover.py` with Python 3.10+ to check the complete payload.

This revision fits 990 reviewed records absent from the previous training
snapshot: 480 Down, 240 Thumb Down, 150 Stop Inverted, 90 Like and 30 rejection
examples. Larger fits regressed existing examples and were rejected. The
accepted update preserves fixed test class metrics; its training-fit improvement
is modest and is not a live accuracy claim. The model graph, scalar gates,
detector, geometry and frame scheduling are retained. GestureLab 2.0.11 source
and a rebuilt diagnostic APK are included. The exact model/configuration are
under `source_model/`; current evidence is in
`static_gesture_runtime/validation/feedback_refresh_release.json`.
Ghost policy tuning is deferred; physical device acceptance is pending.

The older archives remain unchanged for history. This delivery supersedes the
earlier 11 November `right_stability` bundle. [START_HERE.md](final_docs/START_HERE.md) and
[RELEASE_NOTES.md](final_docs/RELEASE_NOTES.md) describe integration and changes.
