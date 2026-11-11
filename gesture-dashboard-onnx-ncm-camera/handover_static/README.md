# Current Boss handover

Give the Boss app developer this complete archive:

[Boss_Static_Gesture_Handover_20261111_right_stability.zip](Boss_Static_Gesture_Handover_20261111_right_stability.zip)

Its root `README.md` explains how to integrate the full pipeline, why the
model alone is insufficient, and which file does what. `START_HERE.md` covers
the release summary. The bundle contains the latest Flutter/iOS static runtime,
models, existing dynamic-detector integration, two-hand preview, validation and
device acceptance. The current Android tester source and verified test APK are
included as optional diagnostics. The iOS app must be built/signed on a Mac and
qualified with the supported iPhone/NCM board.

Verify the adjacent archive `.sha256` before transfer, and run the extracted
`verify_handover.py` with Python 3.10+ to check the complete payload.

This revision trains only the Right output using the 331 new Right poses and
90 new hard-negative landmark records alongside the previous training data.
It corrects the Right side-view/open-palm conflict with metric-world corroboration.
It uses .75 confidence/.10 margin gates and preserves fast confirmation and
latest-frame behavior. It contains GestureLab 2.0.10 source
and a rebuilt diagnostic release APK with the later USB-NCM and preview fixes.
Face/ear filtering, thumb-direction and Open Palm/Stop Inverted fixes are
retained. The exact ONNX/configuration are included under `source_model/`.

The older archives remain unchanged for history. This delivery supersedes the
earlier 11 November `direction_stability` bundle. [START_HERE.md](final_docs/START_HERE.md) and
[RELEASE_NOTES.md](final_docs/RELEASE_NOTES.md) describe integration and changes.
