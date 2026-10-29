# Current Boss handover

Give the Boss app developer this complete archive:

[Boss_Static_Gesture_Handover_20261029_fix1.zip](Boss_Static_Gesture_Handover_20261029_fix1.zip)

Its root `README.md` explains how to integrate the full pipeline, why the
model alone is insufficient, and which file does what. `START_HERE.md` covers
the release summary. The bundle contains the latest Flutter/iOS static runtime,
models, existing dynamic-detector integration, two-hand preview, validation and
device acceptance. The current Android tester source and verified test APK are
included as optional diagnostics. The iOS app must be built/signed on a Mac and
qualified with the supported iPhone/NCM board.

Verify the adjacent archive `.sha256` before transfer, and run the extracted
`verify_handover.py` with Python 3.10+ to check the complete payload.

This revision includes the thumb-direction fix and corrects the shared
Open Palm/Stop Inverted handedness swap. It contains GestureLab 2.0.1 source
and the rebuilt Android APK.

The older `candidate`, `overlay` and `final` archives remain here for history.
The `fix1` archive supersedes them.
