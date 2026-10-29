# Final delivery changes

This archive contains the latest static runtime and the Android tester source
used for GestureLab v2.0.1. It replaces the earlier candidate/overlay/final handovers.

- Like/Thumb Down recovery now checks the thumb's camera-pixel direction and
  its position above/below folded fingertips, even when the model strongly
  favors the opposite thumb label.
- Surface geometry consumes physical Left/Right directly. Removing a second
  handedness swap fixes the reversed Open Palm/Stop Inverted behavior in both
  the tester and static handover. Mirrored detector input is normalized once.

- Public API exports `StaticDetectorAdapter` so the app can reuse its existing
  dynamic detector without a second camera or hand inference pipeline.
- The adapter supports explicit detector-input mirror parity for handedness;
  the NCM default remains upright and unmirrored.
- The full detector, GPU preference/CPU fallback, FP32 classifier, 76-feature
  extractor, pose geometry and two-hand temporal controller are included.
- The two-hand preview pairs a completed JPEG with the hand results from that
  exact frame and draws the original dynamic detector boxes, rotated ROIs,
  21 landmarks per hand and skeletons. Empty detections clear stale overlays.
- The Android tester now shares the same runtime for phone and NCM cameras.
  Its NCM client connects directly, validates JLIP/CRC/JPEG frames, replies to
  heartbeats, reconnects and keeps only the newest pending frame. The strict
  exact-address Android route precheck has been removed.
- The Android APK contains ARM64 libraries only and the LiteRT models. Retired
  ONNX/MediaPipe-task assets and audio/storage permissions are excluded.
- The iOS example enables the OpenCV modules used by the detector with one
  native compilation job. Native builds remain enabled in this delivery.
- Complete payload hashes, model/APK consistency checks and source provenance
  are supplied with the final archive.

The existing learned gesture vocabulary, weights and label/action mapping are
preserved. Runtime timing follows the new two-hand controller documented in the
runtime README; it intentionally differs from the old selected-owner desktop
controller. The INT8 experiment remains unqualified and blocked from loading.

The Android APK is a release-mode test artifact, signed with the debug key.
The iOS example is source for the owning team to build and sign. Review
`static_gesture_runtime/VALIDATION.md` for the scope of prior model/geometry
checks and `release_manifest.json` for final host checks.
