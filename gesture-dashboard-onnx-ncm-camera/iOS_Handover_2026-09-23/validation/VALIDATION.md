# Validation record

The handover was assembled from the current project workspace on 2026-09-23.

- Production ONNX SHA-256:
  `f598a0eb676f6bf7a42c1a46a4ff31615f6b8fc36d730edd1cfe8391d5bceecd`
- The ONNX bytes are unchanged from the qualified ten-gesture release.
- The complete Python backend test suite passed after the Down geometry update.
- The four `reverse_palm_edge_down` landmark variants pass the geometry and
  temporal recovery tests in `reference_python/tests/test_geometry.py`.
- An 8,331-row held-out sample replay produced zero new triggers from the new
  rule.
- The Swift resolver includes a two-variant startup regression check covering
  both possible MediaPipe outer-edge finger assignments.

Swift compilation was not run in this Windows workspace because an Apple Swift
toolchain is not installed here. The iOS developer must add the three supplied
Swift files to the Xcode target and run the startup checks and physical-device
acceptance list in the package README.
