# Candidate validation, 29 October 2026

The original deployed ONNX SHA-256 is
`11d6991cc555dfd5f490a647270c4f326dd9dfb14ce4ad1cf503300aea29c331`.
The new default FP32 TFLite SHA-256 is
`712c1fa0689370861ec42ff649117853fffb6cacf885f7d3b570c3e3acbb3c1a`.
The conversion copies the learned weights; no retraining was performed.

| Check | Result | Scope |
|---|---|---|
| FP32 model conversion | Passed | All 9,031 validation and 9,042 test rows retain the same top label and scalar gate decision |
| Maximum FP32 probability error | 0.00000140071 | Compared with original ONNX on both caches |
| Maximum FP32 known-mass error | 0.00000175834 | Compared with original ONNX on both caches |
| Default model size | 78,960 bytes | TFLite builtin operations; no Flex dependency |
| Canonical feature/geometry regressions | 1,142/1,142 passed | Features, adjusted rows, final decisions and geometry support; includes 98.3% incorrect thumb-label cases |
| Canonical known-mass qualifications | 5,710/5,710 passed | Includes strict low-mass recoveries and rejection |
| Additional held-out cache geometry | 400/400 passed | All twelve commands and hard negatives; forced classifier labels isolate geometry |
| Native Flutter LiteRT inference | 26/26 reference cases passed | Actual persistent interpreter/isolate, maximum probability error about 0.000000970 |
| Flutter tests | 48 runtime + 2 tester passed | Includes palm/back surface checks for both physical hands and mirror modes, controller, parser, coordinates, TCP lifecycle and preview |
| Dart analysis | No issues | Package, tests and NCM example |
| Detector provenance | Hash checked | 40 source/config/asset files copied from supplied dynamic archive |

The complete test suite exercised real LiteRT inference and a local TCP/JLIP
server. The hand detector was injected in lifecycle tests. OpenCV's native
Windows build hook was temporarily skipped because these tests do not execute
OpenCV. Normal native builds are enabled in the delivered pubspec. See
`HOST_TESTING.md` for the reproducible host-test configuration.

The `fix1` host run used one Flutter test file per invocation to keep memory
within the Windows host's available commit limit. All runtime/tester test files
passed. Surface tests run detector-adapter output through geometry using
explicit physical Left/Right fixtures. The extra geometry handedness swap was
removed, and mirrored detector input is normalized to unmirrored image/world X.
The shared parity runner now supplies each Python fixture's physical hand
directly. Thumb regressions reconstruct the reported side view; screenshots
were not replayed through MediaPipe. See `validation/thumb_surface_checks.json`.

## INT8 candidate

The candidate is 23,240 bytes and uses calibrated INT8 dense layers, with FP32
normalization and the open-set probability tail. It is not fully integer.

Validation top-label agreement is 99.3799%, below the required 99.5%. Validation
unknown false acceptance also increases from 1.2746% to 1.6222%, exceeding the
allowed 0.2 percentage point increase. Its manifest therefore has
`qualification.passed: false`, and the runtime refuses to load it. No thresholds
were relaxed to accept this candidate. FP32 retains the original model's
predictions and is the delivered default.

## Boundaries of the evidence

The scalar gate comparison uses confidence 0.70, known mass 0.70, margin 0.08
and disabled Dorsal rejection. It does not include pose geometry, the new
controller timing, tracking or action dispatch. Those are tested separately.

The canonical geometry fixture contains synthetic and reconstructed landmark
cases, not new camera images. The additional cache cases have no original
source dimensions or world depth, so they use declared square-frame dimensions
and offline Fist geometry. Separate world-coordinate rotation and live-depth
requirements are covered by adapter/geometry tests.

Calibration used 1,391 stratified rows from the existing replay cache. Validation
and untouched test rows were not used to compute quantization ranges. Detailed
file hashes and numerical results are in `validation/conversion_report.json`.

Host inference timing in that report is only native invocation time. It does
not measure iPhone performance, JPEG transport, detection, display rendering or
capture-to-command latency, and must not be used to claim a mobile speedup.

iOS compilation, actual GPU/CPU detector selection, NCM accessory routing,
lighting/distance accuracy, sustained FPS, heat/power and false commands per
minute must be measured in the consuming app on physical iPhones. The package
includes the example and aggregate timing output needed to run those checks
without sharing private application data.
