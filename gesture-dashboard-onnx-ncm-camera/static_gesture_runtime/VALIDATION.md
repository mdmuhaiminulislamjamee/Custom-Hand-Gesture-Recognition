# Direction stability validation, 11 November 2026

The model was retrained from 3,416 usable saved feedback landmark records;
11 invalid rows were skipped. The full direction JPEG replay improved from
358/408 to 375/408 correct final directions and from 362/408 to 378/408 correct
actions. Every model installation gate passed. Saved feedback is training
evidence, not independent camera validation.

`validation/direction_stability_release.json` records baseline comparisons,
class-by-class metrics, regression failures and timing. The new rules preserve
the previous model's stateless predictions on the held-out landmark sets.
Retraining improves aggregate metrics; individual class rates can shift by a
few samples, so the complete per-class results are retained.

Current Flutter validation passes 74 runtime and 8 tester tests, including 96
model-backed pointing/mirror/low-mass/world-conflict checks and direction event
sequences. Native Android USB checks pass 17 groups and an independent wire
oracle. Existing backend failures are reported with their pre-change baseline;
they are not presented as a fully passing backend suite.

The historical false-hand update passed 52 runtime tests and 2 tester tests, including the
real face model's tensor/blank-image contract and 42 Python/Dart policy cases.
The native Kotlin filter and detector wrapper compile against MediaPipe 1.1.1
and pass the same 42 Python/Kotlin cases. Swift policy fixtures and a Mac runner
are supplied; iOS compilation and physical-device tests remain pending.
The Dart face decoder also matches the expected face regions on 12 saved
face/ear images (box IoU above 0.5) and reports no face on 24 saved background
images, using actual TFLite model outputs. This checks face preprocessing and
decoding, rather than mobile hand-detection or action accuracy.
The detector checks above describe the 31 October revision. Fresh host results
for the current runtime/tester are recorded in the complete delivery's
`HOST_CHECKS.json`; retain physical-device acceptance as a separate step.

The original deployed ONNX SHA-256 is
`7e3dbcd61b55d2ba386059d2634ef4a737747b454c28cf200d1f117d976064ed`.
The new default FP32 TFLite SHA-256 is
`a0d8d4df9c55ae68e21e90733253a365862ae89ceb666975d0e0e69655ed12d2`.
The source was retrained; conversion copies its learned weights without further training.

| Check | Result | Scope |
|---|---|---|
| FP32 model conversion | Passed | All 9,031 validation and 9,042 test rows retain the same top label and scalar gate decision |
| Maximum FP32 probability error | 0.00000232459 | Compared with current ONNX on both caches |
| Maximum FP32 known-mass error | 0.00000160933 | Compared with current ONNX on both caches |
| Default model size | 78,960 bytes | TFLite builtin operations; no Flex dependency |
| Canonical feature/geometry regressions | 1,142/1,142 passed | Features, adjusted rows, final decisions and geometry support; includes 98.3% incorrect thumb-label cases |
| Canonical known-mass qualifications | 5,710/5,710 passed | Includes strict low-mass recoveries and rejection |
| Additional held-out cache geometry | 400/400 passed | All twelve commands and hard negatives; forced classifier labels isolate geometry |
| Native Flutter LiteRT inference | 26 current reference cases supplied | Actual persistent interpreter/isolate check is rerun in current `HOST_CHECKS.json` |
| Flutter tests and Dart analysis | Current results in `HOST_CHECKS.json` | Includes palm/back surfaces, controller, parser, coordinates, TCP lifecycle, latest-frame handling and preview |
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

With the current learned weights, the INT8 conversion passes the existing
offline gate; exact current label agreement is in `validation/conversion_report.json`.
This does not promote INT8 into the app. Its manifest separately
records `conversion_passed: true` and `qualification.passed: false` because
precision-specific geometry/controller and target-device acceptance are pending.
The runtime continues to reject INT8 loading. FP32 retains the current ONNX
predictions exactly at the scalar decision level and remains the default.

## Boundaries of the evidence

The scalar gate comparison uses confidence 0.75, known mass 0.70, margin 0.10
and disabled Dorsal rejection. It does not include pose geometry, the new
controller timing, tracking or action dispatch. Those are tested separately.
The converter now reads these gates from the current runtime JSON and records
that file's hash, rather than qualifying new weights with historical thresholds.

The current horizontal-pointing, side-view Like, and strict Fist thumb-radius
and PIP-overshoot rules have been ported to Dart. Canonical and refreshed cache
geometry checks pass all rows and geometry-support decisions.

## Speed comparison

`validation/latency_comparison.json` records 4,000 timed warm CPU invokes for
each FP32 model on identical inputs, alternating the previous/current model
order. Medians are 0.0048 ms previous and 0.0046 ms current; P95 is 0.0063 ms
previous and 0.0059 ms current. Both models are 78,960 bytes. These are same-host native invocation
measurements, not camera or phone latency. The 100 ms admission interval,
single pending-frame slot, persistent interpreter, immediate strong-result
confirmation and ordinary two-of-three consensus remain unchanged. No fixed
action cooldown has been introduced. Strict geometry recovery still has holds.

`validation/direction_rule_latency.json` separately compares geometry: median
3.0339 ms previous and 3.02875 ms current, with P95 4.81907 ms and 4.70015 ms.
The new continuation state update has P95 0.0008 ms. No extra detector or
classifier pass was added. The measurements do not establish physical-phone
performance or zero camera latency.

The NCM runtime now reports each completed frame's `receiveToResultMs`, including
pending wait, so the tester can display current local pipeline latency. The
board and host clocks remain unsynchronized, so this excludes capture-to-receipt
time and cannot establish zero latency.

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
