# Current model delivery - 11 November 2026

- Retrained from 3,416 usable saved feedback landmark records, with 11 invalid
  rows excluded. The installed ONNX is
  `7e3dbcd61b55d2ba386059d2634ef4a737747b454c28cf200d1f117d976064ed`.
  The previous production bundle is backed up in the source repository.
- Added strict entry and bounded continuation for Left/Right/Up/Down. One
  marginal frame may retain a confirmed direction within 200 ms; two invalid
  observations release it. Clear opposite directions and other gestures can
  enter immediately. Missing hands and multi-finger contradictions cannot
  retain a direction. Existing positive 3-D single-index turns are preserved.
- Added a strict model/3-D single-index exception to incorrect generic Dorsal
  rejection on Left/Right. It requires full known mass and cannot grant a new
  low-mass recovery. No extra prediction panels were added.
- JPEG replay improved from 358/408 to 375/408 correct final directions and
  362/408 to 378/408 correct actions. These are training feedback regressions.
- Matched current scalar gates: confidence .75, margin .10, known mass .70.
  Regenerated conversion, model and geometry fixtures for this source.
- Retained the previous speed policy: 100 ms minimum admission spacing,
  newest pending frame replacement, persistent one-thread classifier,
  immediate strong valid predictions, ordinary two of three confirmation,
  independent hand state and no fixed action cooldown. Geometry recoveries
  keep their existing hold requirements.
- GestureLab 2.0.9 retains the continuous NCM preview, USB descriptor
  diagnostics and board-specific USB-NCM fallback with forced interface claims.
  Camera admission uses monotonic time; paused or replaced sessions discard
  in-flight results. Repeated commands after a valid release appear in history.
- Retained face/ear/background filtering before classification and overlays,
  Open Palm/Stop Inverted surface correction and Like/Thumb Down direction
  correction. The same detector/runtime serves phone and NCM sources.
- Included current Android source and a rebuilt ARM64 diagnostic release APK,
  matching runtime assets and manifests, exact source ONNX/configuration,
  integration guides and extracted-payload verification.
- FP32 remains the only integration default. The refreshed INT8 candidate
  passes offline conversion gates but remains blocked pending its own geometry,
  controller and mobile qualification.

`HOST_CHECKS.json` records fresh checks and source hashes.
`static_gesture_runtime/validation/direction_stability_release.json` includes
class-by-class comparisons, 4,903 passing backend tests, six remaining baseline
failures, zero new backend failures, and 74 runtime plus 8 tester Flutter tests.
One previous Left regression now passes. The backend is not described as fully
passing. Individual class rates can move by a few samples after retraining.
`static_gesture_runtime/validation/latency_comparison.json` compares old/current
classifier invocation on the same host. It excludes detection, camera transport,
rendering and action confirmation and does not establish mobile latency.
Classifier medians are 0.0048 ms previous and 0.0046 ms current. Rule medians
are 3.0339 ms previous and 3.02875 ms current. No extra inference pass was added.

The APK uses release Dart code with Android shrinking disabled and the Android
debug signing key for testing. iOS build/signing and physical iPhone/NCM tests
remain app-team acceptance steps. The native filter update is retained as a
reference; the complete integration path in this delivery is Flutter/LiteRT.
