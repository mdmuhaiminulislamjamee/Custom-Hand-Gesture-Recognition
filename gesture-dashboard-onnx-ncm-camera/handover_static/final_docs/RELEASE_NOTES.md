# Current model delivery - 11 November 2026

- Retrained only the 65 Right output parameters using 3,837 usable feedback
  landmarks, including 331 new Right poses and 90 new hard negatives. Eleven
  invalid rows were excluded. Normalization, hidden layers and all other output
  columns remain bit-identical. The installed ONNX is
  `aa3846307c53ee4e64f0a9734b2eefce21b7bf819f43f145a5095a26706f0992`.
  The previous production bundle is backed up in the source repository.
- Retained strict entry and bounded continuation for Left/Right/Up/Down. One
  marginal frame may retain a confirmed direction within 200 ms; two invalid
  observations release it. Clear opposite directions and other gestures can
  enter immediately. Missing hands and multi-finger contradictions cannot
  retain a direction. Existing positive 3-D single-index turns are preserved.
- Retained the strict model/3-D single-index exception to incorrect generic Dorsal
  rejection on Left/Right. It requires full known mass and cannot grant a new
  low-mass recovery. No extra prediction panels were added.
- Added a Right side-view correction for folded fingers wrongly triggering the
  generic open-palm angle veto. Entry requires metric world landmarks, no
  competing-finger contradiction, a strongly isolated straight index, confidence
  >= .95 and full known mass. Missing depth cannot use this new entry path.
  Existing one-frame/200 ms grace bridges a marginal result without authorizing
  a new command. Global thresholds and other gesture rules are unchanged.
- Replayed all 739 saved Left/Right JPEGs. Per-class final-direction and action
  counts are in `validation/right_stability_release.json` inside the runtime.
  Saved images and transformed poses are training regression evidence.
- Matched current scalar gates: confidence .75, margin .10, known mass .70.
  Regenerated conversion, model and geometry fixtures for this source.
- Retained the previous speed policy: 100 ms minimum admission spacing,
  newest pending frame replacement, persistent one-thread classifier,
  immediate strong valid predictions, ordinary two of three confirmation,
  independent hand state and no fixed action cooldown. Geometry recoveries
  keep their existing hold requirements.
- GestureLab 2.0.10 retains the continuous NCM preview, USB descriptor
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
`static_gesture_runtime/validation/right_stability_release.json` includes exact
non-Right preservation checks on both held-out splits, old feedback and new hard
negatives, per-class comparisons, backend failures and current Flutter counts.
The six existing backend failures were reproduced before this change; they
remain explicit rather than being presented as a fully passing backend suite.
The extra Right fixtures check 96 mass/shape combinations and confidence-dip,
release and rearming behavior. The v2.0.9 direction report is historical.
`static_gesture_runtime/validation/latency_comparison.json` compares old/current
classifier invocation on the same host. It excludes detection, camera transport,
rendering and action confirmation and does not establish mobile latency.
Current measured medians and percentiles are retained in the timing JSON.
No extra inference pass was added; the FP32 model remains 78,960 bytes.

The APK uses release Dart code with Android shrinking disabled and the Android
debug signing key for testing. iOS build/signing and physical iPhone/NCM tests
remain app-team acceptance steps. The native filter update is retained as a
reference; the complete integration path in this delivery is Flutter/LiteRT.
