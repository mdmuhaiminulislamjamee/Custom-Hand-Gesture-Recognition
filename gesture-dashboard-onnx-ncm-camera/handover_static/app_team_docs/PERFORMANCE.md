# Preserve the tested performance and behavior

This is a packaging-only release. Every file under `flutter/`, including the
example, detector, model assets, tests, dependency specifications and lockfiles,
is identical to the tested ownership/preview handover. The optional phone
helpers are also copied unchanged from that tested release. No model conversion,
retraining, extra detector/classifier pass or new per-frame work was added.

`validation/app_team_package_preservation.json` lists the exact protected hashes.
The original checks are retained in
`reference/qualified_release/HOST_CHECKS.json`: 86 runtime tests, 10 tester tests,
model fixture checks, analysis and Android build verification. The historical
tester build is evidence only; no tester app or APK is delivered here.

## Integration choices that preserve speed

1. Reuse one camera stream and one persistent detector. If an existing dynamic
   pipeline supplies compatible filtered landmarks, use that pipeline and
   `StaticDetectorAdapter`. Do not start the NCM wrapper or phone helper as a
   second detector on the same frames.
2. Initialize the detector/classifier once per active session. Retain the
   persistent workers; do not create interpreters inside frame callbacks.
3. Preserve the 100 ms minimum admission interval (up to 10 FPS), one active
   frame and one latest pending frame. The NCM wrapper implements this. Use
   `PhoneFrameGate` with the optional phone helper. With an existing detector,
   preserve the host pipeline's equivalent admission/backpressure policy.
4. Keep the full FP16 detector models, GPU preference with observable CPU
   fallback, and the persistent one-thread CPU FP32 static classifier.
5. Keep the supplied model/geometry/scalar gates and confirmation code. A
   different delegate, resolution, crop, dependency version or queue policy
   requires acceptance in the app where it is changed.
6. Keep command handlers lightweight. Dispatch work to the app's existing
   transport/action layer. Avoid blocking inference on I/O, image logging or
   repeated heavyweight UI work.
7. Use `StaticNcmPreview` for the NCM image/overlay pair. It preserves the last
   matching image and overlay while the next JPEG loads. Use `onFrame` without
   `onPreview` if preview is not required. Never redetect a frame just to draw it.

## Timing rules retained

| Behavior | Current rule |
|---|---|
| Frame admission | At most one start every 100 ms; actual throughput depends on processing |
| Strong prediction | Immediate confirmation after scalar/geometry qualification, confidence >= .92 and margin >= .25; known mass >= .70 |
| Ordinary prediction | Two matching observations among the latest three |
| Low-known-mass geometry recoveries | Existing 400-500 ms holds, depending on recovery |
| Hand switch | At least three consistent observations spanning 250 ms, then fresh gesture evidence; no command on transfer frame |
| Missing owner | 350 ms return grace |
| Repeating the held command | Suppressed until confirmed no-gesture release; handoff/reset has its documented reset behavior |
| Fixed action cooldown | None |

Switching hands and changing gestures on the same hand have different timing.
Do not add the handoff hold to every same-hand gesture transition.

## Measure in the consuming app

Unchanged source/assets prevent a performance change caused by repackaging.
They cannot guarantee identical FPS on every phone or after host integration.
Use release mode on the target iPhone/Android device and compare the same
camera resolution, lighting, backend and preview setting before and after
integration. Record achieved FPS, p50/p95 detector and decision times, frame
replacements, false/repeated commands, heat and sustained-session behavior.

`onTelemetry` exposes these NCM aggregates. Each preview frame also exposes
`receiveToResultMs`. Receiver timings exclude camera exposure and unknown USB
transit time. `validation/latency_comparison.json` is a host model-invoke
benchmark, not a target-phone camera-to-command latency measurement.

Physical iOS compilation, signing, camera/NCM compatibility and sustained
performance remain app-team acceptance tasks in `DEVICE_ACCEPTANCE.md`.
Movement suppression and further ear/ghost filtering have not been introduced.
