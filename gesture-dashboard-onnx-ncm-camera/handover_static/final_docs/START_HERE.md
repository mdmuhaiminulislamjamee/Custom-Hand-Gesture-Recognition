# Boss static gesture handover - 11 November 2026 feedback refresh

Use `Boss_Static_Gesture_Handover_20261111_feedback_refresh.zip`. It supersedes
the earlier `right_stability` delivery and includes the shared Flutter runtime,
iOS integration source, GestureLab 2.0.11 source and matching Android test APK.
Earlier archives remain unchanged.

Read [README.md](README.md) for integration and [RELEASE_NOTES.md](RELEASE_NOTES.md)
for this revision. Keep the detector, feature extraction, geometry, model and
command controller together. `source_model/` contains the exact production ONNX,
configuration, metadata and integrity manifest used for mobile conversion.

Training used 990 previously unused reviewed landmark records: 480 Down,
240 Thumb Down, 150 Stop Inverted, 90 Like and 30 rejection examples.
Their labels were preserved. Normalization, hidden layers, graph topology and
unaffected output columns are frozen. Larger candidate fits were rejected when
preservation checks failed. The accepted step retains every per-class metric
on the fixed regression test split. Its improvement on new training samples is
modest and does not establish generalization to new people.

The .75 confidence, .10 margin and .70 known-mass gates are retained, together
with strict directional entry, bounded 200 ms continuation, metric-world Right
support, geometry rejection, independent hand state and the existing false-hand
policy. Ghost detection tuning is deferred to a later task.

The persistent classifier, latest-frame replacement, 10 FPS admission policy,
immediate strong confirmation, ordinary two-of-three confirmation and zero
fixed action cooldown are retained. No detector/model pass was added. Host
timings are in `validation/onnx_latency.json` and `validation/latency_comparison.json`.
Physical phone/NCM camera-to-command latency still requires measurement.

`static_gesture_runtime/validation/feedback_refresh_release.json` records training,
rejected alternatives, per-class comparisons, image regressions, timing and
test results. The existing six backend failures remain listed with their
baseline; a fully passing backend suite is not claimed.

## Integrate

1. Extract the complete archive, preserving sibling runtime/tester directories.
2. Run `python verify_handover.py` from `Boss_Static_Gesture_Handover_20261111_feedback_refresh/`.
3. Follow `static_gesture_runtime/IOS_INTEGRATION.md` and the root README.
4. Initialize the static runtime once, feed fresh detector results with monotonic
   timestamps, and dispatch actions through `onCommand`. Reset on reconnect or
   orientation changes. Reuse the compatible supplied detector.
5. Build/sign the iOS host on a Mac and complete `DEVICE_ACCEPTANCE.md`.

The Android tester is
`flutter_gesture_tester/releases/GestureLab-v2.0.11-litert-ncm-arm64-diagnostic-release.apk`.
It targets ARM64 Android API 24+, uses release Dart, disables Android shrinking
for diagnostics and uses the Android debug signing key. It displays commands;
the consuming product supplies actions. Its direct USB fallback is board-specific.

`HOST_CHECKS.json` binds the current model, source and APK to fresh host checks.
An IPA, iOS compilation and physical iPhone/NCM qualification are not included.
The native filter update remains a reference alongside the complete Flutter path.
