# Boss static gesture handover - 14 November 2026 two-hand stability

Use `Boss_Static_Gesture_Handover_20261114_two_hand_stability.zip`. It supersedes
the `feedback_refresh` delivery and includes the shared Flutter runtime,
iOS integration source, GestureLab 2.0.12 source and matching Android test APK.
Earlier archives remain unchanged.

Read [README.md](README.md) for integration and [RELEASE_NOTES.md](RELEASE_NOTES.md)
for this revision. Keep the detector, feature extraction, geometry, model and
command controller together. `source_model/` contains the exact production ONNX,
configuration, metadata and integrity manifest used for mobile conversion.

The adapter now keeps already identified, spatially separated hands when
Left/Right confidence temporarily falls below .75. It uses the existing track's
confirmed side with the current frame's landmarks. Weak labels alone cannot
create a new identity or revive one after the existing 350 ms expiry or reset.
The overlap guard still abstains when the hands cannot be distinguished.

The prior mobile model binaries and matching source snapshot are byte-for-byte
preserved. No retraining or model conversion was performed. The .75 classifier
confidence, .10 margin and .70 known-mass gates are retained, together
with strict directional entry, bounded 200 ms continuation, metric-world Right
support, geometry rejection, independent hand state and the existing false-hand
policy. Ghost detection tuning is deferred to a later task.

The persistent classifier, latest-frame replacement, 10 FPS admission policy,
immediate strong confirmation, ordinary two-of-three confirmation and zero
fixed action cooldown are retained. No detector/model pass was added. Host
timings are in `validation/onnx_latency.json` and `validation/latency_comparison.json`.
Physical phone/NCM camera-to-command latency still requires measurement.

`static_gesture_runtime/validation/two_hand_stability_release.json` records the
source/asset comparison with the prior archive. Fresh Flutter and APK checks
are in `HOST_CHECKS.json`. Earlier training/conversion reports apply to the
preserved weights; their figures are not new live-camera measurements.

## Integrate

1. Extract the complete archive, preserving sibling runtime/tester directories.
2. Run `python verify_handover.py` from `Boss_Static_Gesture_Handover_20261114_two_hand_stability/`.
3. Follow `static_gesture_runtime/IOS_INTEGRATION.md` and the root README.
4. Initialize the static runtime once, feed fresh detector results with monotonic
   timestamps, and dispatch actions through `onCommand`. Reset on reconnect or
   orientation changes. Reuse the compatible supplied detector.
5. Build/sign the iOS host on a Mac and complete `DEVICE_ACCEPTANCE.md`.

The Android tester is
`flutter_gesture_tester/releases/GestureLab-v2.0.12-litert-ncm-arm64-diagnostic-release.apk`.
It targets ARM64 Android API 24+, uses release Dart, disables Android shrinking
for diagnostics and uses the Android debug signing key. It displays commands;
the consuming product supplies actions. Its direct USB fallback is board-specific.

`HOST_CHECKS.json` binds the current model, source and APK to fresh host checks.
An IPA, iOS compilation and physical iPhone/NCM qualification are not included.
The native filter update remains a reference alongside the complete Flutter path.
