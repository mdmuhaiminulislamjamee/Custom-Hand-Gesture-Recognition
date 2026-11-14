# Boss static gesture handover - 14 November 2026 ownership and preview

Use `Boss_Static_Gesture_Handover_20261114_ownership_preview.zip`. It supersedes
the `two_hand_stability` delivery and includes the shared Flutter runtime,
iOS integration source, GestureLab 2.0.13 source and matching Android test APK.
Earlier archives remain unchanged.

Read [README.md](README.md) for integration and [RELEASE_NOTES.md](RELEASE_NOTES.md)
for this revision. Keep the detector, feature extraction, geometry, model and
command controller together. `source_model/` contains the exact production ONNX,
configuration, metadata and integrity manifest used for mobile conversion.

The runtime keeps the first qualified hand selected when another appears
beside it. A closer challenger or the only valid peer gesture must confirm
across three processed observations spanning at least 250 ms. Closeness uses
coherent world/image projection (1.10 ratio) or consistent palm size (1.15
ratio), with .50 smoothing. Commands pause during confirmation; the transfer
frame cannot dispatch, and fresh evidence is required afterwards. Missing
owners get 350 ms to return. Simultaneous equally distant hands wait for a
clear choice. `selectedHand`, `switchPending` and `selectionReason` are exposed
in each result. Display the selected hand; dispatch only `onCommand` events.

The NCM preview retains the old decoded picture AND its matching overlay while
the next JPEG loads, then swaps them together. Genuine empty-hand frames clear
the overlay on display; disconnect/reset must clear the saved preview.

The adapter also keeps already identified, spatially separated hands when
Left/Right confidence temporarily falls below .75. It uses the existing track's
confirmed side with the current frame's landmarks. Weak labels alone cannot
create a new identity or revive one after the existing 350 ms expiry or reset.
The overlap guard still abstains when the hands cannot be distinguished.

The prior mobile model binaries and matching source snapshot are byte-for-byte
preserved. No retraining or model conversion was performed. The .75 classifier
confidence, .10 margin and .70 known-mass gates are retained, together
with strict directional entry, bounded 200 ms continuation, metric-world Right
support, geometry rejection, independent hand state and the existing false-hand
policy. Ghost/ear detection and incidental-movement tuning are deferred.

The persistent classifier, latest-frame replacement, 10 FPS admission policy,
immediate strong confirmation, ordinary two-of-three confirmation and zero
fixed action cooldown are retained for single-hand recognition. Ownership
switches add the confirmation/fresh-evidence requirements above. No detector/model pass was added. Host
timings are in `validation/onnx_latency.json` and `validation/latency_comparison.json`.
Physical phone/NCM camera-to-command latency still requires measurement.

`static_gesture_runtime/validation/ownership_preview_release.json` records the
source/asset comparison with the prior archive. Fresh Flutter and APK checks
are in `HOST_CHECKS.json`. Earlier training/conversion reports apply to the
preserved weights; their figures are not new live-camera measurements.

## Integrate

1. Extract the complete archive, preserving sibling runtime/tester directories.
2. Run `python verify_handover.py` from `Boss_Static_Gesture_Handover_20261114_ownership_preview/`.
3. Follow `static_gesture_runtime/IOS_INTEGRATION.md` and the root README.
4. Initialize the static runtime once, feed fresh detector results with monotonic
   timestamps, and dispatch actions through `onCommand`. Reset on reconnect or
   orientation changes. Reuse the compatible supplied detector.
5. Build/sign the iOS host on a Mac and complete `DEVICE_ACCEPTANCE.md`.

The Android tester is
`flutter_gesture_tester/releases/GestureLab-v2.0.13-litert-ncm-arm64-diagnostic-release.apk`.
It targets ARM64 Android API 24+, uses release Dart, disables Android shrinking
for diagnostics and uses the Android debug signing key. It displays commands;
the consuming product supplies actions. Its direct USB fallback is board-specific.

`HOST_CHECKS.json` binds the current model, source and APK to fresh host checks.
An IPA, iOS compilation and physical iPhone/NCM qualification are not included.
The native filter update remains a reference alongside the complete Flutter path.
