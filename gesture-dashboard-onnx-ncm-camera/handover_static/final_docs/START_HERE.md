# Boss static gesture handover - 11 November 2026

Use `Boss_Static_Gesture_Handover_20261111_right_stability.zip`. It supersedes
the earlier 11 November direction handover and includes the Right-retrained model,
shared Flutter runtime, iOS integration source, GestureLab 2.0.10 source and test APK.
Previous archives remain unchanged.

Read [README.md](README.md) for integration and [RELEASE_NOTES.md](RELEASE_NOTES.md)
for this revision. Keep the complete runtime and detector together: copying the
model alone omits feature extraction, geometry, rejection and command state.

Only the Right output weights were retrained using the saved reviewed landmark
feedback, including 331 new Right poses and 90 new hard negatives. Normalization,
hidden layers and all other gesture output weights remain identical.
Its qualified production ONNX is converted to TFLite without further training. FP32 remains the
default. The release manifest binds the converted model, fixtures, source ONNX
and APK to their SHA-256 identities. `source_model/` contains the matching ONNX,
metadata, runtime JSON and artifact manifest.

The previous speed policy is retained: a persistent classifier, one active and
one replaceable pending frame, up to 10 FPS, immediate confirmation of strong
valid predictions and two of three confirmation otherwise. Strict geometry
recoveries retain their hold requirements. There is no fixed action cooldown.
See `static_gesture_runtime/validation/latency_comparison.json` for the same-host
classifier comparison. Physical camera-to-command latency still needs measurement
on the intended phone and NCM camera.

Left/Right/Up/Down retain strict entry. A confirmed direction can survive one
marginal visible-hand frame within 200 ms, with continuation alignment >= .65.
Two consecutive invalid observations release the direction; missing hands,
opposite axes and competing gestures cannot use that grace. Existing positive
3-D single-index turn support is preserved separately. A strict model-backed
3-D check corrects the Left/Right versus Dorsal conflict and requires full known
mass. No additional prediction panels are introduced.

The additional Right side-view correction requires metric world landmarks,
strong isolated-index geometry, model confidence >= .95 and full known mass.
It prevents folded fingers that look straight in 2-D from invoking the open-palm
angle veto. Missing depth and competing fingers cannot use this entry exception.
The existing one-frame/200 ms continuation bounds apply; grace emits no command.

`static_gesture_runtime/validation/right_stability_release.json` records the
training, JPEG replay, baseline comparison, timing and regression results.

## Integrate

1. Extract the complete ZIP, keeping the sibling runtime/tester directories.
2. Run `python verify_handover.py` from `Boss_Static_Gesture_Handover_20261111_right_stability/`.
3. Follow `static_gesture_runtime/IOS_INTEGRATION.md` and the root README.
4. Reuse the existing compatible detector where available. Initialize the static
   runtime once, feed fresh results with monotonic timestamps and dispatch only
   through `onCommand`. Reset it after reconnect/orientation changes.
5. Build/sign the iOS host on a Mac and complete `DEVICE_ACCEPTANCE.md`.

The Android tester is
`flutter_gesture_tester/releases/GestureLab-v2.0.10-litert-ncm-arm64-diagnostic-release.apk`.
It targets ARM64 Android API 24+, uses release Dart code, disables Android
shrinking for this diagnostic build and is signed with the Android debug key.
It displays commands; the consuming app supplies product actions. Its direct
USB fallback is board-specific and requires physical acceptance.

`HOST_CHECKS.json` records current host checks and their limits. The iOS example
is source for the app team to build and sign. An IPA and physical iPhone/NCM
qualification are not included. The retained `native_false_hand_update/` is a
native filtering reference, not a complete native replacement for the Flutter
pipeline.
