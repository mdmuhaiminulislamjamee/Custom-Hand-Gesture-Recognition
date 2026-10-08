# Validation and device acceptance

## Checks completed on Windows, 7 October 2026

| Check | Result |
|---|---|
| Backend suite, including current reported-pose regressions | 483 passed; two dependency deprecation warnings |
| Production artifact hashes, 12-class contract and graph load | PASS |
| MediaPipe initialization and 76-feature inference smoke | PASS |
| Python installed dependency consistency | PASS |
| TypeScript noEmit | PASS after clearing one obsolete generated Next.js validator |
| ESLint | PASS |
| Production dashboard build | PASS; bundler reported nonfatal dynamic-import warnings |
| Isolated desktop handover test suite | 483 passed from the staged package using the host test interpreter; clean dependency installation not rerun |
| Regression-fixture export | 290 source tests passed; 294 distinct geometry cases and 26 ONNX cases exported |

Command outputs are in `validation/`. These are software checks, not a new physical-camera accuracy measurement. The source metadata records offline qualification and still marks live-camera validation as pending.

## Checks required on the Mac/device

No Swift compiler, Xcode, CocoaPods iOS build, iPhone or connected NCM hardware was available in this release environment. The following remain **not run**:

- `sh ios/swift/tests/verify_swift.sh`, including all geometry/feature/temporal parity cases.
- Build/link the native adapters against resolved CocoaPods versions.
- Execute the 26-case ONNX comparison on the physical device.
- Port/compare the complete frame preparation, detection recovery, face guard and landmark smoothing.
- Verify iOS USB-NCM networking and live gesture/action performance.
- Test app signing, permissions, lifecycle and product action handlers.

A model fixture PASS establishes tensor-level parity only. Core fixture PASS establishes parity on those supplied cases only. Neither is a substitute for live, end-to-end acceptance.

## Acceptance record

For each tested device record: release/model hash, app build, Xcode/Swift version, Podfile.lock versions, device model, iOS version, camera/accessory/firmware, crop/orientation policy, lighting and test date.

1. Verify all five app resources and run both parity suites. Retain output.
2. Check unmirrored image semantics: positive index x gives Left and negative x gives Right, without another UI swap.
3. Test every command on both hands across realistic angles/distances and normal/low light. Include Like versus Fist/Thumbs Down, current Fist accepted/negative views, a prominent Down index, close-finger Peace versus a lone middle finger, Dorsal 0–150-degree range, sideways Open Palm, and OK with only index-thumb contact.
4. Test empty scenes, face/ear contact, partly visible/clipped hands, motion blur, relaxed/bent hands, retired thumb-and-little pose, camera obstruction and tracking loss. Rejections must produce no action.
5. Hold a command: fire once. Release it and present it again: rearm after release/cooldown. Brief detector dropouts must not retrigger a held toggle.
6. Exercise permission denial/regrant, camera switch, rotation, foreground/background, stream disconnect/reconnect and NCM unplug/replug. Do not replay stale frames/actions.
7. Measure end-to-end action latency (median/p95), delivered inference FPS, 100-ms budget rate, preview responsiveness, memory, battery/thermal behavior and queue age.
8. Measure a labeled live confusion matrix, per-class precision/recall, false actions per minute with no command shown, and rejected-frame reasons. Offline model accuracy is not live iOS accuracy.
9. Verify each action reaches the app's intended handler exactly once; confirm product-specific tracking/recording/media semantics.
10. App and camera owners review the recorded results against their agreed release criteria. Do not label iOS production-ready before these results exist.

Use the complete Python frame pipeline as the behavior reference. If parity fails, compare in order: prepared pixels → selected image/world landmarks and handedness → stabilized landmarks → 76 features → named ONNX outputs → geometry flags → temporal decision → latched action.
