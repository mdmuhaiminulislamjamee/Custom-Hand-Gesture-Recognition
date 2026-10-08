# Android acceptance checklist

## Verified while preparing this delivery

- Source ONNX SHA-256 matches the model metadata and the iOS NCM reference delivery.
- The package builder runs the repository ONNX preflight and checks all delivered payload hashes.
- The package verifier checks exact model identity and the package contents.
- The desktop reference and Android asset copies contain the same model files.

These are host-side packaging/model checks. They do **not** establish that the Kotlin sources compile in the app or that an Android phone exposes a USB-NCM route.

## Required on the app team's Android build

1. Resolve exact MediaPipe Tasks Vision and ONNX Runtime Android dependency versions; record them in the app's Gradle lock/version catalog. Build an `arm64-v8a` app with `minSdk` at least 24.
2. Put the five model/configuration files in `src/main/assets`. Run `GestureParity.run(context)` in a debug build: 294 feature cases and 26 ONNX cases must pass at tolerance `0.0001`.
3. On the intended phone/tablet and USB cable, confirm the board enumerates as an Android network interface. Confirm the link has a route to `192.168.50.2:5000`; select the corresponding `Network` for the socket.
4. Check JLIP version, payload length, CRC, JPEG decode, heartbeat survival, disconnect/reconnect, and rising valid-frame count. Verify the pixels are upright and unmirrored.
5. Compare the same NCM frames against the Windows dashboard's 0.92 center crop and 10 FPS policy. Measure latency and memory on the target device.
6. Port and run the full geometry/face/recovery/landmark-stabilization/temporal/action rules from the current reference. Compare all 294 geometry expectations, hold behavior, all 12 gestures and negative/rejected poses before enabling actions.
7. Test left and right hands, distance, varied angle and light, empty frames, tracking loss, long holds, repeat release, and all twelve app action mappings. Confirm no stale action after reconnect or background/foreground transition.

`GestureFrameAnalyzer` emits diagnostic probabilities only. Treat an action-dispatch build as incomplete until steps 3–7 pass on the actual Android/NCM hardware.
