# Gesture Lab — Flutter LiteRT tester

This Android tester now runs the same static runtime delivered in
`static_gesture_runtime/flutter`:

`phone camera or USB-NCM JPEG -> dynamic two-hand detector -> 76 features -> qualified FP32 TFLite classifier -> geometry -> temporal command controller`

It uses the supplied dynamic detector's full FP16 palm/landmark models with GPU
preference and XNNPACK fallback, a persistent one-thread LiteRT classifier
isolate, newest-frame-only scheduling at up to 10 FPS, independent left/right
hand state, and the complete twelve-command static policy. The preview draws
both raw detected hands with bounding boxes, rotated ROIs, 21 landmarks and
skeleton connections even when a pose is rejected.

## NCM camera

The app connects directly to `192.168.50.2:5000`, parses JLIP packets, validates
CRC/JPEG framing, replies to heartbeats, reconnects automatically and retains
only the newest pending frame. It no longer refuses to start unless Android
reports exactly `192.168.50.1/30`; Android's route table selects the connected
USB Ethernet/NCM interface.

The phone and board firmware must still expose a usable USB-NCM/Ethernet link.
If Android itself creates no network interface or route, a normal application
cannot assign one without platform/firmware support. Use a USB data cable,
accept any Ethernet/USB network prompt, and ensure the board is listening on
TCP port 5000.

## Build and install

```powershell
flutter pub get
flutter test
flutter build apk --release --target-platform android-arm64
```

Install `build/app/outputs/flutter-apk/app-release.apk` on a 64-bit Android API
24 or newer device. This project is a tester: recognized commands are shown on
screen but are not connected to Boss app actions.

The verified handover build is
`releases/GestureLab-v2.0.1-litert-ncm-arm64-release.apk`. It is a
release-optimized test build signed with the Android debug certificate; replace
that signing configuration with the owner's production key before store or Boss
app distribution.

Version 2.0.1 fixes the reversed Open Palm/Stop Inverted surface check in the
shared runtime and includes the Like/Thumb Down direction correction. A palm
facing the camera resolves to Open Palm; the back of the upright open hand
resolves to Stop Inverted. Both physical hands and mirrored detector inputs
are covered by host regressions; live camera acceptance remains a device check.
