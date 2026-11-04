# Gesture Lab — Flutter LiteRT tester

This Android tester now runs the same static runtime delivered in
`static_gesture_runtime/flutter`:

`phone camera or USB-NCM JPEG -> dynamic two-hand detector -> 76 features -> qualified FP32 TFLite classifier -> geometry -> temporal command controller`

It uses the supplied dynamic detector's full FP16 palm/landmark models with GPU
preference and XNNPACK fallback, a persistent one-thread LiteRT classifier
isolate, newest-frame-only scheduling at up to 10 FPS, independent left/right
hand state, and the complete twelve-command static policy. The preview draws
accepted hand candidates with bounding boxes, rotated ROIs, 21 landmarks and
skeleton connections even when a real hand's gesture is rejected. Version 2.0.2
filters compact face/ear, undersized, incomplete and collapsed candidates inside
the detector worker before classification, overlays or continued ROI tracking.
The persistent CPU BlazeFace model refreshes at most every 200 ms; cached face
boxes expire after 500 ms. Both phone and NCM cameras use this shared worker.

## NCM camera

The app connects directly to `192.168.50.2:5000`, parses JLIP packets, validates
CRC/JPEG framing, replies to heartbeats, reconnects automatically and retains
only the newest pending frame. Version 2.0.3 selects Android's Ethernet network
with the best route to the board on every connection attempt. The board TCP
socket is bound through that network's socket factory. A token-protected local
relay delivers the unchanged bytes to the shared Dart parser. Wi-Fi and cellular
keep their existing routing. No exact host address or Internet validation is
required; Android must supply a usable route.

The phone and board firmware must still expose a usable USB-NCM/Ethernet link.
If Android itself creates no network interface or route, a normal application
cannot assign one without platform/firmware support. Use a USB data cable,
accept any Ethernet/USB network prompt, and ensure the board is listening on
TCP port 5000.

## Build and install

On Windows, extract the complete handover into a short path such as
`C:\GestureBuild` and keep the runtime and tester as sibling folders. Native
shader/OpenCV tools can exceed Windows path limits in deeply nested folders.

```powershell
flutter pub get
flutter test
flutter build apk --release --target-platform android-arm64
```

Install `build/app/outputs/flutter-apk/app-release.apk` on a 64-bit Android API
24 or newer device. This project is a tester: recognized commands are shown on
screen but are not connected to Boss app actions.

The current NCM tester build is
`releases/GestureLab-v2.0.7-litert-ncm-arm64-diagnostic-release.apk`. It is a
release-optimized test build signed with the Android debug certificate; replace
that signing configuration with the owner's production key before store or Boss
app distribution.

Version 2.0.7 keeps the last decoded NCM picture visible while the next JPEG
loads, removing the black gap caused by replacing the image widget every frame.
Landmarks are hidden during that decode interval and return with their matching
picture. Disconnects, pause/stop and inference errors still clear stale evidence.
The USB transport retains the v2.0.6 forced interface claims.

Version 2.0.6 enables forced claims of the validated NCM control and data
interfaces, addressing the v2.0.5 claim failure reported on the phone. Android
may detach an existing kernel driver for those interfaces. USB diagnostics
identifies v2.0.6 and records `force=true` before the claims. This needs a
physical camera test; successful interface claims do not guarantee streaming.

Version 2.0.5 adds an experimental direct USB-NCM fallback for the camera whose
descriptors were verified as VID `1a86`, PID `df07`, CDC-NCM control and CDC data
alternate 1. A usable Android Ethernet route is still preferred. Otherwise the
app requests Android USB permission, reads the CDC union/Ethernet/NCM functional
descriptors, negotiates NTB16, selects the data alternate setting, and runs one
bounded ARP/IPv4/TCP client for `192.168.50.1` to `192.168.50.2:5000`. This does
not create a kernel network interface. No root, VPN, DHCP service or process-wide
network binding is used. Vendor interface 0 is not claimed.

The transport handles packet checksums, bounded receive/transmit queues, TCP
retransmission, flow control, sequence wrap, duplicate suppression and connection
reset. Out-of-order TCP data is recovered through cumulative ACK/retransmission.
It is a board-specific TCP transport, with no IPv6, IP fragmentation, arbitrary
network routing or general socket service. It is not qualified as a general TCP
stack. Physical Realme 9/camera acceptance and sustained throughput measurement
are still required. The gesture pipeline remains the shared qualified FP32 path.

Install v2.0.7, connect the powered camera with OTG enabled, select NCM and choose
**Allow** in Android's USB access dialog. After a denial, tap Reconnect to request
permission again. If streaming fails, use **USB diagnostics → Copy report**;
the report now includes recent native initialization/link/TCP errors. Logcat tag
`USB_NCM` provides the same progress. Unplugging/stopping clears the transport,
resets the board TCP connection and releases USB interfaces.

Host verification: `scripts/check_native_usb_ncm.py` compiles against cached real
Android/Flutter APIs, runs wire and TCP state-machine regressions and checks
transmitted packets against an independent Python checksum/layout oracle.
These checks do not replace the first physical-device test.
The v2.0.7 APK rebuilds the Flutter library for the preview fix. Earlier
v2.0.5/v2.0.6 transport updates reused verified v2.0.4 Dart library/assets after
checking identical source, tests and dependency lockfiles. The preview widget
regression verifies picture retention and matching overlay timing across an
uncached asynchronous decode. Physical-device flicker acceptance is still needed.

Version 2.0.4 records USB descriptor diagnostics in Android logcat under
the `USB_DESCRIPTOR` tag. With the camera connected, select NCM (or Reconnect)
to capture them, then run `adb logcat -d -s USB_DESCRIPTOR:I`.
The NCM error panel also has **USB diagnostics → Copy report**, so ADB is
optional. Each inspection reports the current attached-device count, including
an explicit message when no device is enumerated. The records show VID/PID, interface
class/subclass/protocol, endpoint type and direction, which identifies whether
the board exposes CDC-NCM before a native USB transport is attempted.

The `diagnostic-release` APK uses release Dart code with Android
shrinking disabled to permit packaging on this memory-constrained host. It is
signed with the same Android debug key as the tester releases. Descriptor
inspection itself does not open or claim USB interfaces. The v2.0.5 NCM fallback
opens and claims only the validated NCM interfaces after Android USB permission.

Version 2.0.3 reports the selected interface/address and distinguishes missing
Ethernet from missing board routes and TCP errors. Reconnect selects the route
again after USB unplug/replug. Stop, pause, stream close and activity teardown
release the native sockets. Realme 9 hardware acceptance remains required;
the APK cannot create a missing USB-NCM driver or assign the phone's Ethernet IP.

Version 2.0.2 includes false-hand filtering and retains the 2.0.1 correction to
the reversed Open Palm/Stop Inverted surface check in the
shared runtime and includes the Like/Thumb Down direction correction. A palm
facing the camera resolves to Open Palm; the back of the upright open hand
resolves to Stop Inverted. Both physical hands and mirrored detector inputs
are covered by host regressions; live camera acceptance remains a device check.
Compact real hands directly over a face may be rejected. Keep a clear separation;
dim or distant hands can be harder to acquire at the .55 detection threshold.
