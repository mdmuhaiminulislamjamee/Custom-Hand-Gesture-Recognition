# 05 - Set up the platform, lifecycle and device checks

## What this part does

The runtime manages inference and NCM reconnects. The host app still owns
platform permissions, screen/app lifecycle, its UI state, signing and
supported-device testing.

## What the app team integrates

1. **Android NCM:** keep `android.permission.INTERNET`. Confirm that the device
   has a route to the board before testing gestures. The qualified Android
   build used ARM64 and minimum API 24.
2. **iOS NCM:** add `NSLocalNetworkUsageDescription`, build/sign on a Mac and
   resolve native dependencies for at least iOS 15 or a higher required target.
   Check NCM routing on the actual iPhone. An iOS build/device test is still
   required; the handover is not an IPA.
3. **Lifecycle:** serialize start/stop so two sessions cannot run together.
   Stop on background or when the view is paused, clear saved preview/result,
   start again on resume, and close the engine when the view is disposed.
   Ignore late callbacks from an old session.
4. **Before integration:** run `python verify_handover.py` from the extracted
   `static_gesture_runtime/` folder. Python is only for handover verification,
   not required inside the mobile app.
5. **After integration:** run the host app's analysis/tests, build in release
   mode, then follow [DEVICE_ACCEPTANCE.md](../../DEVICE_ACCEPTANCE.md) on
   target phones. Check every gesture, two hands, hand switching, empty frames,
   preview flicker, reconnect, background/resume and sustained FPS.

Keep the FP32 model, detector settings, image orientation and dependency
versions as supplied during first integration. Actual device speed and
recognition depend on the host camera/transport and native dependency build.
Movement suppression and extra ear/ghost detection tuning are not in this
release.
