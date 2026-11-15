# 02 - Connect the NCM camera

## What this part does

`NcmStaticGestureRuntime` connects to the camera's TCP JPEG stream, keeps the
detector and classifier alive, admits the latest frames, reconnects and reports
results. It uses the same detector, model and two-hand rules as the tested
Flutter runtime.

## What the app team integrates

1. Ensure the phone can reach the board at `192.168.50.2:5000` through its
   network route. The host/port can be changed to match the deployed board.
2. Create one `StaticGestureRuntimeApi` with `StaticTfliteModel.float32` and
   pass it to one `NcmStaticGestureRuntime` for the active screen/session.
3. Use `onPreview` to receive each paired JPEG/result. Show it as explained in
   [03 - Preview and selected hand](../03_preview_and_hand/README.md).
4. Use `onCommand` for actions as explained in
   [04 - Gesture commands](../04_gesture_commands/README.md).
5. Call `await engine.start()` when active, `await engine.stop()` when paused,
   and `await engine.close()` when the session is permanently disposed. Clear
   the saved preview/result when connection state is not `connected` or on error.

Copy the setup pattern from the [minimal NCM example](../../flutter/example/lib/main.dart)
and the [Route B instructions](../../INTEGRATION_GUIDE.md). Do not start a second
detector or another camera-input route for the same frames.

## Transport condition

This package handles the NCM TCP stream **after the phone has a network route**.
It does not create an Android USB Ethernet route. The earlier tester's Android
raw USB fallback was part of that tester app and is not in this handover. If
the product needs that fallback, the team must establish the OS route or pass
its own connector through `socketConnector` and `closeTransport`.
