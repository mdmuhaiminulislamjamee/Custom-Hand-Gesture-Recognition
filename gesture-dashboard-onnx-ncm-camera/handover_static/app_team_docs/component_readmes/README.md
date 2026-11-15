# App-team component READMEs

These short files explain what each part does and what the app team must add to
its own Flutter app. For an NCM camera, read **01 through 05 in order**. Read 06
only if the app will use the phone's built-in camera instead of NCM.

| Readme | Purpose |
|---|---|
| [01 - Runtime package and models](01_runtime_package/README.md) | Add the complete package and its dependencies. |
| [02 - NCM camera connection](02_ncm_connection/README.md) | Receive camera frames and run detection. |
| [03 - Preview and selected hand](03_preview_and_hand/README.md) | Show the correct image, overlay and controlling hand. |
| [04 - Gesture commands](04_gesture_commands/README.md) | Connect confirmed gestures to the app's existing actions. |
| [05 - Platform setup and lifecycle](05_platform_and_lifecycle/README.md) | Permissions, start/stop, connection checks and device testing. |
| [06 - Phone camera alternative](06_phone_camera_optional/README.md) | Use the phone camera when NCM is not the input. |

The full API details and examples are in [INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md).
The [minimal NCM Flutter example](../flutter/example/lib/main.dart) is runnable
after the host platform files and permissions are set up as its README explains.
The separate Flutter tester app and APK are not required.

Keep the whole `static_gesture_runtime/` folder together. Do not select only a
TFLite model or copy individual runtime source files into the app. The app team
supplies its own screen, product action handlers and supported-device testing.
