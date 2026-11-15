# iOS USB-NCM example

This is an integration example for a Flutter iOS host. It prints recognized
commands without connecting them to motors or product actions. Its model assets
are loaded from the `static_gesture_runtime` package.

On a Mac with Flutter and Xcode, create the standard generated host files:
Use Flutter 3.47.8 / Dart 3.13.7 to match the qualified lockfiles.

```sh
cd example
flutter create --platforms=ios --project-name static_gesture_ncm_example .
flutter pub get
```

The pubspec retains normal OpenCV native compilation, limited to the
`imgcodecs` and `imgproc` modules used by the detector. Do not enable the
host-test `skip_build` setting for an iPhone build.

Add the following keys inside the dictionary in `ios/Runner/Info.plist`:

```xml
<key>NSLocalNetworkUsageDescription</key>
<string>Connect to the USB-NCM gesture camera.</string>
```

Set the iOS deployment target according to the resolved LiteRT/OpenCV native
dependencies (at least iOS 15). Resolve pods on the Mac and retain `Podfile.lock`.
Connect the USB-NCM board and verify that `192.168.50.2:5000` is reachable. Run:

```sh
flutter run --release -d <physical-iphone-id>
# Select INT8 only after its manifest is qualified:
flutter run --release -d <physical-iphone-id> --dart-define=STATIC_INT8=true
```

The app uses the board's upright, unmirrored JPEGs. It does not require access to
the iPhone camera. If board firmware rotates/mirrors its image, adapt that stream
before the common detector and reset the runtime after the change.

The preview is decoded separately by Flutter for display. The inference path
decodes the JPEG once in the detector isolate. For performance measurements,
remove/disable `StaticNcmPreview` to exclude preview rendering overhead.

The default preview draws the supplied dynamic detector's axis-aligned and
rotated ROI boxes, 21 landmark dots and finger skeleton connections for up to
two hands. It displays raw detections even when the static gesture is rejected.
Each overlay uses the same JPEG that produced its landmarks, fits the source
aspect ratio and stays unmirrored. Empty detections clear the overlay, and the
example clears its preview on disconnect or detector errors.

Displayed receive-to-result latency begins when the JPEG reaches the phone.
The board capture clock is retained separately and is not synchronized with the
phone, so the telemetry does not include unknown capture/USB transit time.

The source has host-side analysis/tests. iOS compilation, delegate selection,
USB routing, local-network permission, sustained FPS, heat/power and physical
gesture acceptance must be checked on the supported iPhones. The dynamic
archive does not include a native dependency lock for the Boss application's
exact versions.
