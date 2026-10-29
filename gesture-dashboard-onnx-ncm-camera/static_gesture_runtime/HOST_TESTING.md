# Host checks and what they exercise

The model and geometry checks work without OpenCV or a camera:

```powershell
python tools/verify_static_runtime.py --package-root .
cd flutter
dart tool/check_geometry_parity.dart ../validation/python_reference.json ../validation/dart_geometry_parity.json
dart tool/check_geometry_parity.dart ../validation/cache_geometry_reference.json ../validation/dart_cache_geometry_parity.json
```

Use the direct `dart <file.dart>` command for the geometry script. `dart run`
may invoke native build hooks for every dependency, even though the geometry
script only uses the Dart standard library and relative source imports.

Flutter unit/integration tests use a real native LiteRT interpreter for the
model fixtures, and injected detector results for lifecycle/two-hand tests.
The NCM tests create a local TCP server and exercise the actual JLIP reader.
They do not run the MediaPipe detector on camera pixels.

On Windows the transitive `dartcv4` package normally invokes a native OpenCV
build that requires CMake and a suitable C/C++ compiler. For unit tests that
never call OpenCV, its documented `skip_build` hook can be enabled temporarily
in `flutter/pubspec.yaml`:

```yaml
hooks:
  user_defines:
    dartcv4:
      skip_build: true
```

Then run `flutter test --no-pub --concurrency=1`. Remove this hook setting before
building or running a real detector application. The delivered pubspec enables
normal native builds. LiteRT still loads its real native library during these
tests; only the unused OpenCV build hook is bypassed.

The source package's `flutter analyze` and the example's `flutter analyze` check
Dart compatibility. They do not compile iOS native plugins. Build and run the
example on the supported physical iPhones to qualify the detector, delegates,
local-network route and camera-to-command performance.
