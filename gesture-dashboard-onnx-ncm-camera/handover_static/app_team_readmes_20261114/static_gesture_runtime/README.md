# Static Gesture Runtime

**Start with [START_HERE.md](START_HERE.md).**

This is the complete app integration package, without the Flutter tester app.
Keep this folder together and add `flutter/` to the consuming app as a path
dependency. Production model: `flutter/assets/models/gesture_static_float32.tflite`.

- [Integration guide](INTEGRATION_GUIDE.md)
- [Separate component READMEs](component_readmes/README.md)
- [Model locations](MODEL_GUIDE.md)
- [Performance and behavior](PERFORMANCE.md)
- [iOS technical reference](IOS_INTEGRATION.md)
- [Device acceptance](DEVICE_ACCEPTANCE.md)
- [Validation summary](VALIDATION.md)

Run `python verify_handover.py` before integration. Read
`validation/app_team_package_preservation.json` for the exact comparison with
the previously tested release. Earlier model training/conversion reports remain
evidence for the preserved model; they are not new camera measurements.
