# Final app-team handover

Give your boss and the app team this archive:

[static_gesture_runtime_20261114_app_team_readmes.zip](static_gesture_runtime_20261114_app_team_readmes.zip)

It extracts to one `static_gesture_runtime/` folder. It contains the complete
runtime, model assets, detector, optional phone-camera helpers, NCM example,
clear integration guides, separate component READMEs, validation and verification. The Flutter tester app
and APK are excluded.

Start with [START_HERE.md](app_team_readmes_20261114/static_gesture_runtime/START_HERE.md)
in the already extracted copy. The app team then follows the
[separate component READMEs](app_team_readmes_20261114/static_gesture_runtime/component_readmes/README.md)
and `INTEGRATION_GUIDE.md`.
The package to add to the app is `flutter/`; the production model is
`flutter/assets/models/gesture_static_float32.tflite`.

All 84 files under `flutter/` and both reusable phone helpers are byte-for-byte
the qualified ownership/preview release. Models, features, geometry, dependencies,
detector settings, scheduling, ownership and timing have not been modified for
this repackaging. `validation/app_team_package_preservation.json` records the
comparison. The extracted native model fixtures and all payload checks pass.

Verify the adjacent `.sha256` before transfer and run `python verify_handover.py`
from the extracted folder. Physical iOS compilation, signing and device
performance/recognition acceptance remain app-team tasks. Movement suppression
and further ear/ghost detection tuning remain deferred.

Previous Boss/tester deliveries and their archives remain unchanged for history.
