# Android integration and run guide

This bundle targets an existing Kotlin Android app. The image source is the USB-NCM board camera. The supplied Kotlin code brings validated NCM JPEGs through MediaPipe hand landmarks and the production ONNX model and emits raw diagnostic results. It does not yet implement all desktop gesture-decision and frame-recovery rules, so do not connect raw model scores directly to app actions. Use the current Python/Swift reference and `VALIDATION.md` to finish and qualify that port.

## 1. Prepare the project

Use a physical Android device and Android Studio. MediaPipe Tasks requires Android SDK/API 24 or higher; set `minSdk` to at least 24. Start with the existing app, or create a Kotlin app module if none exists. Merge `android/app-build.gradle.kts.fragment` into the app module's Gradle file. Its key entries are:

~~~kotlin
dependencies {
    implementation("com.google.mediapipe:tasks-vision:latest.release")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:latest.release")
}
~~~

The version expressions above follow the vendors' installation guides for initial resolution. Once the app team has built and tested on its target device, replace them with the exact resolved versions in the app's version catalog/lockfile. This bundle does not assert an Android dependency version that was never resolved here. See the official [MediaPipe Android setup](https://developers.google.com/edge/mediapipe/solutions/setup_android), [Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/android) and [ONNX Runtime installation guide](https://onnxruntime.ai/docs/install/).

## 2. Copy sources and assets

Copy all `.kt` files in `android/kotlin/` to the app's `src/main/java/com/example/gesturehandover/` directory, or move them into the app's package and update their `package` declarations together. Merge `android/AndroidManifest.fragment.xml` into the app manifest. This uses TCP networking and network-state access; it does not use the built-in Android camera or request `CAMERA` permission.

Copy all five files from `android/models/` into `app/src/main/assets/` without renaming:

- `gesture_mlp_production.onnx`
- `gesture_mlp_onnx_metadata.json`
- `gesture_mobile_runtime_config.json`
- `hand_landmarker.task`
- `blaze_face_short_range.tflite`

The face model is supplied for the full frame-policy port; the starter analyzer does not yet invoke it. For debug parity, also copy `android/tests/parity_vectors.json` to `src/debug/assets/` (or `src/main/assets/` while testing). Run on a background thread:

~~~kotlin
val report = GestureParity.run(applicationContext)
check(report.featureCases == 294 && report.onnxCases == 26)
println("Android parity maximum error: ${report.maximumError}")
~~~

This checks 76-value feature extraction and named ONNX outputs; it does not check the unported geometry/action pipeline. The ONNX input is float32 `landmark_features` `[1,76]`; outputs are `probabilities` `[1,12]` and `known_gesture_mass` `[1,1]`. Do not add a second scaler or softmax.

## 3. Establish the NCM route

The board listens on `192.168.50.2:5000`. Connect the board with the product's intended USB host cable. In Android network diagnostics, verify an OS-provided Ethernet/NCM interface with host address `192.168.50.1/30` or an equivalent route to the board. Some Android phones do not expose USB-NCM as a usable app network; the supplied TCP client cannot create that OS support.

When the interface is visible, select its Android `Network` and pass it to the client. `Network.getSocketFactory()` binds only this connection to the NCM route, so Wi-Fi/cellular may remain available for the rest of the app. If `NcmNetworkSelector.find()` returns null, diagnose the device/USB route before starting recognition.

~~~kotlin
val connectivity = getSystemService(ConnectivityManager::class.java)
val boardNetwork = requireNotNull(NcmNetworkSelector.find(connectivity)) {
    "USB-NCM board network is unavailable"
}
val analyzer = GestureFrameAnalyzer(
    applicationContext,
    onObservation = { observation ->
        // Diagnostic only: display scores and record frame/latency data.
        // Do not dispatch an app action from raw probabilities.
    },
    onMissingHand = { /* clear any app-side preview/hold state */ },
    onError = { error -> android.util.Log.e("Gesture", "Frame failed", error) },
)
val ncm = GestureNcmClient(
    network = boardNetwork,
    onState = { state ->
        if (state != NcmState.CONNECTED) analyzer.clearPending()
        android.util.Log.i("Gesture", "NCM $state")
    },
    onJpeg = analyzer::submit,
    onError = { error -> android.util.Log.e("Gesture", "NCM failed", error) },
)
ncm.start()
// Stop the NCM client and close the analyzer from a background thread when
// leaving the feature; the analyzer waits for its worker to exit.
~~~

Import `android.net.ConnectivityManager` in the app source containing this example. NCM callbacks are on a worker thread, so send UI changes to the main thread. `GestureFrameAnalyzer` keeps only the newest pending image, uses a 0.92 central square crop, and starts analysis no faster than once per 100 ms. It does not rotate or mirror incoming pixels. The board's orientation must be confirmed and, if needed, corrected in image preparation before MediaPipe.

## 4. Finish the decision path before actions

The Windows dashboard is the behavioral reference. Port the full frame preparation, low-light, two-pass hand recovery, face overlap veto, hand-size checks, landmark stabilization, gesture geometry resolver, world-landmark Fist guard, temporal hold, rejection and action latch. Preserve handedness and 21 world landmarks from the same selected hand/frame; image `z` is not a substitute for world coordinates. The current `GestureFrameAnalyzer` exposes only image landmarks and raw model scores, so it is intentionally insufficient to trigger actions.

Use `android/reference/backend/` and `android/reference/swift/` as read-only source for the current rules. The geometry fixtures in `android/tests/parity_vectors.json` contain expected geometry and temporal decisions for 294 cases. Port tests that compare those expectations on Android. Only emit an app action after the full resolver, temporal gate and latch accept a gesture; `no_gesture` always means no action.

The production analysis convention is upright, unmirrored NCM pixels: normalized x increases to the right and y increases downward. The trained horizontal command calibration maps positive index x to `left` and negative index x to `right`. Do not swap those commands a second time in the app UI. Use `SystemClock.elapsedRealtime()` for strictly increasing detector and decision timestamps. Clear pending frames and recognition state when the NCM route drops, when crop/orientation changes, or after app suspension.

## 5. Device acceptance

Run [VALIDATION.md](VALIDATION.md) on the actual Android phone/tablet and board. A Windows NCM run proves the model/board reference; it does not prove that a particular Android device enumerates the board or supports the final latency and gesture behavior. This delivery is an integration handover, not a signed APK or complete Android Studio project.
