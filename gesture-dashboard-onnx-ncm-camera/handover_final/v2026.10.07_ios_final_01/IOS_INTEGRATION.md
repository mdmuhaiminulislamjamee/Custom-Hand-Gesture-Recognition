# iOS integration and run guide

This guide targets the app team's existing Swift iOS application. The five core Swift files implement the post-landmark classifier decisions. The additional adapters wire ONNX and MediaPipe. A complete iOS camera port also needs the frame preparation, recovery and face policies listed below; the supplied MediaPipe starter does not implement those stages.

## 1. Prepare the Mac project

Use Xcode on a Mac and a physical iPhone/iPad. Create an iOS Swift app if you do not already have one. Choose the real app's signing team and deployment target. The Podfile template starts at iOS 15; raise the target if the resolved dependencies require it.

Copy `ios/swift/ios/Podfile.example` beside the Xcode project as `Podfile`, replace `YourApp` with the actual target name, and run:

~~~sh
pod install
open YourApp.xcworkspace
~~~

Both MediaPipeTasksVision and onnxruntime-objc are CocoaPods dependencies. Use the workspace created by CocoaPods. Resolve and record exact versions in Podfile.lock after testing; this Windows-prepared delivery does not contain a fabricated Apple-platform lockfile. See the official [MediaPipe iOS guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/ios) and [ONNX Objective-C/Swift setup](https://onnxruntime.ai/docs/get-started/with-obj-c.html).

## 2. Add sources, resources and permissions

Add all five `ios/swift/*.swift` core files and all `ios/swift/ios/*.swift` adapters to the app target. Do not add `tests/main.swift` to an iOS target: it is a Mac command-line test entry point.

Copy these five files from the SAME `ios/models` directory into the app's Copy Bundle Resources:

- gesture_mlp_production.onnx
- gesture_mlp_onnx_metadata.json
- gesture_mobile_runtime_config.json
- hand_landmarker.task
- blaze_face_short_range.tflite

Use an Xcode group that copies these resources into one bundle directory, or supply that actual directory URL to the adapters. Verify target membership and resource URLs at runtime. Do not put models into an asset catalog.

Add `Gesture-Bridging-Header.h` to the project and set **Build Settings → Objective-C Bridging Header** to its project-relative path, for example `YourApp/Gesture-Bridging-Header.h`. If your app already has a bridging header, add its ONNX import to that existing file.

Merge the permission keys from `Info.plist.fragment` into the app's existing Info settings. Camera capture needs NSCameraUsageDescription; board networking needs NSLocalNetworkUsageDescription. Request camera permission before creating a capture session. Broadcast discovery may need additional iOS networking entitlement work; see NCM_PROTOCOL.md.

For integration diagnostics, also add `ios/swift/tests/parity_vectors.json` to Copy Bundle Resources. It may be removed from the shipping app after qualification.

## 3. Run native core regression checks before camera work

From the extracted outer bundle on the Mac:

~~~sh
sh ios/swift/tests/verify_swift.sh
~~~

The script compiles the core with xcrun swiftc and tests 294 landmark/geometry cases, feature values, temporal decisions at three known-mass levels, and tracking-gap action rearming. Expected: PASS and maximum numerical differences at or below 0.0001. A nonzero exit means stop and inspect the named case. It requires no CocoaPods because it tests the core only.

This command has NOT been executed on this Windows host. Do not treat the included Python expectations as proof that Swift already passes.

## 4. Test ONNX on the phone before enabling actions

Initialize one classifier, keep it alive, and call the parity checker:

~~~swift
guard let modelURL = Bundle.main.url(forResource: "gesture_mlp_production", withExtension: "onnx"),
      let vectorsURL = Bundle.main.url(forResource: "parity_vectors", withExtension: "json") else {
    fatalError("Gesture resources missing from app target")
}
let modelsDirectory = modelURL.deletingLastPathComponent()
let classifier = try GestureONNXClassifier(modelsDirectory: modelsDirectory)
let maximumError = try GestureONNXParity.run(classifier: classifier, vectorsURL: vectorsURL)
print("ONNX maximum error:", maximumError)
~~~

Expected: all 26 cases pass with maximum absolute error ≤ 0.0001. Run this on a background queue. The wrapper verifies model hash, taxonomy, key temporal settings, tensor names, float32 output shapes and finite probabilities. It uses CPU with one intra-op thread. Test alternate execution providers separately before enabling them.

The graph already includes its trained transforms. Supply exactly the 76 features returned by GestureFeatureExtractor; do not apply another scaler or softmax. Input: `landmark_features` float32 [1,76]. Outputs: `probabilities` float32 [1,12] and `known_gesture_mass` float32 [1,1]. Ignore the auxiliary label output for application decisions.

## 5. Connect your camera frames to MediaPipe and ONNX

Keep one detector and one GestureLandmarkEngine per stream. The first-light wiring is:

~~~swift
let engine = try GestureLandmarkEngine(modelsDirectory: modelsDirectory)
let detector = try GestureHandLandmarker(
    modelURL: modelsDirectory.appendingPathComponent("hand_landmarker.task")
)

// Call from the app's serial background inference queue.
// 'image' must describe this frame with its correct orientation.
func classify(image: MPImage, timestampMS: Int) throws -> GesturePipelineResult {
    let result = try detector.detect(image: image, timestampMS: timestampMS)
    let points = result.landmarks.first?.map {
        SIMD2<Float>(Float($0.x), Float($0.y))
    }
    let world = result.worldLandmarks.first?.map {
        SIMD3<Float>(Float($0.x), Float($0.y), Float($0.z))
    }
    let hand = result.handedness.first?.first
    return try engine.process(
        landmarks: points, worldLandmarks: world,
        handedness: hand?.categoryName,
        handednessConfidence: hand.map { Float($0.score) },
        timestamp: Double(timestampMS) / 1000.0
    )
}
~~~

Import MediaPipeTasksVision in the source containing this sample. This demonstrates data wiring; for parity, replace raw `points` with the accepted, stabilized points from the full pipeline below, and pass nil on a rejected/missing hand. Do not silently skip rejected frames: the temporal gate needs them to clear old holds.

Use AVCaptureVideoDataOutput in the app's existing camera session and create MPImage from correctly oriented frames. Google's linked sample provides a camera UI. Keep analysis pixels unmirrored and normalized image coordinates with origin at top left, y increasing downward. A mirrored preview is a UI choice; its transform must not leak into inference. Match the world landmarks and handedness to the same selected hand/frame. Never replace world landmarks with normalized image z.

Use a monotonic clock, e.g. ProcessInfo.processInfo.systemUptime. MediaPipe receives strictly increasing integer milliseconds; the decision engine receives seconds from that SAME clock. Do not use board wall-clock timestamps directly for temporal holds.

Admit at most one frame per 100 ms, only when the serial pipeline is idle. Retain at most the newest pending frame, or drop busy frames. Do not replay one image as multiple independent camera observations. Dispatch display updates and non-nil actions to the main queue. Reset the engine and recreate the VIDEO detector when changing camera, rotating the analysis coordinate system, reconnecting, or returning from a suspended session.

Only `decision.action` drives an action:

~~~swift
if let action = decision.action {
    DispatchQueue.main.async {
        // Route this string to your existing app action handler.
        // Move Left, Play / Pause, etc. are the contract values in README.md.
    }
}
~~~

Replace the comment with your app handler. Recognition does not implement recording, media control, object tracking, movement or hardware commands for your product.

## 6. Complete the frame pipeline for desktop-equivalent behavior

These stages are essential to reproducing the observed BAT behavior. Copying the model and first-light snippet alone does not reproduce them. Current Python source is bundled under `ios/reference/backend`; its tests and runnable app are in desktop_reference.zip.

| Stage | Authoritative implementation | Required behavior |
|---|---|---|
| Image policy | model_runtime.py: InferenceEngine.process_frame | Full frame for standalone BAT; optional 0.92 central square for dashboard. Downscale only above 960 pixels; no analysis mirroring. |
| Low light | model_runtime.py: adaptive_low_light_preprocess | Near-blank rejection; bounded CLAHE/gamma enhancement. The bundled model alone does not perform this. |
| Detection/recovery | hand_detection.py: HandDetector | VIDEO primary 0.55 detection/presence, 0.60 tracking; IMAGE recovery 0.20/0.30; view/crop coordinate restoration; candidate selection and two-frame recovery confirmation. |
| Face rejection | face_guard.py: FaceGuard and face_context_rejection | BlazeFace inference, cached boxes, small false-hand and face/ear-contact rejection. Include the asset and port the actual overlap policy. |
| Landmark acceptance | model_runtime.py: RuntimeSession.stabilize_landmarks | Pixel-size/clipping/plausibility checks, adaptive EMA, jump rejection and persistent-motion reacquisition. |
| Features/model | supplied Swift core + ONNX wrapper | Exact 76-D image-x/y features and named model outputs. World landmarks are separate geometry evidence. |
| Geometry | supplied GestureGeometryResolver | Latest Down precedence, Like reach guard/recovery, index-only OK pinch, Peace pair/world checks, Dorsal angle/surface checks, Fist depth/view rules. |
| Temporal/action | supplied gate and latch | Holds, probability EMA, rejection reset, cooldown and release. Missing-hand tracking gaps use six frames at the 10-FPS target. |
| Product behavior | app team; follow_object.py for reference | Route actions once. The optional dashboard Follow Object sequence is a separate explicit user-started state machine. It is inactive in the standalone viewer. |

The JSON describes common settings; geometry-specific exceptions are defined by the source and fixtures. The normal known-mass floor is 0.70. Supported recovery generally uses 0.15; strict Down and Dorsal-range recovery may accept lower mass, and Open Palm has its 0.12/sideways exceptions. These are narrowly gated source rules, not permission to lower the global threshold.

Normal confidence floor is 0.70, margin 0.08, probability EMA alpha 0.45, three stable frames, minimum hold 0.18 seconds, cooldown 0.80 seconds, and three release frames for a real pose change. Low-mass geometry recovery has longer holds: Dorsal 0.35, directions 0.40, supported hand shapes 0.45 seconds.

Live Fist requires world depth. The current rules allow a 20-degree transition below horizontal with strict depth/view exceptions and explicit negative-view exclusions. Dorsal accepts its validated surface/finger shape only in signed XY range 0–150 degrees. These rules supersede older handover prose describing all Fist/Open Palm orientations identically.

The new Swift sources use this release's constants. If you change runtime JSON/geometry policies later, port and requalify them together. Do not assume every arbitrary JSON setting is dynamically implemented by the native code.

## 7. Run and qualify on the physical device

Select the real device and app scheme, build from the CocoaPods workspace, and grant camera/local-network access as appropriate. First log decisions without invoking product actions. Pass core and ONNX tests, then compare the same camera poses and negatives with the bundled Windows reference. Finish the checklist in VALIDATION.md, record Podfile.lock, Xcode version, device/iOS version and measured latency, then enable product actions.

Xcode signing, the camera screen, lifecycle/background handling, full preprocessing/recovery/face guard, NCM networking, and product action handlers remain app-team integration work. This package supplies assets, updated decision code, references and tests; it does not claim measured end-to-end iOS parity.
