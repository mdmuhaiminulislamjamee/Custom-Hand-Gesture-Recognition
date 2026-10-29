import 'dart:io';

import 'package:camera/camera.dart';
import 'package:hand_detection/hand_detection.dart';
import 'package:static_gesture_runtime/detector/camera_frame_adapter.dart';
import 'package:static_gesture_runtime/detector/hand_detector_factory.dart';
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';

class PhoneStaticGestureFrame {
  const PhoneStaticGestureFrame({
    required this.detections,
    required this.result,
    required this.width,
    required this.height,
    required this.detectorMs,
    required this.totalMs,
  });

  final List<Hand> detections;
  final StaticGestureFrameResult result;
  final int width;
  final int height;
  final double detectorMs;
  final double totalMs;
}

/// Phone-camera front end for the same detector, TFLite classifier, geometry,
/// two-hand identity and temporal command controller used by the NCM runtime.
class PhoneStaticGestureEngine {
  PhoneStaticGestureEngine({
    this.backend = MobileInferenceBackend.gpu,
    this.onCommand,
  });

  final MobileInferenceBackend backend;
  final void Function(StaticGestureCommand command)? onCommand;
  final StaticDetectorAdapter _adapter = StaticDetectorAdapter();
  final Stopwatch _clock = Stopwatch()..start();

  MobileHandDetectorSession? _detector;
  StaticGestureRuntimeApi? _runtime;
  Future<void>? _initializing;
  bool _closed = false;
  double _lastTimestampMs = -1;

  String get detectorBackend => _detector?.backend ?? 'initializing';

  Future<void> initialize() {
    if (_closed) throw StateError('Phone gesture engine is closed.');
    return _initializing ??= _initialize();
  }

  Future<void> _initialize() async {
    final runtime = StaticGestureRuntimeApi(
      classifier: StaticTfliteClassifier(model: StaticTfliteModel.float32),
      onCommand: onCommand,
    );
    await runtime.initialize();
    MobileHandDetectorSession? detector;
    try {
      detector = await MobileHandDetectorFactory.create(
        requestedBackend: backend,
        requestedLandmarkModel: MobileLandmarkModelVariant.full,
        detectionConfidence: .50,
        landmarkConfidence: .60,
      );
    } catch (_) {
      await runtime.close();
      rethrow;
    }
    if (_closed) {
      await detector.detector.dispose();
      await runtime.close();
      throw StateError('Phone gesture engine closed during initialization.');
    }
    _runtime = runtime;
    _detector = detector;
  }

  Future<PhoneStaticGestureFrame> analyze(
    CameraImage image, {
    required int rotationDegrees,
    required bool modelInputMirrored,
  }) async {
    await initialize();
    if (_closed) throw StateError('Phone gesture engine is closed.');
    final detector = _detector!;
    final rotation = _rotation(rotationDegrees);
    final direct = directCameraFrame(
      image: image,
      rotation: rotation,
      isAndroid: Platform.isAndroid,
      isIOS: Platform.isIOS,
    );
    final watch = Stopwatch()..start();
    final timed = direct != null
        ? await detector.detector.detectFromCameraFrameTimed(direct)
        : await detector.detector.detectFromCameraImageTimed(
            image,
            rotation: rotation,
            isBgra: Platform.isIOS,
          );
    final detectorMs = watch.elapsedMicroseconds / 1000;
    final quarterTurn = rotationDegrees == 90 || rotationDegrees == 270;
    final width = quarterTurn ? image.height : image.width;
    final height = quarterTurn ? image.width : image.height;
    var timestampMs = _clock.elapsedMicroseconds / 1000;
    if (timestampMs <= _lastTimestampMs) timestampMs = _lastTimestampMs + .001;
    _lastTimestampMs = timestampMs;
    final frame = _adapter.frame(
      timestampMs: timestampMs,
      width: width,
      height: height,
      detections: timed.hands,
      modelInputMirrored: modelInputMirrored,
    );
    final result = await _runtime!.processFrame(frame);
    if (result == null) throw StateError('Phone camera frame was superseded.');
    return PhoneStaticGestureFrame(
      detections: List<Hand>.unmodifiable(timed.hands.take(2)),
      result: result,
      width: width,
      height: height,
      detectorMs: detectorMs,
      totalMs: watch.elapsedMicroseconds / 1000,
    );
  }

  Future<void> reset() async {
    _adapter.reset();
    _lastTimestampMs = -1;
    _runtime?.reset();
    await _detector?.detector.resetTracking();
  }

  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    try {
      await _initializing;
    } catch (_) {
      // Initialization already surfaced the original error.
    }
    await _detector?.detector.dispose();
    _detector = null;
    await _runtime?.close();
    _runtime = null;
  }

  CameraFrameRotation? _rotation(int degrees) => switch (degrees % 360) {
    90 => CameraFrameRotation.cw90,
    180 => CameraFrameRotation.cw180,
    270 => CameraFrameRotation.cw270,
    _ => null,
  };
}
