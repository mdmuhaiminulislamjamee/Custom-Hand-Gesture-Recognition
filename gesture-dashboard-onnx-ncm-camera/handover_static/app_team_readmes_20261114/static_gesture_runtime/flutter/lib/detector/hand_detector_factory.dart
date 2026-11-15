import 'package:flutter/foundation.dart';
import 'package:hand_detection/hand_detection_native.dart';

enum MobileInferenceBackend {
  xnnpack('XNNPACK CPU'),
  gpu('LiteRT GPU · FP16'),
  nnapi('NNAPI · experimental'),
  npu('LiteRT NPU · experimental'),
  coreml('Core ML / Neural Engine');

  const MobileInferenceBackend(this.label);

  final String label;

  bool get isAvailableOnCurrentPlatform {
    if (kIsWeb) return false;
    return switch (defaultTargetPlatform) {
      TargetPlatform.android => this != MobileInferenceBackend.coreml,
      TargetPlatform.iOS =>
        this == MobileInferenceBackend.xnnpack ||
            this == MobileInferenceBackend.gpu ||
            this == MobileInferenceBackend.coreml,
      _ => this == MobileInferenceBackend.xnnpack,
    };
  }

  static MobileInferenceBackend fromEnvironment() {
    const value = String.fromEnvironment(
      'MOBILE_INFERENCE_BACKEND',
      defaultValue: 'gpu',
    );
    return switch (value) {
      'compiled_gpu' || 'gpu' => MobileInferenceBackend.gpu,
      'nnapi' => MobileInferenceBackend.nnapi,
      'npu' || 'compiled_npu' => MobileInferenceBackend.npu,
      'coreml' => MobileInferenceBackend.coreml,
      _ => MobileInferenceBackend.xnnpack,
    };
  }
}

enum MobileLandmarkModelVariant {
  lite('MediaPipe Lite · FP16'),
  full('MediaPipe Full · FP16'),
  w8a8('Qualcomm W8A8 · INT8 · CPU only');

  const MobileLandmarkModelVariant(this.label);

  final String label;

  bool get isQuantized => this == MobileLandmarkModelVariant.w8a8;

  HandLandmarkModel get standardModel => this == MobileLandmarkModelVariant.full
      ? HandLandmarkModel.full
      : HandLandmarkModel.lite;

  String get standardAssetName => 'hand_landmark_${standardModel.name}.tflite';

  static MobileLandmarkModelVariant defaultForPlatform() =>
      MobileLandmarkModelVariant.full;
}

/// Creates only the palm and 21-landmark pipeline.
///
/// `enableGestures` is deliberately false: this project streams landmarks and
/// does not execute the supplied application's gesture classifiers.
abstract final class MobileHandDetectorFactory {
  static const String _int8PalmAsset =
      'packages/hand_detection/assets/models/palm_detector_w8a8.tflite';
  static const String _int8LandmarkAsset =
      'packages/hand_detection/assets/models/hand_landmark_detector_w8a8.tflite';

  static Future<MobileHandDetectorSession> create({
    required MobileInferenceBackend requestedBackend,
    required MobileLandmarkModelVariant requestedLandmarkModel,
    required double detectionConfidence,
    required double landmarkConfidence,
  }) async {
    final useInt8CpuModels =
        requestedBackend == MobileInferenceBackend.xnnpack &&
        requestedLandmarkModel.isQuantized &&
        !kIsWeb &&
        defaultTargetPlatform == TargetPlatform.android;
    final standardLandmarkModel = requestedLandmarkModel.standardModel;
    final standardLandmarkAsset = requestedLandmarkModel.standardAssetName;
    if (!requestedBackend.isAvailableOnCurrentPlatform ||
        requestedBackend == MobileInferenceBackend.xnnpack ||
        requestedLandmarkModel.isQuantized) {
      return MobileHandDetectorSession(
        detector: await _create(
          performanceConfig: const PerformanceConfig.xnnpack(),
          landmarkModel: standardLandmarkModel,
          detectionConfidence: detectionConfidence,
          landmarkConfidence: landmarkConfidence,
          useInt8Assets: useInt8CpuModels,
        ),
        backend: useInt8CpuModels
            ? 'xnnpack_int8_w8a8'
            : 'xnnpack_fp16_${standardLandmarkModel.name}',
        requestedBackend: requestedBackend,
        palmModelName: useInt8CpuModels
            ? 'palm_detector_w8a8.tflite'
            : 'hand_detection.tflite',
        landmarkModelName: useInt8CpuModels
            ? 'hand_landmark_detector_w8a8.tflite'
            : standardLandmarkAsset,
        modelWeightStorage: useInt8CpuModels ? 'int8' : 'float16',
        modelIoType: useInt8CpuModels ? 'uint8' : 'float32',
        delegateStatus: requestedBackend == MobileInferenceBackend.xnnpack
            ? 'active'
            : 'xnnpack_fallback_incompatible_selection',
      );
    }

    try {
      if (requestedBackend == MobileInferenceBackend.nnapi) {
        return MobileHandDetectorSession(
          detector: await _create(
            performanceConfig: PerformanceConfig.disabled,
            landmarkModel: standardLandmarkModel,
            detectionConfidence: detectionConfidence,
            landmarkConfidence: landmarkConfidence,
            useNnApi: true,
          ),
          backend: 'nnapi_requested_fp16_${standardLandmarkModel.name}',
          requestedBackend: requestedBackend,
          palmModelName: 'hand_detection.tflite',
          landmarkModelName: standardLandmarkAsset,
          modelWeightStorage: 'float16',
          modelIoType: 'float32',
          delegateStatus: 'initialized_unverified_nnapi_placement',
        );
      }

      if (requestedBackend == MobileInferenceBackend.coreml) {
        return MobileHandDetectorSession(
          detector: await _create(
            performanceConfig: const PerformanceConfig.coreml(),
            landmarkModel: standardLandmarkModel,
            detectionConfidence: detectionConfidence,
            landmarkConfidence: landmarkConfidence,
          ),
          backend: 'coreml_requested_fp16_${standardLandmarkModel.name}',
          requestedBackend: requestedBackend,
          palmModelName: 'hand_detection.tflite',
          landmarkModelName: standardLandmarkAsset,
          modelWeightStorage: 'float16',
          modelIoType: 'float32',
          delegateStatus: 'initialized_unverified_coreml_placement',
        );
      }

      final accelerator = requestedBackend == MobileInferenceBackend.npu
          ? Accelerator.npu
          : Accelerator.gpu;
      return MobileHandDetectorSession(
        detector: await _create(
          performanceConfig: const PerformanceConfig.xnnpack(),
          landmarkModel: standardLandmarkModel,
          detectionConfidence: detectionConfidence,
          landmarkConfidence: landmarkConfidence,
          useCompiledModel: true,
          accelerators: <Accelerator>{accelerator},
        ),
        backend:
            'litert_${requestedBackend.name}_fp16_${standardLandmarkModel.name}',
        requestedBackend: requestedBackend,
        palmModelName: 'hand_detection.tflite',
        landmarkModelName: standardLandmarkAsset,
        modelWeightStorage: 'float16',
        modelIoType: 'float32',
        delegateStatus: 'initialized_strict_${requestedBackend.name}_request',
      );
    } catch (error) {
      debugPrint(
        '${requestedBackend.label} failed; using XNNPACK fallback: $error',
      );
      return MobileHandDetectorSession(
        detector: await _create(
          performanceConfig: const PerformanceConfig.xnnpack(),
          landmarkModel: standardLandmarkModel,
          detectionConfidence: detectionConfidence,
          landmarkConfidence: landmarkConfidence,
        ),
        backend:
            'xnnpack_fp16_${standardLandmarkModel.name}_after_'
            '${requestedBackend.name}_initialization_failure',
        requestedBackend: requestedBackend,
        palmModelName: 'hand_detection.tflite',
        landmarkModelName: standardLandmarkAsset,
        modelWeightStorage: 'float16',
        modelIoType: 'float32',
        delegateStatus: 'fallback_to_xnnpack',
        fallbackReason: error.toString(),
      );
    }
  }

  static Future<HandDetector> _create({
    required PerformanceConfig performanceConfig,
    required HandLandmarkModel landmarkModel,
    required double detectionConfidence,
    required double landmarkConfidence,
    bool useCompiledModel = false,
    bool useNnApi = false,
    Set<Accelerator> accelerators = const <Accelerator>{
      Accelerator.gpu,
      Accelerator.cpu,
    },
    bool useInt8Assets = false,
  }) {
    return HandDetector.create(
      mode: HandMode.boxesAndLandmarks,
      landmarkModel: landmarkModel,
      detectorConf: detectionConfidence,
      maxDetections: 2,
      minLandmarkScore: landmarkConfidence,
      performanceConfig: performanceConfig,
      useCompiledModel: useCompiledModel,
      useNnApi: useNnApi,
      accelerators: accelerators,
      precision: Precision.fp16,
      palmModelAsset: useInt8Assets ? _int8PalmAsset : null,
      handLandmarkModelAsset: useInt8Assets ? _int8LandmarkAsset : null,
      enableTracking: true,
      trackingConfig: const TrackingConfig(),
      enableGestures: false,
    );
  }
}

/// The detector together with the backend that actually initialized.
class MobileHandDetectorSession {
  const MobileHandDetectorSession({
    required this.detector,
    required this.backend,
    required this.requestedBackend,
    required this.palmModelName,
    required this.landmarkModelName,
    required this.modelWeightStorage,
    required this.modelIoType,
    required this.delegateStatus,
    this.fallbackReason,
  });

  final HandDetector detector;
  final String backend;
  final MobileInferenceBackend requestedBackend;
  final String palmModelName;
  final String landmarkModelName;
  final String modelWeightStorage;
  final String modelIoType;
  final String delegateStatus;
  final String? fallbackReason;
}
