import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show setEquals;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:meta/meta.dart';
import 'package:flutter_litert/native.dart';
import '../shared/hand_geometry.dart';
import '../util/image_utils.dart';
import '../native/nnapi_interpreter_options.dart';

// The [PalmDetection] result type and the pure-Dart box decoding /
// post-processing live in `shared/hand_geometry.dart` so the web implementation
// can reuse them. Re-exported here so existing `palm_detector.dart` importers
// keep seeing [PalmDetection].
export '../shared/hand_geometry.dart' show PalmDetection;

/// SSD-based palm detector for Stage 1 of the hand detection pipeline.
///
/// Detects palm locations in images using a Single Shot Detector (SSD) architecture
/// with anchor-based decoding. Returns rotation rectangles suitable for cropping
/// hand regions for landmark extraction.
///
/// This is a direct port of the Python PalmDetection class.
class PalmDetector {
  final InterpreterPool _pool = InterpreterPool(poolSize: 1);
  bool _isInitialized = false;

  /// Input dimensions (192x192 for palm detection model).
  late int _inH;
  late int _inW;

  /// Pre-generated SSD anchors.
  late List<List<double>> _anchors;

  /// Preprocessing state - matches Python's calculation.
  /// These use original image dimensions like Python does.
  int _imageHeight = 0;
  int _imageWidth = 0;

  /// square_standard_size = max(image_height, image_width)
  int _squareStandardSize = 0;

  /// square_padding_half_size = abs(image_height - image_width) // 2
  int _squarePaddingHalfSize = 0;

  /// Score threshold for detection filtering.
  final double scoreThreshold;

  /// IoU threshold for the weighted non-maximum suppression that fuses
  /// overlapping palm boxes. Higher keeps more nearby detections.
  final double nmsIouThreshold;

  /// Expansion factor applied to each palm box when building its rotated
  /// square ROI for the landmark model (MediaPipe's detection-to-ROI scale).
  final double roiScale;

  /// Pre-allocated input buffer.
  Float32List? _inputBuffer;

  /// Pre-allocated box regressor output, flat Float32 view of [1, 2016, 18].
  /// Backed by a ByteBuffer that's passed directly to TFLite to avoid the
  /// boxed-double round-trip in flutter_litert's Tensor.copyTo for List dst.
  Float32List? _boxesData;

  /// Pre-allocated classification score output, flat Float32 view of [1, 2016, 1].
  Float32List? _scoresData;

  /// Cached `Float32List` views of the input/output tensor native memory,
  /// captured once after `allocateTensors` and reused every inference on
  /// the direct-invoke path.
  TensorFloat32Views? _views;

  /// Qualcomm AI Hub W8A8 models use uint8 input and output tensors. Keep
  /// their raw buffers quantized through inference and dequantize only the
  /// palm candidates that survive the score threshold.
  bool _usesUint8Io = false;
  Uint8List? _uint8InputBuffer;
  Uint8List? _uint8BoxesData;
  Uint8List? _uint8ScoresData;
  Tensor? _uint8InputTensor;
  Tensor? _uint8BoxesTensor;
  Tensor? _uint8ScoresTensor;
  double _uint8BoxesScale = 1.0;
  int _uint8BoxesZeroPoint = 0;
  double _uint8ScoresScale = 1.0;
  int _uint8ScoresZeroPoint = 0;

  /// Number of values per anchor in the box regressor output (18: cx, cy, w, h
  /// followed by 7 keypoint x/y pairs). Cached after init.
  int _boxStride = 18;

  /// LiteRT Next [CompiledModel] engine (GPU with CPU fallback), used instead
  /// of the [InterpreterPool] when initialized via
  /// [initializeCompiledFromBuffer]. Non-null selects the compiled path.
  CompiledModel? _compiled;

  /// CompiledModel output indices, resolved from output byte sizes at init
  /// (the boxes output is `stride` floats per anchor; scores is one per anchor).
  int _cmBoxesIdx = 0;
  int _cmScoresIdx = 1;

  /// Creates a palm detector with the specified score threshold, NMS IoU
  /// threshold, and ROI expansion scale.
  PalmDetector({
    this.scoreThreshold = 0.45,
    this.nmsIouThreshold = 0.45,
    this.roiScale = 2.6,
  });

  /// Initializes the palm detector by loading the TFLite model.
  Future<void> initialize({PerformanceConfig? performanceConfig}) async {
    const String assetPath =
        'packages/hand_detection/assets/models/hand_detection.tflite';
    await _initWith(
      (options) async => Interpreter.fromAsset(assetPath, options: options),
      performanceConfig,
    );
  }

  /// Initializes the palm detector from pre-loaded model bytes.
  ///
  /// Used by [HandDetectorIsolate] to initialize within a background isolate
  /// where Flutter asset loading is not available. Passes
  /// `useIsolateInterpreter: false` to skip the nested IsolateInterpreter
  /// that would otherwise add a per-inference message hop while already
  /// running inside a worker isolate.
  Future<void> initializeFromBuffer(
    Uint8List modelBytes, {
    PerformanceConfig? performanceConfig,
    bool useNnApi = false,
  }) async {
    await _initWith(
      (options) async => Interpreter.fromBuffer(modelBytes, options: options),
      performanceConfig,
      useIsolateInterpreter: false,
      useNnApi: useNnApi,
    );
  }

  /// Initializes the palm detector from pre-loaded bytes using the LiteRT Next
  /// [CompiledModel] engine (GPU with CPU fallback) instead of the classic
  /// Interpreter pool.
  ///
  /// Used inside the detection isolate when CompiledModel is requested. Input
  /// size and box stride are derived from the compiled model's tensor byte
  /// sizes. Throws [UnsupportedError] if the model's I/O shapes are not the
  /// expected square-input / anchor-aligned-output palm layout (the caller
  /// falls back to the Interpreter engine).
  Future<void> initializeCompiledFromBuffer(
    Uint8List modelBytes, {
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) async {
    if (_isInitialized) await dispose();
    final compiled =
        setEquals(accelerators, const {Accelerator.gpu, Accelerator.cpu})
            ? CompiledModel.fromBufferWithGpuFallback(modelBytes,
                precision: precision)
            : CompiledModel.fromBuffer(modelBytes,
                accelerators: accelerators, precision: precision);
    try {
      _setupCompiled(compiled);
    } catch (_) {
      compiled.close();
      rethrow;
    }
    _compiled = compiled;
    _isInitialized = true;
  }

  Future<void> _initWith(
    Future<Interpreter> Function(InterpreterOptions) loader,
    PerformanceConfig? performanceConfig, {
    bool useIsolateInterpreter = true,
    bool useNnApi = false,
  }) async {
    if (_isInitialized) await dispose();
    await _pool.initialize(
      (options, _) async {
        if (useNnApi) requestNnApi(options);
        final interpreter = await loader(options);
        interpreter.allocateTensors();
        _setupAnchorsAndBuffers(interpreter);
        return interpreter;
      },
      performanceConfig: performanceConfig,
      useIsolateInterpreter: useIsolateInterpreter,
    );
    _isInitialized = true;
  }

  void _setupAnchorsAndBuffers(Interpreter interpreter) {
    final inTensor = interpreter.getInputTensor(0);
    final inShape = inTensor.shape;
    _inH = inShape[1];
    _inW = inShape[2];

    // Read the box stride from the actual output tensor shape so we don't
    // hardcode 18 (some model variants use a different layout).
    final boxesShape = interpreter.getOutputTensor(0).shape;
    _boxStride = boxesShape.last;

    if (inTensor.type == TensorType.uint8) {
      final boxesTensor = interpreter.getOutputTensor(0);
      final scoresTensor = interpreter.getOutputTensor(1);
      if (boxesTensor.type != TensorType.uint8 ||
          scoresTensor.type != TensorType.uint8) {
        throw UnsupportedError(
          'Quantized palm models must use uint8 input and outputs.',
        );
      }
      final inputParams = inTensor.params;
      if ((inputParams.scale - (1.0 / 255.0)).abs() > 1e-7 ||
          inputParams.zeroPoint != 0) {
        throw UnsupportedError(
          'The uint8 palm input must use scale=1/255 and zeroPoint=0.',
        );
      }
      if (boxesShape.length < 3 ||
          scoresTensor.numElements() != boxesShape[1]) {
        throw UnsupportedError(
          'Quantized palm boxes and scores have incompatible candidate counts.',
        );
      }

      _usesUint8Io = true;
      // Qualcomm's W8A8 graph applies its 2944-anchor transform internally,
      // so the original MediaPipe 192x192 external anchor table is neither
      // needed nor valid for this 256x256 model.
      _anchors = const <List<double>>[];
      _uint8InputBuffer = Uint8List(inTensor.numBytes());
      _uint8BoxesData = Uint8List(boxesTensor.numBytes());
      _uint8ScoresData = Uint8List(scoresTensor.numBytes());
      _uint8InputTensor = inTensor;
      _uint8BoxesTensor = boxesTensor;
      _uint8ScoresTensor = scoresTensor;
      final boxesParams = boxesTensor.params;
      final scoresParams = scoresTensor.params;
      _uint8BoxesScale = boxesParams.scale;
      _uint8BoxesZeroPoint = boxesParams.zeroPoint;
      _uint8ScoresScale = scoresParams.scale;
      _uint8ScoresZeroPoint = scoresParams.zeroPoint;
      _boxesData = null;
      _scoresData = null;
      _inputBuffer = null;
      _views = null;
      return;
    }

    if (inTensor.type != TensorType.float32) {
      throw UnsupportedError(
        'Palm input type ${inTensor.type} is unsupported; expected float32 '
        'or uint8.',
      );
    }

    _usesUint8Io = false;
    _anchors = buildPalmAnchors(_inH, _inW);
    final numAnchors = _anchors.length;

    // Allocate flat Float32 outputs and pass their .buffer (ByteBuffer)
    // to TFLite. This avoids the slow Tensor.copyTo path that otherwise
    // boxes ~38k Doubles per inference.
    _boxesData = Float32List(numAnchors * _boxStride);
    _scoresData = Float32List(numAnchors);

    // Pre-allocate the input buffer eagerly so the first inference doesn't
    // pay an alloc.
    _inputBuffer = Float32List(_inH * _inW * 3);

    // Cache Float32List views into the interpreter's tensor native memory
    // for the direct-invoke path. Tensors are stable after allocateTensors()
    // so the views remain valid for the lifetime of this detector.
    _views = TensorFloat32Views.capture(interpreter);
  }

  /// Derives input size, anchors, and box stride from a [CompiledModel]'s
  /// tensor byte sizes (there is no Interpreter to query shapes from).
  void _setupCompiled(CompiledModel compiled) {
    final int side = compiledSquareInputSide(compiled, label: 'palm detection');
    _inH = side;
    _inW = side;
    _anchors = buildPalmAnchors(_inH, _inW);
    final int numAnchors = _anchors.length;

    if (compiled.outputCount < 2) {
      throw UnsupportedError(
        'Compiled palm detection expects at least two outputs; got '
        '${compiled.outputCount}.',
      );
    }
    final int out0 = compiled.outputByteSizes[0] ~/ 4;
    final int out1 = compiled.outputByteSizes[1] ~/ 4;
    // Scores carry one value per anchor; boxes carry `stride` values per anchor.
    // Resolve the index order from the byte sizes so we do not assume it.
    if (out1 == numAnchors && out0 % numAnchors == 0) {
      _cmBoxesIdx = 0;
      _cmScoresIdx = 1;
      _boxStride = out0 ~/ numAnchors;
    } else if (out0 == numAnchors && out1 % numAnchors == 0) {
      _cmBoxesIdx = 1;
      _cmScoresIdx = 0;
      _boxStride = out1 ~/ numAnchors;
    } else {
      throw UnsupportedError(
        'Compiled palm detection outputs ($out0, $out1 floats) do not align '
        'with $numAnchors anchors.',
      );
    }

    _boxesData = Float32List(numAnchors * _boxStride);
    _scoresData = Float32List(numAnchors);
    _inputBuffer = Float32List(_inH * _inW * 3);
  }

  /// Returns true if the detector has been initialized.
  bool get isInitialized => _isInitialized;

  /// Disposes the detector and releases resources.
  Future<void> dispose() async {
    await _pool.dispose();
    _compiled?.close();
    _compiled = null;
    _inputBuffer = null;
    _boxesData = null;
    _scoresData = null;
    _views = null;
    _usesUint8Io = false;
    _uint8InputBuffer = null;
    _uint8BoxesData = null;
    _uint8ScoresData = null;
    _uint8InputTensor = null;
    _uint8BoxesTensor = null;
    _uint8ScoresTensor = null;
    _isInitialized = false;
  }

  /// Detects palms in the given image.
  ///
  /// Returns a list of [PalmDetection] objects containing rotation rectangle
  /// parameters for each detected palm.
  Future<List<PalmDetection>> detectOnMat(cv.Mat image) async {
    if (!_isInitialized) {
      throw StateError('PalmDetector not initialized.');
    }

    _imageHeight = image.rows;
    _imageWidth = image.cols;

    _squareStandardSize = math.max(_imageHeight, _imageWidth);
    _squarePaddingHalfSize = (_imageHeight - _imageWidth).abs() ~/ 2;

    final (paddedImage, resizedImage) = ImageUtils.keepAspectResizeAndPad(
      image,
      _inW,
      _inH,
    );

    Float32List? boxesView;
    Float32List? scoresView;
    Uint8List? uint8BoxesView;
    Uint8List? uint8ScoresView;

    final compiled = _compiled;
    if (compiled != null) {
      // CompiledModel (LiteRT Next) path: preprocess into the scratch input
      // buffer, then runAsync returns fresh Float32List outputs we decode
      // directly. Same [0,1] RGB preprocessing as the Interpreter path.
      final input = _inputBuffer!;
      ImageUtils.matToFloat32Tensor(paddedImage, buffer: input);
      final outputs = await compiled.runAsync([input]);
      boxesView = outputs[_cmBoxesIdx];
      scoresView = outputs[_cmScoresIdx];
    } else {
      await _pool.withInterpreter((interp, iso) async {
        if (_usesUint8Io) {
          final input = _uint8InputBuffer!;
          ImageUtils.matToRgbUint8Tensor(paddedImage, buffer: input);
          if (iso != null) {
            await iso.runForMultipleInputs(
              <Object>[input.buffer],
              <int, Object>{
                0: _uint8BoxesData!.buffer,
                1: _uint8ScoresData!.buffer,
              },
            );
          } else {
            _uint8InputTensor!.data = input;
            interp.invoke();
            _uint8BoxesTensor!.copyTo(_uint8BoxesData!);
            _uint8ScoresTensor!.copyTo(_uint8ScoresData!);
          }
          uint8BoxesView = _uint8BoxesData!;
          uint8ScoresView = _uint8ScoresData!;
        } else if (iso != null) {
          // IsolateInterpreter path: must go through runForMultipleInputs.
          // Convert into our scratch _inputBuffer first, then ship its
          // ByteBuffer to the iso. Outputs land in _boxesData / _scoresData
          // via the ByteBuffer fast branch of Tensor.copyTo.
          ImageUtils.matToFloat32Tensor(paddedImage, buffer: _inputBuffer);
          await iso.runForMultipleInputs(
            [_inputBuffer!.buffer],
            <int, Object>{
              0: _boxesData!.buffer,
              1: _scoresData!.buffer,
            },
          );
          boxesView = _boxesData!;
          scoresView = _scoresData!;
        } else {
          // Direct path (no nested isolate): write the BGR→RGB normalized
          // tensor straight into the input tensor's native memory, then
          // invoke() and read output tensors as Float32List views, no
          // runForMultipleInputs, no Tensor.copyTo, no marshalling. The
          // tensor views are cached at init so there is no per-inference
          // wrapper allocation here.
          final views = _views!;
          ImageUtils.matToFloat32Tensor(paddedImage, buffer: views.inputs[0]);

          interp.invoke();

          boxesView = views.outputs[0];
          scoresView = views.outputs[1];
        }
      });
    }

    resizedImage.dispose();
    paddedImage.dispose();

    final decodedBoxes = _usesUint8Io
        ? decodePostprocessedUint8PalmBoxes(
            uint8BoxesView!,
            uint8ScoresView!,
            _boxStride,
            scoreThreshold,
            boxScale: _uint8BoxesScale,
            boxZeroPoint: _uint8BoxesZeroPoint,
            scoreScale: _uint8ScoresScale,
            scoreZeroPoint: _uint8ScoresZeroPoint,
            inputWidth: _inW,
            inputHeight: _inH,
          )
        : decodePalmBoxes(
            boxesView!,
            scoresView!,
            _anchors,
            _boxStride,
            scoreThreshold,
            scale: _inW.toDouble(),
          );
    return postprocessPalms(
      decodedBoxes,
      imageWidth: _imageWidth,
      imageHeight: _imageHeight,
      squareStandardSize: _squareStandardSize,
      squarePaddingHalfSize: _squarePaddingHalfSize,
      roiScale: roiScale,
      iouThreshold: nmsIouThreshold,
    );
  }

  /// Exposes anchor generation for testing.
  @visibleForTesting
  List<List<double>> get anchorsForTest => _anchors;

  /// Exposes input width for testing.
  @visibleForTesting
  int get inputWidth => _inW;

  /// Exposes input height for testing.
  @visibleForTesting
  int get inputHeight => _inH;
}
