import 'dart:async';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';

import 'model_manifest.dart';
import 'runtime_api.dart';

enum StaticTfliteModel { float32, int8 }

/// CPU LiteRT interpreter with one persistent background inference isolate.
/// The larger MediaPipe detector may independently use its GPU/Core ML backend.
class StaticTfliteClassifier implements StaticGestureClassifier {
  StaticTfliteClassifier(
      {this.model = StaticTfliteModel.float32,
      this.assetPrefix = 'packages/static_gesture_runtime/assets/models',
      AssetBundle? assetBundle})
      : _bundle = assetBundle ?? rootBundle;

  final StaticTfliteModel model;
  final String assetPrefix;
  final AssetBundle _bundle;
  StaticModelManifest? _manifest;
  Interpreter? _interpreter;
  IsolateInterpreter? _worker;
  Future<void>? _initializing;
  Completer<void>? _active;
  bool _closed = false;
  late Object _input;
  late List<Object> _inputs;
  late Map<int, Object> _outputs;

  StaticModelManifest? get manifest => _manifest;
  String get backend => 'LiteRT CPU (one inference isolate)';
  @override
  bool get isInitialized => !_closed && _worker != null;

  @override
  Future<void> initialize() {
    if (_closed) throw StateError('Static classifier is closed.');
    return _initializing ??= _initialize();
  }

  Future<void> _initialize() async {
    final stem = 'gesture_static_${model.name}';
    final manifest = StaticModelManifest.decode(
        await _bundle.loadString('$assetPrefix/${stem}_manifest.json'));
    final data = await _bundle.load('$assetPrefix/${manifest.modelFile}');
    final bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    if (bytes.length != manifest.modelBytes ||
        sha256.convert(bytes).toString() != manifest.modelSha256) {
      throw StateError(
          'Static TFLite model hash/size differs from its manifest.');
    }
    final options = InterpreterOptions()..threads = 1;
    Interpreter? interpreter;
    IsolateInterpreter? worker;
    try {
      interpreter = Interpreter.fromBuffer(bytes, options: options);
      _validateInterpreter(interpreter, manifest);
      worker = await IsolateInterpreter.create(
          address: interpreter.address,
          debugName: 'StaticGestureClassifier-${model.name}');
      if (_closed)
        throw StateError('Static classifier closed during initialization.');
      _input = _newBuffer(manifest.input);
      _inputs = <Object>[_input];
      _outputs = <int, Object>{
        for (final entry in manifest.outputs.values)
          entry.index: _newBuffer(entry)
      };
      _manifest = manifest;
      _interpreter = interpreter;
      _worker = worker;
    } catch (_) {
      await worker?.close();
      interpreter?.close();
      rethrow;
    } finally {
      options.delete();
    }
  }

  @override
  Future<StaticClassifierOutput> predict(Float32List features) async {
    if (!isInitialized)
      throw StateError('Initialize the static classifier first.');
    if (_active != null)
      throw StateError(
          'Classifier is serial; submit frames through StaticGestureRuntimeApi.');
    if (features.length != 76 || features.any((value) => !value.isFinite)) {
      throw ArgumentError('Expected 76 finite float32 features.');
    }
    final active = Completer<void>();
    _active = active;
    try {
      final manifest = _manifest!;
      if (_input is Float32List) {
        (_input as Float32List).setAll(0, features);
      } else {
        final quantized = _input as Int8List;
        for (var i = 0; i < features.length; i++) {
          quantized[i] = roundHalfToEven(
                  features[i] / manifest.input.scale + manifest.input.zeroPoint)
              .clamp(-128, 127);
        }
      }
      final watch = Stopwatch()..start();
      await _worker!.runForMultipleInputs(_inputs, _outputs);
      watch.stop();
      final probabilityContract = manifest.outputs['probabilities']!;
      final probabilities = _read(probabilityContract);
      final mass = _read(manifest.outputs['known_gesture_mass']!).single;
      final sum = probabilities.fold(0.0, (a, b) => a + b);
      if (!sum.isFinite ||
          sum <= 0 ||
          probabilities.any((v) => !v.isFinite || v < -1e-6 || v > 1.01) ||
          !mass.isFinite ||
          mass < -0.01 ||
          mass > 1.01) {
        throw StateError('Invalid TFLite gesture probabilities or known mass.');
      }
      return StaticClassifierOutput(
          probabilities: probabilities,
          knownMass: mass.clamp(0, 1),
          inferenceMs: watch.elapsedMicroseconds / 1000);
    } finally {
      _active = null;
      active.complete();
    }
  }

  Object _newBuffer(StaticTensorContract contract) =>
      contract.dtype == 'float32'
          ? Float32List(contract.elements)
          : Int8List(contract.elements);

  List<double> _read(StaticTensorContract contract) {
    final data = _outputs[contract.index]!;
    if (data is Float32List) return data.toList(growable: false);
    return (data as Int8List)
        .map((v) => (v - contract.zeroPoint) * contract.scale)
        .toList(growable: false);
  }

  void _validateInterpreter(
      Interpreter interpreter, StaticModelManifest manifest) {
    if (interpreter.getInputTensors().length != 1 ||
        interpreter.getOutputTensors().length != 2) {
      throw StateError('Static TFLite tensor count differs from manifest.');
    }
    _validateTensor(
        interpreter.getInputTensor(manifest.input.index), manifest.input);
    for (final contract in manifest.outputs.values) {
      _validateTensor(interpreter.getOutputTensor(contract.index), contract);
    }
  }

  void _validateTensor(Tensor tensor, StaticTensorContract contract) {
    final expected =
        contract.dtype == 'float32' ? TensorType.float32 : TensorType.int8;
    if (tensor.type != expected ||
        tensor.shape.length != contract.shape.length ||
        List<int>.generate(tensor.shape.length, (i) => i)
            .any((i) => tensor.shape[i] != contract.shape[i])) {
      throw StateError(
          '${contract.name}: TFLite dtype/shape differs from manifest.');
    }
    if (contract.dtype == 'int8' &&
        ((tensor.params.scale - contract.scale).abs() > 1e-7 ||
            tensor.params.zeroPoint != contract.zeroPoint)) {
      throw StateError(
          '${contract.name}: TFLite quantization differs from manifest.');
    }
  }

  @override
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    try {
      await _initializing;
    } catch (_) {/* Already reported by initialize. */}
    await _active?.future;
    await _worker?.close();
    _worker = null;
    _interpreter?.close();
    _interpreter = null;
    _manifest = null;
  }
}

/// Matches NumPy/TFLite calibration rounding at exact half-way values.
int roundHalfToEven(double value) {
  final lower = value.floor();
  final fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower.isEven ? lower : lower + 1;
}
