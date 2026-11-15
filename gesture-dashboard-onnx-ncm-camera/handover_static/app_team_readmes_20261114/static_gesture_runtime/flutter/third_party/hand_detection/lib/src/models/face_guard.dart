import 'dart:typed_data';
import 'package:flutter_litert/native.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import '../util/image_utils.dart';
import '../shared/false_hand_policy.dart';

/// Persistent, one-thread CPU model, called only inside the detector isolate.
class FaceGuardRunner {
  Interpreter? _model;
  TensorFloat32Views? _views;
  final cache = FaceBoxCache();
  final _clock = Stopwatch()..start();
  int _boxesIndex = 0, _scoresIndex = 1;

  void initialize(Uint8List bytes) {
    final options = InterpreterOptions()..threads = 1;
    try {
      final model = _model = Interpreter.fromBuffer(bytes, options: options);
      model.allocateTensors();
      final input = model.getInputTensor(0);
      if (input.type != TensorType.float32 ||
          input.shape.join(',') != '1,128,128,3') {
        throw StateError('Unexpected BlazeFace input');
      }
      final outputs = model.getOutputTensors();
      _boxesIndex = outputs.indexWhere(
          (v) => v.numElements() == 896 * 16 && v.type == TensorType.float32);
      _scoresIndex = outputs.indexWhere(
          (v) => v.numElements() == 896 && v.type == TensorType.float32);
      if (outputs.length != 2 || _boxesIndex < 0 || _scoresIndex < 0)
        throw StateError('Unexpected BlazeFace outputs');
      _views = TensorFloat32Views.capture(model);
    } catch (_) {
      dispose();
      rethrow;
    } finally {
      options.delete();
    }
  }

  List<FaceBox> detect(cv.Mat image) {
    final now = _clock.elapsedMicroseconds / 1000;
    if (!cache.refresh(now, image.cols, image.rows)) return cache.boxes;
    final (padded, resized) =
        ImageUtils.keepAspectResizeAndPad(image, 128, 128);
    try {
      final input = _views!.inputs[0];
      ImageUtils.matToFloat32Tensor(padded, buffer: input);
      for (var i = 0; i < input.length; i++) {
        input[i] = input[i] * 2 - 1;
      }
      _model!.invoke();
      cache.observe(
          decodeFaces(_views!.outputs[_boxesIndex],
              _views!.outputs[_scoresIndex], image.cols, image.rows),
          now);
      return cache.boxes;
    } catch (_) {
      cache.reset();
      rethrow;
    } finally {
      padded.dispose();
      resized.dispose();
    }
  }

  void reset() => cache.reset();
  void dispose() {
    _views = null;
    _model?.close();
    _model = null;
    cache.reset();
  }
}
