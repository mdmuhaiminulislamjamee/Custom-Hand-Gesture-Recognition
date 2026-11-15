import 'dart:io';

import 'package:flutter_litert/native.dart' show InterpreterOptions;
// flutter_litert exposes this NNAPI symbol only through its generated binding.
// ignore: implementation_imports
import 'package:flutter_litert/src/bindings/bindings.dart' show tfliteBinding;

/// Enables LiteRT's legacy NNAPI request on an interpreter options object.
///
/// flutter_litert 3.5.1 includes the C API symbol but does not expose NNAPI in
/// PerformanceConfig. Keep this small bridge isolated so it can be removed
/// when the dependency gains a supported public NNAPI API.
void requestNnApi(InterpreterOptions options) {
  if (!Platform.isAndroid) {
    throw UnsupportedError('NNAPI is available only on Android.');
  }
  tfliteBinding.TfLiteInterpreterOptionsSetUseNNAPI(options.base, true);
}
