import 'dart:convert';

import 'command_controller.dart';

class StaticTensorContract {
  const StaticTensorContract(
      {required this.name,
      required this.dtype,
      required this.shape,
      required this.index,
      required this.scale,
      required this.zeroPoint});

  factory StaticTensorContract.fromJson(Map<String, dynamic> json) {
    final dtype = json['dtype'] as String;
    final shape = List<int>.from(json['shape'] as List);
    final scale = (json['scale'] as num?)?.toDouble() ?? 0;
    final zeroPoint = (json['zero_point'] as num?)?.toInt() ?? 0;
    final index = (json['index'] as num).toInt();
    if (!const <String>{'float32', 'int8'}.contains(dtype) ||
        index < 0 ||
        shape.any((dim) => dim <= 0) ||
        !scale.isFinite ||
        (dtype == 'int8' &&
            (scale <= 0 || zeroPoint < -128 || zeroPoint > 127))) {
      throw const FormatException('Invalid static tensor contract.');
    }
    return StaticTensorContract(
        name: json['name'] as String,
        dtype: dtype,
        shape: List<int>.unmodifiable(shape),
        index: index,
        scale: scale,
        zeroPoint: zeroPoint);
  }

  final String name;
  final String dtype;
  final List<int> shape;
  final int index;
  final double scale;
  final int zeroPoint;
  int get elements => shape.fold(1, (a, b) => a * b);
}

class StaticModelManifest {
  StaticModelManifest.decode(String source) {
    final root = jsonDecode(source) as Map<String, dynamic>;
    if (root['schema_version'] != 1 || root['normalization_embedded'] != true) {
      throw const FormatException('Unsupported static model manifest.');
    }
    if ((root['qualification'] as Map?)?['passed'] != true) {
      throw const FormatException(
          'Static model has not passed release qualification.');
    }
    modelName = root['model_name'] as String;
    modelFile = root['model_file'] as String;
    modelSha256 = root['model_sha256'] as String;
    sourceOnnxSha256 = root['source_onnx_sha256'] as String;
    modelBytes = (root['model_bytes'] as num).toInt();
    labels =
        List<String>.unmodifiable(List<String>.from(root['labels'] as List));
    featureNames = List<String>.unmodifiable(
        List<String>.from(root['feature_names'] as List));
    input = StaticTensorContract.fromJson(
        Map<String, dynamic>.from(root['input'] as Map));
    final rawOutputs = root['outputs'] as Map<String, dynamic>;
    outputs = Map<String, StaticTensorContract>.unmodifiable({
      for (final name in <String>['probabilities', 'known_gesture_mass'])
        name: StaticTensorContract.fromJson(
            Map<String, dynamic>.from(rawOutputs[name] as Map)),
    });
    if (modelBytes <= 0 ||
        !RegExp(r'^[0-9a-f]{64}$').hasMatch(modelSha256) ||
        !RegExp(r'^[0-9a-f]{64}$').hasMatch(sourceOnnxSha256) ||
        !_same(labels, staticGestureLabels) ||
        !_same(featureNames, staticFeatureNames) ||
        featureNames.toSet().length != 76 ||
        input.index != 0 ||
        !_same(input.shape, <int>[1, 76]) ||
        !_same(outputs['probabilities']!.shape, <int>[1, 13]) ||
        !_same(outputs['known_gesture_mass']!.shape, <int>[1, 1]) ||
        outputs.values.map((value) => value.index).toSet().length != 2 ||
        outputs.values.any((value) => value.index > 1)) {
      throw const FormatException('Static model contract mismatch.');
    }
  }

  late final String modelName;
  late final String modelFile;
  late final String modelSha256;
  late final String sourceOnnxSha256;
  late final int modelBytes;
  late final List<String> labels;
  late final List<String> featureNames;
  late final StaticTensorContract input;
  late final Map<String, StaticTensorContract> outputs;
}

final List<String> staticFeatureNames = List<String>.unmodifiable(<String>[
  for (var i = 0; i < 21; i++) ...<String>[
    'landmark_${i}_x',
    'landmark_${i}_y'
  ],
  for (var i = 0; i < 21; i++) 'landmark_${i}_radius',
  'index_segment_dx',
  'index_segment_dy',
  'wrist_index_dx',
  'wrist_index_dy',
  'thumb_index_gap_ratio',
  'thumb_extension',
  'index_extension',
  'middle_extension',
  'ring_extension',
  'pinky_extension',
  'other_fingers_folded',
  'thumb_tip_index_mcp_ratio',
  'thumb_reach_ratio',
]);

bool _same<T>(List<T> a, List<T> b) =>
    a.length == b.length &&
    List<bool>.generate(a.length, (i) => a[i] == b[i]).every((v) => v);
