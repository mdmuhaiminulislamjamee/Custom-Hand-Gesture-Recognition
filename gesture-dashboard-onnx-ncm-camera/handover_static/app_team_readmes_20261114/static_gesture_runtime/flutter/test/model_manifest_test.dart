import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:static_gesture_runtime/src/model_manifest.dart';

void main() {
  final source = File('assets/models/gesture_static_float32_manifest.json')
      .readAsStringSync();
  test(
      'qualified production manifest preserves the 13 labels and exact 76 feature order',
      () {
    final manifest = StaticModelManifest.decode(source);
    expect(manifest.input.shape, <int>[1, 76]);
    expect(manifest.featureNames, staticFeatureNames);
    expect(manifest.labels[6], 'dorsal');
  });
  test('unqualified quantized candidate refuses initialization', () {
    final candidate = File('assets/models/gesture_static_int8_manifest.json')
        .readAsStringSync();
    expect(() => StaticModelManifest.decode(candidate), throwsFormatException);
  });
  test('feature permutations and output overlap fail closed', () {
    final json = jsonDecode(source) as Map<String, dynamic>;
    final features = json['feature_names'] as List;
    final first = features[0];
    features[0] = features[1];
    features[1] = first;
    expect(() => StaticModelManifest.decode(jsonEncode(json)),
        throwsFormatException);
    final other = jsonDecode(source) as Map<String, dynamic>;
    other['outputs']['known_gesture_mass']['index'] =
        other['outputs']['probabilities']['index'];
    expect(() => StaticModelManifest.decode(jsonEncode(other)),
        throwsFormatException);
  });
}
