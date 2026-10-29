import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:static_gesture_runtime/src/tflite_classifier.dart';

class _FileAssets extends CachingAssetBundle {
  @override
  Future<ByteData> load(String key) async =>
      ByteData.sublistView(await File(key).readAsBytes());
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('persistent native LiteRT worker matches all ONNX reference outputs',
      () async {
    final classifier = StaticTfliteClassifier(
        assetBundle: _FileAssets(), assetPrefix: 'assets/models');
    try {
      await classifier.initialize();
      final fixture = jsonDecode(
              File('../validation/model_reference.json').readAsStringSync())
          as Map<String, dynamic>;
      final cases = fixture['model_cases'] as List;
      expect(cases.length, 26);
      double maximumError = 0;
      for (final raw in cases) {
        final row = raw as Map<String, dynamic>;
        final features = Float32List.fromList((row['features'] as List)
            .map((v) => (v as num).toDouble())
            .toList());
        final actual = await classifier.predict(features);
        final expected = (row['probabilities'] as List)
            .map((v) => (v as num).toDouble())
            .toList();
        for (var i = 0; i < 13; i++) {
          final error = (actual.probabilities[i] - expected[i]).abs();
          if (error > maximumError) maximumError = error;
          expect(error, lessThanOrEqualTo(.0001),
              reason: 'class $i for reference ${row['label']}');
        }
        expect(
            (actual.knownMass - (row['known_gesture_mass'] as num).toDouble())
                .abs(),
            lessThanOrEqualTo(.0001));
      }
      // Ensure buffers/interpreter remain reusable for repeated predictions.
      final features = Float32List.fromList(
          ((cases.first as Map)['features'] as List)
              .map((v) => (v as num).toDouble())
              .toList());
      final repeated = await classifier.predict(features);
      expect(repeated.probabilities.length, 13);
      // ignore: avoid_print
      print(
          'Native LiteRT: ${cases.length} ONNX cases passed; max probability error $maximumError');
    } finally {
      await classifier.close();
    }
  });

  test('unqualified INT8 candidate cannot load even with floating-point IO',
      () async {
    final classifier = StaticTfliteClassifier(
        model: StaticTfliteModel.int8,
        assetBundle: _FileAssets(),
        assetPrefix: 'assets/models');
    try {
      await expectLater(classifier.initialize(), throwsFormatException);
      expect(classifier.isInitialized, isFalse);
    } finally {
      await classifier.close();
    }
  });
}
