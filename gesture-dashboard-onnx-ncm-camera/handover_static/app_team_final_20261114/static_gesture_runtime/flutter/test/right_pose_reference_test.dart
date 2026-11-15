import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:static_gesture_runtime/src/geometry_policy.dart';
import 'package:static_gesture_runtime/src/runtime_api.dart';
import 'package:static_gesture_runtime/src/command_controller.dart';

void main() {
  test('Right projection matches Python with mass and competing-finger guards',
      () {
    final fixture = jsonDecode(
            File('../validation/right_pose_reference.json').readAsStringSync())
        as Map;
    List<List<double>> points(Object data) => [
          for (final p in data as List)
            [for (final x in p as List) (x as num).toDouble()]
        ];
    for (final row in fixture['cases'] as List) {
      for (final expected in row['expected'] as List) {
        final result = StaticGeometryPolicy.evaluate(
          probabilities: [
            for (final x in row['probabilities'] as List) (x as num).toDouble()
          ],
          knownMass: (expected['mass'] as num).toDouble(),
          normalizedLandmarks: points(row['landmarks']),
          worldLandmarks: row['world_landmarks'] == null
              ? null
              : points(row['world_landmarks']),
          physicalHand: row['physical_hand'] as String,
          handednessConfidence: .99,
          width: (row['shape'][1] as num).toDouble(),
          height: (row['shape'][0] as num).toDouble(),
        );
        expect(result.accepted, expected['accepted'],
            reason: 'case ${row['id']} mass ${expected['mass']}');
        expect(result.label, expected['label'],
            reason: 'case ${row['id']} mass ${expected['mass']}');
      }
    }
  });
  test('Right side view keeps one confidence dip and releases two', () async {
    final fixture = jsonDecode(
            File('../validation/right_pose_reference.json').readAsStringSync())
        as Map;
    final row = (fixture['cases'] as List).first as Map;
    List<List<double>> points(Object data) => [
          for (final p in data as List)
            [for (final x in p as List) (x as num).toDouble()]
        ];
    final classifier = _RightClassifier();
    final runtime = StaticGestureRuntimeApi(classifier: classifier);
    await runtime.initialize();
    StaticGestureFrame frame(double time) => StaticGestureFrame(
            timestampMs: time,
            frameWidth: 640,
            frameHeight: 480,
            hands: [
              StaticGestureHand(
                  hand: StaticPhysicalHand.right,
                  normalizedLandmarks: [
                    for (final p in points(row['landmarks'])) [...p, 0.0]
                  ],
                  worldLandmarks: points(row['world_landmarks']),
                  handednessConfidence: .99)
            ]);
    final entered = (await runtime.processFrame(frame(0)))!;
    expect(entered.command?.label, 'right');
    classifier.confidence = .94;
    final dip = (await runtime.processFrame(frame(100)))!;
    expect(dip.hands.single.geometry.label, 'right');
    expect(dip.command, isNull);
    final release = (await runtime.processFrame(frame(200)))!;
    expect(release.hands.single.geometry.accepted, isFalse);
    expect(release.command, isNull);
    classifier.confidence = .96;
    expect((await runtime.processFrame(frame(300)))!.command?.label, 'right');
    await runtime.close();
  });
}

class _RightClassifier implements StaticGestureClassifier {
  double confidence = .96;
  bool ready = false;
  @override
  bool get isInitialized => ready;
  @override
  Future<void> initialize() async {
    ready = true;
  }

  @override
  Future<void> close() async {
    ready = false;
  }

  @override
  Future<StaticClassifierOutput> predict(Float32List features) async =>
      StaticClassifierOutput(probabilities: [
        for (final label in staticGestureLabels)
          label == 'right' ? confidence : (1 - confidence) / 12
      ], knownMass: 1);
}
