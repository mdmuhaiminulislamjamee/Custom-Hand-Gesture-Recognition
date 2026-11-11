import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:static_gesture_runtime/src/geometry_policy.dart';

void main() {
  test(
      'model-backed pointing matches Python; low mass and second fingers cannot recover',
      () {
    final fixture = jsonDecode(
        File('../validation/direction_pose_reference.json')
            .readAsStringSync()) as Map;
    List<List<double>> points(Object data) => [
          for (final p in data as List)
            [for (final x in p as List) (x as num).toDouble()]
        ];
    for (final row in fixture['cases'] as List) {
      for (final expected in row['expected'] as List) {
        final result = StaticGeometryPolicy.evaluate(
            probabilities: [
              for (final x in row['probabilities'] as List)
                (x as num).toDouble()
            ],
            knownMass: (expected['mass'] as num).toDouble(),
            normalizedLandmarks: points(row['landmarks']),
            worldLandmarks: points(row['world_landmarks']),
            physicalHand: row['physical_hand'] as String,
            handednessConfidence: .99,
            width: (row['shape'][1] as num).toDouble(),
            height: (row['shape'][0] as num).toDouble());
        expect(result.accepted, expected['accepted'],
            reason: 'case ${row['id']} mass ${expected['mass']}');
        expect(result.label, expected['label'],
            reason: 'case ${row['id']} mass ${expected['mass']}');
      }
    }
  });
}
