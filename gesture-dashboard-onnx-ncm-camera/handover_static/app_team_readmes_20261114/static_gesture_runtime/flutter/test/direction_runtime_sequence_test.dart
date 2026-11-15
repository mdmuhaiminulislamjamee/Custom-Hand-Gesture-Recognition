import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:static_gesture_runtime/src/command_controller.dart';
import 'package:static_gesture_runtime/src/runtime_api.dart';

class _DirectionClassifier implements StaticGestureClassifier {
  _DirectionClassifier(this.label);
  final String label;
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
        for (final x in staticGestureLabels) x == label ? 1 : 0
      ], knownMass: 1);
}

StaticGestureFrame frame(String direction, double time, double tilt) {
  final angle = ({
        'up': 0.0,
        'left': math.pi / 2,
        'down': math.pi,
        'right': -math.pi / 2
      }[direction]!) +
      tilt;
  const pose = [
    [.50, .72],
    [.42, .64],
    [.37, .57],
    [.34, .52],
    [.31, .48],
    [.48, .58],
    [.48, .45],
    [.48, .31],
    [.48, .16],
    [.55, .57],
    [.57, .55],
    [.58, .58],
    [.57, .62],
    [.61, .59],
    [.63, .58],
    [.64, .62],
    [.62, .66],
    [.66, .62],
    [.68, .62],
    [.69, .66],
    [.67, .70],
  ];
  final points = [
    for (final p in pose)
      [
        .5 + (p[0] - .5) * math.cos(angle) - (p[1] - .72) * math.sin(angle),
        .72 + (p[0] - .5) * math.sin(angle) + (p[1] - .72) * math.cos(angle),
        0.0
      ]
  ];
  return StaticGestureFrame(
      timestampMs: time,
      frameWidth: 600,
      frameHeight: 600,
      hands: [
        StaticGestureHand(
            hand: StaticPhysicalHand.right,
            normalizedLandmarks: points,
            handednessConfidence: .99)
      ]);
}

void main() {
  for (final direction in ['left', 'right', 'up', 'down']) {
    test(
        '$direction full geometry keeps one marginal frame, releases two, and rearms',
        () async {
      final runtime =
          StaticGestureRuntimeApi(classifier: _DirectionClassifier(direction));
      await runtime.initialize();
      final first = (await runtime.processFrame(frame(direction, 0, 0)))!;
      expect(first.command?.label, direction,
          reason: 'strong entry must have no new hold');
      final jitter =
          (await runtime.processFrame(frame(direction, 100, math.pi / 4)))!;
      expect(jitter.hands.single.geometry.label, direction);
      expect(jitter.command, isNull,
          reason: 'grace cannot emit or duplicate a command');
      final released =
          (await runtime.processFrame(frame(direction, 200, math.pi / 4)))!;
      expect(released.hands.single.geometry.accepted, isFalse);
      expect(released.command, isNull);
      final reacquired =
          (await runtime.processFrame(frame(direction, 300, 0)))!;
      expect(reacquired.command?.label, direction);
      await runtime.close();
    });
    test(
        '$direction diagonal cannot enter; lost hand and expired grace do not persist',
        () async {
      final runtime =
          StaticGestureRuntimeApi(classifier: _DirectionClassifier(direction));
      await runtime.initialize();
      final diagonal =
          (await runtime.processFrame(frame(direction, 0, math.pi / 4)))!;
      expect(diagonal.hands.single.geometry.accepted, isFalse);
      await runtime.processFrame(frame(direction, 100, 0));
      final expired =
          (await runtime.processFrame(frame(direction, 351, math.pi / 4)))!;
      expect(expired.hands.single.geometry.accepted, isFalse);
      await runtime.processFrame(frame(direction, 400, 0));
      await runtime.processFrame(StaticGestureFrame(
          timestampMs: 450, frameWidth: 600, frameHeight: 600, hands: []));
      final returned =
          (await runtime.processFrame(frame(direction, 500, math.pi / 4)))!;
      expect(returned.hands.single.geometry.accepted, isFalse);
      await runtime.close();
    });
  }
}
