import 'package:flutter_test/flutter_test.dart';
import 'package:static_gesture_runtime/src/direction_stability.dart';
import 'package:static_gesture_runtime/src/command_controller.dart';

String? observe(DirectionStability state, String candidate, double now,
        {bool qualified = false,
        bool confirmed = false,
        bool safe = true,
        bool hard = false,
        double alignment = .70,
        String raw = 'left'}) =>
    state.update(
        candidate: candidate,
        qualified: qualified,
        confirmed: confirmed,
        nowMs: now,
        directional: {
          'continuation_safe': safe,
          'axis_dominance': .71,
          'axis_alignment': {
            for (final d in DirectionStability.directions) d: alignment
          }
        },
        hardRejection: hard,
        confidence: .94,
        margin: .80,
        knownMass: .99,
        rawLabel: raw);

void main() {
  for (final direction in DirectionStability.directions) {
    test(
        '$direction bridges one dropout and releases two; deadline never refreshes',
        () {
      final state = DirectionStability();
      observe(state, direction, 0, qualified: true, confirmed: true);
      expect(observe(state, 'no_gesture', 100), direction);
      expect(observe(state, 'no_gesture', 200), isNull);
      expect(state.label, isNull);
      observe(state, direction, 300, qualified: true, confirmed: true);
      expect(observe(state, 'no_gesture', 501), isNull);
      expect(state.label, isNull);
    });
    test(
        '$direction cannot bridge another gesture, unsafe shape or opposite axis',
        () {
      for (final other in [
        'peace',
        'rock',
        'ok',
        'fist',
        'like',
        'thumb_down',
        'open_palm',
        'stop_inverted'
      ]) {
        final state = DirectionStability();
        observe(state, direction, 0, qualified: true, confirmed: true);
        expect(observe(state, other, 100, qualified: true), isNull);
        expect(state.label, isNull);
      }
      final state = DirectionStability();
      observe(state, direction, 0, qualified: true, confirmed: true);
      expect(observe(state, 'no_gesture', 100, safe: false), isNull);
      expect(observe(state, 'no_gesture', 150, alignment: -1), isNull);
    });
    test(
        '$direction neutral release must be consecutive and grace cannot duplicate an action',
        () {
      final gate = StaticCommandController();
      StaticControllerResult frame(String label, double time) => gate.process(
          label: label, confidence: .99, margin: .98, timestampMs: time);
      expect(frame(direction, 0).emit, isTrue);
      expect(frame('no_gesture', 100).confirmed, isFalse);
      expect(frame(direction, 200).emit, isFalse);
      expect(frame('no_gesture', 300).confirmed, isFalse);
      expect(frame('no_gesture', 400).confirmed, isTrue);
      expect(frame(direction, 500).emit, isTrue);
    });
  }
  test(
      'boundary direction cannot acquire indefinitely; clear opposite has no added hold',
      () {
    final state = DirectionStability();
    observe(state, 'left', 0, qualified: true, confirmed: true);
    expect(observe(state, 'up', 100, qualified: true, confirmed: true), 'left');
    expect(observe(state, 'up', 200, qualified: true, confirmed: true),
        'no_gesture');
    observe(state, 'left', 300, qualified: true, confirmed: true);
    expect(
        observe(state, 'right', 400,
            qualified: true, confirmed: true, alignment: -1),
        isNull);
    expect(state.label, 'right');
  });
}
