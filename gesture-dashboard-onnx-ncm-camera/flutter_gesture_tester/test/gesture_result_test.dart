import 'package:flutter_test/flutter_test.dart';
import 'package:gesture_model_tester/models/gesture_result.dart';
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';

void main() {
  StaticHandResult hand(StaticPhysicalHand side, double confidence) =>
      StaticHandResult(
        hand: side,
        geometry: StaticGeometryDecision(
          label: side == StaticPhysicalHand.left ? 'left' : 'right',
          accepted: true,
          confidence: confidence,
          margin: .8,
          minimumHoldMs: 0,
          reason: 'valid',
        ),
        controller: const StaticControllerResult(
          label: 'left',
          confirmed: true,
          emit: false,
          reason: 'test',
          matches: 3,
        ),
        classifier: StaticClassifierOutput(
          probabilities: [
            confidence,
            1 - confidence,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
          ],
          knownMass: 1,
        ),
      );

  test(
    'display follows selected owner rather than highest confidence peer',
    () {
      for (final pending in [false, true]) {
        final result = GestureResult.fromStatic(
          frame: StaticGestureFrameResult(
            timestampMs: 100,
            hands: [
              hand(StaticPhysicalHand.right, .99),
              hand(StaticPhysicalHand.left, .90),
            ],
            command: null,
            selectedHand: StaticPhysicalHand.left,
            switchPending: pending,
            runtimeMs: 0,
            queueWaitMs: 0,
            replacedFrames: 0,
          ),
          detections: [],
          frameWidth: 640,
          frameHeight: 480,
          totalLatencyMs: 0,
        );
        expect(result.label, 'left');
        expect(result.handedness, 'left');
        expect(result.commandEmitted, isFalse);
      }
    },
  );

  test(
    'unowned pair cannot leak either valid gesture into the public label',
    () {
      final result = GestureResult.fromStatic(
        frame: StaticGestureFrameResult(
          timestampMs: 100,
          hands: [
            hand(StaticPhysicalHand.right, .99),
            hand(StaticPhysicalHand.left, .90),
          ],
          command: null,
          runtimeMs: 0,
          queueWaitMs: 0,
          replacedFrames: 0,
        ),
        detections: [],
        frameWidth: 640,
        frameHeight: 480,
        totalLatencyMs: 0,
      );
      expect(result.label, 'no_gesture');
      expect(result.handedness, isNull);
      expect(result.predictions, isEmpty);
    },
  );

  test('parses an accepted native result', () {
    final result = GestureResult.fromMap({
      'status': 'ok',
      'label': 'open_palm',
      'rawLabel': 'open_palm',
      'action': 'Toggle Object Tracking',
      'reason': 'Diagnostic thresholds passed',
      'confidence': .94,
      'knownMass': .97,
      'margin': .81,
      'latencyMs': 42,
      'hands': 1,
      'angle': -91.2,
      'angleAllowed': true,
      'handedness': 'Right',
      'landmarks': List.generate(21, (index) => [index / 21, .5]),
      'probabilities': [
        {'index': 4, 'label': 'open_palm', 'value': .94, 'disabled': false},
      ],
    });

    expect(result.accepted, isTrue);
    expect(result.landmarks, hasLength(21));
    expect(result.predictions.single.label, 'open_palm');
    expect(displayGesture(result.label), 'Open Palm');
  });

  test('treats no-hand frames as rejected diagnostics', () {
    final result = GestureResult.fromMap({
      'status': 'no_hand',
      'label': 'no_gesture',
      'reason': 'Show one hand inside the guide',
      'hands': 0,
      'latencyMs': 18,
    });

    expect(result.accepted, isFalse);
    expect(result.hasHand, isFalse);
    expect(result.predictions, isEmpty);
  });
}
