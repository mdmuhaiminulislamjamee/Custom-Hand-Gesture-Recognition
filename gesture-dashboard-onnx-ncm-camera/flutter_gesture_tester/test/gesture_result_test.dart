import 'package:flutter_test/flutter_test.dart';
import 'package:gesture_model_tester/models/gesture_result.dart';

void main() {
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
