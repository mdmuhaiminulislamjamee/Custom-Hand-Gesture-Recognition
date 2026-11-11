import 'package:flutter_test/flutter_test.dart';
import 'package:gesture_model_tester/services/phone_frame_gate.dart';

void main() {
  test('busy callbacks are dropped and slow inference has no extra delay', () {
    var now = 0;
    final gate = PhoneFrameGate(nowMicroseconds: () => now);
    final first = gate.begin();
    expect(first, isNotNull);
    now = 150000;
    expect(gate.begin(), isNull);
    gate.finish();
    expect(gate.begin(), isNotNull);
  });

  test('fast inference keeps the previous 10 FPS start interval', () {
    var now = 0;
    final gate = PhoneFrameGate(nowMicroseconds: () => now);
    expect(gate.begin(), isNotNull);
    gate.finish();
    now = 99999;
    expect(gate.begin(), isNull);
    now = 100000;
    expect(gate.begin(), isNotNull);
  });

  test('pause invalidates late results without overlapping detector calls', () {
    var now = 0;
    final gate = PhoneFrameGate(nowMicroseconds: () => now);
    final old = gate.begin()!;
    gate.invalidate();
    expect(gate.isCurrent(old), isFalse);
    now = 200000;
    expect(gate.begin(), isNull);
    gate.finish();
    final fresh = gate.begin()!;
    expect(gate.isCurrent(fresh), isTrue);
  });
}
