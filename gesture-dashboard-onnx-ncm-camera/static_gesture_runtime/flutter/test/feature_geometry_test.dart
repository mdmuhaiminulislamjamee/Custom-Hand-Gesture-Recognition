import 'package:flutter_test/flutter_test.dart';
import '../tool/check_geometry_parity.dart';
import '../lib/src/feature_extractor.dart';
import '../lib/src/geometry_policy.dart';

void main() {
  test('76 features and static geometry match the Python reference cases', () {
    final result = checkGeometryParity('../validation/python_reference.json');
    expect(result['passed'], isTrue,
        reason: '${Map.of(result)..remove('failures')}');
  });

  test('invalid inputs cannot qualify as a command', () {
    final result = StaticGeometryPolicy.evaluate(
        probabilities: List.filled(13, double.nan),
        knownMass: 1,
        normalizedLandmarks: const [],
        physicalHand: 'left',
        handednessConfidence: 1,
        width: 640,
        height: 480);
    expect(result.accepted, isFalse);
    expect(result.label, 'no_gesture');
    expect(
        () => StaticFeatureExtractor.extract(const []), throwsFormatException);
  });
}
