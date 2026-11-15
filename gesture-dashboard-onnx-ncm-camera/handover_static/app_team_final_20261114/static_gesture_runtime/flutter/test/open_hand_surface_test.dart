import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/hand_detection.dart';
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';

// Upright physical right hand, back facing the unmirrored camera: thumb and
// index on image-left. Turning the palm toward the camera reverses that order.
List<List<double>> openHand({required bool right, required bool palm}) {
  final points = <List<double>>[
    [.50, .90],
    [.42, .81],
    [.36, .72],
    [.32, .63],
    [.30, .54],
    for (final x in [.445, .48, .515, .55]) ...<List<double>>[
      [x, .72],
      [x, .60],
      [x, .48],
      [x, .36]
    ],
  ];
  if (right == palm) {
    for (final p in points) {
      p[0] = 1 - p[0];
    }
  }
  return points;
}

void main() {
  for (final right in [false, true]) {
    for (final palm in [false, true]) {
      for (final mirrored in [false, true]) {
        test(
            '${right ? 'right' : 'left'} ${palm ? 'palm' : 'back'} '
            'keeps its surface with detector mirror=$mirrored', () {
          final source = openHand(right: right, palm: palm);
          final rawRight = mirrored ? right : !right;
          final detection = Hand(
            boundingBox: BoundingBox.ltrb(0, 0, 1000, 1000),
            score: .99,
            landmarks: [
              for (var i = 0; i < 21; i++)
                HandLandmark(
                    type: HandLandmarkType.values[i],
                    x: (mirrored ? 1 - source[i][0] : source[i][0]) * 1000,
                    y: source[i][1] * 1000,
                    z: 0,
                    visibility: 1),
            ],
            worldLandmarks: const [],
            imageWidth: 1000,
            imageHeight: 1000,
            handedness: rawRight ? Handedness.right : Handedness.left,
            handednessScore: .99,
          );
          final frame = StaticDetectorAdapter().frame(
            timestampMs: 100,
            width: 1000,
            height: 1000,
            detections: [detection],
            modelInputMirrored: mirrored,
          );
          final hand = frame.hands.single;
          expect(hand.hand,
              right ? StaticPhysicalHand.right : StaticPhysicalHand.left);
          for (final raw in ['open_palm', 'stop_inverted']) {
            final row = List<double>.filled(13, 0);
            row[raw == 'open_palm' ? 4 : 12] = 1;
            final result = StaticGeometryPolicy.evaluate(
              probabilities: row,
              knownMass: 1,
              normalizedLandmarks: hand.normalizedLandmarks,
              physicalHand: hand.hand.name,
              handednessConfidence: hand.handednessConfidence,
              width: frame.frameWidth,
              height: frame.frameHeight,
            );
            expect(result.accepted, isTrue, reason: result.reason);
            expect(result.label, palm ? 'open_palm' : 'stop_inverted');
            final shape =
                result.diagnostics['hand_shape'] as Map<String, Object?>;
            expect(shape['hand_surface'], palm ? 'palmar' : 'dorsal');
          }
        });
      }
    }
  }
}
