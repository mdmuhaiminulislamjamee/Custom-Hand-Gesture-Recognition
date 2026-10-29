import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/hand_detection.dart';
import 'package:static_gesture_runtime/src/detector_adapter.dart';
import 'package:static_gesture_runtime/src/runtime_api.dart';

Hand detectedHand(Handedness raw, double center,
    {double confidence = .99,
    bool world = true,
    bool reverse = false,
    double? rotation,
    double? cropSize,
    bool invalidWorld = false,
    bool duplicateWorld = false}) {
  final points = [
    for (var i = 0; i < 21; i++)
      HandLandmark(
          type: HandLandmarkType.values[i],
          x: center + (i % 5) * 5,
          y: 100 + i * 5,
          z: i.toDouble(),
          visibility: 1)
  ];
  return Hand(
      boundingBox: BoundingBox.ltrb(center, 100, center + 50, 250),
      score: .99,
      landmarks: reverse ? points.reversed.toList() : points,
      worldLandmarks: world
          ? [
              for (var i = 0; i < 21; i++)
                HandLandmark(
                    type: HandLandmarkType
                        .values[duplicateWorld && i == 1 ? 0 : i],
                    x: invalidWorld && i == 1 ? double.nan : i * .01,
                    y: i * .02,
                    z: i * .03,
                    visibility: 1)
            ]
          : [],
      imageWidth: 640,
      imageHeight: 480,
      handedness: raw,
      handednessScore: confidence,
      rotation: rotation,
      rotatedSize: cropSize);
}

void main() {
  test('crop worldXY rotate to source axes while metric z stays unchanged', () {
    final result = StaticDetectorAdapter()
        .frame(timestampMs: 100, width: 640, height: 480, detections: [
      detectedHand(Handedness.left, 200, rotation: math.pi / 2, cropSize: 448)
    ]);
    final hand = result.hands.single;
    expect(hand.worldLandmarks![8][0], closeTo(-.16, 1e-12));
    expect(hand.worldLandmarks![8][1], closeTo(.08, 1e-12));
    expect(hand.worldLandmarks![8][2], .24);
    expect(hand.normalizedLandmarks[8][2], 16 / 640);
  });

  test('mirrored input is normalized to unmirrored image and world axes', () {
    final hand = StaticDetectorAdapter()
        .frame(
            timestampMs: 100,
            width: 640,
            height: 480,
            modelInputMirrored: true,
            detections: [
              detectedHand(Handedness.right, 200,
                  rotation: math.pi / 2, cropSize: 448)
            ])
        .hands
        .single;
    expect(hand.hand, StaticPhysicalHand.right);
    expect(hand.normalizedLandmarks[8][0], closeTo(1 - 215 / 640, 1e-12));
    expect(hand.worldLandmarks![8][0], closeTo(.16, 1e-12));
    expect(hand.worldLandmarks![8][1], closeTo(.08, 1e-12));
    expect(hand.worldLandmarks![8][2], .24);
  });

  test('nonfinite scores skipped and malformed world cannot poison second hand',
      () {
    for (final confidence in [double.nan, double.infinity, 1.1]) {
      final frame = StaticDetectorAdapter()
          .frame(timestampMs: 100, width: 640, height: 480, detections: [
        detectedHand(Handedness.left, 100, confidence: confidence),
        detectedHand(Handedness.right, 400)
      ]);
      expect(frame.hands.length, 1);
      expect(frame.hands.single.hand, StaticPhysicalHand.left);
    }
    for (final duplicate in [false, true]) {
      final frame = StaticDetectorAdapter()
          .frame(timestampMs: 100, width: 640, height: 480, detections: [
        detectedHand(Handedness.left, 100,
            invalidWorld: !duplicate, duplicateWorld: duplicate),
        detectedHand(Handedness.right, 400)
      ]);
      expect(frame.hands.length, 2);
      expect(
          frame.hands
              .singleWhere((h) => h.hand == StaticPhysicalHand.right)
              .worldLandmarks,
          null);
      expect(
          frame.hands
              .singleWhere((h) => h.hand == StaticPhysicalHand.left)
              .worldLandmarks,
          hasLength(21));
    }
  });

  test('tracked near-overlap abstains rather than exchanging histories', () {
    final adapter = StaticDetectorAdapter();
    adapter.frame(timestampMs: 100, width: 640, height: 480, detections: [
      detectedHand(Handedness.left, 100),
      detectedHand(Handedness.right, 200)
    ]);
    final frame = adapter.frame(
        timestampMs: 200,
        width: 640,
        height: 480,
        detections: [
          detectedHand(Handedness.left, 145),
          detectedHand(Handedness.right, 150)
        ]);
    expect(frame.hands, isEmpty);
  });
  test(
      'unmirrored raw category becomes physical side once and world stays metric',
      () {
    final result = StaticDetectorAdapter().frame(
        timestampMs: 100,
        width: 640,
        height: 480,
        detections: [detectedHand(Handedness.left, 200, reverse: true)]);
    final hand = result.hands.single;
    expect(hand.hand, StaticPhysicalHand.right);
    expect(hand.normalizedLandmarks[8], [215 / 640, 140 / 480, 8 / 640]);
    expect(hand.worldLandmarks![8], [.08, .16, .24]);
  });
  test('low confidence and duplicate side detections abstain', () {
    final adapter = StaticDetectorAdapter();
    expect(
        adapter.frame(timestampMs: 100, width: 640, height: 480, detections: [
          detectedHand(Handedness.left, 100, confidence: .6)
        ]).hands,
        isEmpty);
    expect(
        adapter.frame(timestampMs: 200, width: 640, height: 480, detections: [
          detectedHand(Handedness.left, 100),
          detectedHand(Handedness.left, 400)
        ]).hands,
        isEmpty);
  });
  test(
      'spatially continuous two hands survive transient swapped raw categories',
      () {
    final adapter = StaticDetectorAdapter();
    adapter.frame(timestampMs: 100, width: 640, height: 480, detections: [
      detectedHand(Handedness.left, 100),
      detectedHand(Handedness.right, 400)
    ]);
    final result = adapter.frame(
        timestampMs: 200,
        width: 640,
        height: 480,
        detections: [
          detectedHand(Handedness.left, 400),
          detectedHand(Handedness.right, 100)
        ]);
    expect(
        result.hands
            .singleWhere((h) => h.hand == StaticPhysicalHand.right)
            .normalizedLandmarks
            .first
            .first,
        100 / 640);
    expect(
        result.hands
            .singleWhere((h) => h.hand == StaticPhysicalHand.left)
            .normalizedLandmarks
            .first
            .first,
        400 / 640);
  });
  test('missing world coordinates stay absent and stale tracks expire', () {
    final adapter = StaticDetectorAdapter();
    adapter.frame(timestampMs: 100, width: 640, height: 480, detections: [
      detectedHand(Handedness.left, 100),
      detectedHand(Handedness.right, 400)
    ]);
    final result =
        adapter.frame(timestampMs: 500, width: 640, height: 480, detections: [
      detectedHand(Handedness.left, 400, world: false),
      detectedHand(Handedness.right, 100, world: false)
    ]);
    expect(
        result.hands
            .singleWhere((h) => h.hand == StaticPhysicalHand.right)
            .normalizedLandmarks
            .first
            .first,
        400 / 640);
    expect(result.hands.every((h) => h.worldLandmarks == null), isTrue);
  });
}
