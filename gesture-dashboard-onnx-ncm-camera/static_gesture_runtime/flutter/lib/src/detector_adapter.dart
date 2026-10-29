import 'dart:math' as math;

import 'package:hand_detection/hand_detection.dart';

import '../detector/camera_frame_adapter.dart';
import 'runtime_api.dart';

class _TrackedHand {
  _TrackedHand(this.side, this.x, this.y, this.scale, this.lastSeenMs);
  final StaticPhysicalHand side;
  double x, y, scale, lastSeenMs;
}

/// Adapts the exact bundled detector's pixel XYZ into the static contract.
/// NCM inputs are upright and unmirrored. Explicit mirrored detector input is
/// normalized to unmirrored image/world X here. The bundled detector returns
/// metric world landmarks in the rotated hand crop; rotate them into source
/// image axes without scaling or translating their metric units.
class StaticDetectorAdapter {
  final Map<StaticPhysicalHand, _TrackedHand> _tracks = {};
  void reset() => _tracks.clear();

  StaticGestureFrame frame(
      {required double timestampMs,
      required int width,
      required int height,
      required List<Hand> detections,
      bool modelInputMirrored = false}) {
    if (width <= 0 || height <= 0)
      throw ArgumentError('Invalid frame dimensions');
    _tracks.removeWhere((_, track) => timestampMs - track.lastSeenMs >= 350);
    final candidates = <({
      Hand hand,
      StaticPhysicalHand side,
      double x,
      double y,
      double scale
    })>[];
    for (final hand in detections.take(2)) {
      if (hand.imageWidth != width ||
          hand.imageHeight != height ||
          hand.landmarks.length != 21 ||
          hand.landmarks.map((p) => p.type.index).toSet().length != 21)
        continue;
      final physicalRight = physicalIsRightHand(
          modelInputMirrored: modelInputMirrored,
          modelHandedness: hand.handedness);
      final confidence = hand.handednessScore ?? 0;
      if (physicalRight == null ||
          !confidence.isFinite ||
          confidence < .75 ||
          confidence > 1 ||
          (hand.rotation != null && !hand.rotation!.isFinite) ||
          (hand.rotatedSize != null &&
              (!hand.rotatedSize!.isFinite || hand.rotatedSize! <= 0)))
        continue;
      final points = [...hand.landmarks]
        ..sort((a, b) => a.type.index.compareTo(b.type.index));
      if (points.any((p) => !p.x.isFinite || !p.y.isFinite || !p.z.isFinite))
        continue;
      final palm = [0, 5, 9, 13, 17];
      final detectorX =
          palm.fold<double>(0, (v, i) => v + points[i].x) / 5 / width;
      final x = modelInputMirrored ? 1 - detectorX : detectorX;
      final y = palm.fold<double>(0, (v, i) => v + points[i].y) / 5 / height;
      final scale = math.max(
          1 / width,
          math.sqrt(math.pow((points[9].x - points[0].x) / width, 2) +
              math.pow((points[9].y - points[0].y) / height, 2)));
      candidates.add((
        hand: hand,
        side:
            physicalRight ? StaticPhysicalHand.right : StaticPhysicalHand.left,
        x: x,
        y: y,
        scale: scale
      ));
    }
    final assigned = <StaticPhysicalHand, int>{};
    // Two persistent spatially separated hands may survive a transient model
    // handedness flip. A lone opposite hand cannot inherit an absent owner.
    if (candidates.length == 2 && _tracks.length == 2) {
      final sides = StaticPhysicalHand.values;
      double cost(StaticPhysicalHand side, int index) {
        final t = _tracks[side]!, c = candidates[index];
        final ratio = c.scale / t.scale;
        if (ratio < .45 || ratio > 2.2) return double.infinity;
        return math.sqrt(math.pow(c.x - t.x, 2) + math.pow(c.y - t.y, 2)) /
            math.max(c.scale, t.scale);
      }

      final direct = cost(sides[0], 0) + cost(sides[1], 1);
      final reverse = cost(sides[0], 1) + cost(sides[1], 0);
      final first = direct <= reverse ? 0 : 1;
      final separation = math.sqrt(
          math.pow(candidates[0].x - candidates[1].x, 2) +
              math.pow(candidates[0].y - candidates[1].y, 2));
      if (separation <=
          math.max(candidates[0].scale, candidates[1].scale) * .75) {
        return StaticGestureFrame(
            timestampMs: timestampMs,
            frameWidth: width.toDouble(),
            frameHeight: height.toDouble(),
            hands: []);
      }
      if (cost(sides[0], first) < 2 &&
          cost(sides[1], 1 - first) < 2 &&
          separation >
              math.max(candidates[0].scale, candidates[1].scale) * .75) {
        assigned[sides[0]] = first;
        assigned[sides[1]] = 1 - first;
      }
    }
    if (assigned.isEmpty) {
      for (final side in StaticPhysicalHand.values) {
        final matching = [
          for (var i = 0; i < candidates.length; i++)
            if (candidates[i].side == side) i
        ];
        // Duplicate labels are ambiguous; abstain instead of exchanging history.
        if (matching.length == 1) assigned[side] = matching.single;
      }
    }
    final hands = <StaticGestureHand>[];
    for (final entry in assigned.entries) {
      final c = candidates[entry.value], hand = c.hand;
      final points = [...hand.landmarks]
        ..sort((a, b) => a.type.index.compareTo(b.type.index));
      final world = [...hand.worldLandmarks]
        ..sort((a, b) => a.type.index.compareTo(b.type.index));
      final validWorld = world.length == 21 &&
          world.map((p) => p.type.index).toSet().length == 21 &&
          world.every((p) => p.x.isFinite && p.y.isFinite && p.z.isFinite);
      final rotation = hand.rotation ?? 0;
      final cosR = math.cos(rotation), sinR = math.sin(rotation);
      // Image z is still in the bundled model's 224px crop-input coordinates.
      final zScale = (hand.rotatedSize ?? 224) / 224;
      _tracks[entry.key] =
          _TrackedHand(entry.key, c.x, c.y, c.scale, timestampMs);
      hands.add(StaticGestureHand(
          hand: entry.key,
          normalizedLandmarks: [
            for (final p in points)
              [
                modelInputMirrored ? 1 - p.x / width : p.x / width,
                p.y / height,
                p.z * zScale / width
              ]
          ],
          worldLandmarks: validWorld
              ? [
                  for (final p in world)
                    [
                      (p.x * cosR - p.y * sinR) * (modelInputMirrored ? -1 : 1),
                      p.x * sinR + p.y * cosR,
                      p.z
                    ]
                ]
              : null,
          handednessConfidence: hand.handednessScore ?? 0));
    }
    return StaticGestureFrame(
        timestampMs: timestampMs,
        frameWidth: width.toDouble(),
        frameHeight: height.toDouble(),
        hands: hands);
  }
}
