import 'dart:ui';

import 'package:hand_detection/hand_detection.dart';
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';

class GesturePrediction {
  const GesturePrediction({
    required this.index,
    required this.label,
    required this.value,
    required this.disabled,
  });

  factory GesturePrediction.fromMap(Map<Object?, Object?> map) {
    return GesturePrediction(
      index: (map['index']! as num).toInt(),
      label: map['label']! as String,
      value: (map['value']! as num).toDouble(),
      disabled: map['disabled'] as bool? ?? false,
    );
  }

  final int index;
  final String label;
  final double value;
  final bool disabled;
}

class GestureResult {
  const GestureResult({
    required this.status,
    required this.label,
    required this.rawLabel,
    required this.action,
    required this.reason,
    required this.confidence,
    required this.knownMass,
    required this.margin,
    required this.latencyMs,
    required this.hands,
    required this.landmarks,
    required this.predictions,
    this.detections = const <Hand>[],
    this.frameWidth = 0,
    this.frameHeight = 0,
    this.commandEmitted = false,
    this.angle,
    this.angleAllowed,
    this.handedness,
  });

  factory GestureResult.fromMap(Map<Object?, Object?> map) {
    final rawPoints = map['landmarks'] as List<Object?>? ?? const [];
    final rawPredictions = map['probabilities'] as List<Object?>? ?? const [];
    return GestureResult(
      status: map['status'] as String? ?? 'error',
      label: map['label'] as String? ?? 'no_gesture',
      rawLabel: map['rawLabel'] as String? ?? 'no_gesture',
      action: map['action'] as String? ?? 'No action',
      reason: map['reason'] as String? ?? 'No diagnostic result',
      confidence: (map['confidence'] as num?)?.toDouble() ?? 0,
      knownMass: (map['knownMass'] as num?)?.toDouble() ?? 0,
      margin: (map['margin'] as num?)?.toDouble() ?? 0,
      latencyMs: (map['latencyMs'] as num?)?.toInt() ?? 0,
      hands: (map['hands'] as num?)?.toInt() ?? 0,
      angle: (map['angle'] as num?)?.toDouble(),
      angleAllowed: map['angleAllowed'] as bool?,
      handedness: map['handedness'] as String?,
      landmarks: rawPoints
          .map((point) {
            final pair = point! as List<Object?>;
            return Offset(
              (pair[0]! as num).toDouble(),
              (pair[1]! as num).toDouble(),
            );
          })
          .toList(growable: false),
      predictions: rawPredictions
          .map(
            (item) => GesturePrediction.fromMap(item! as Map<Object?, Object?>),
          )
          .toList(growable: false),
    );
  }

  factory GestureResult.fromStatic({
    required StaticGestureFrameResult frame,
    required List<Hand> detections,
    required int frameWidth,
    required int frameHeight,
    required double totalLatencyMs,
  }) {
    StaticHandResult? primary;
    final command = frame.command;
    if (command != null) {
      for (final hand in frame.hands) {
        if (hand.hand == command.hand) {
          primary = hand;
          break;
        }
      }
    }
    if (primary == null && frame.hands.isNotEmpty) {
      final ordered = [...frame.hands]
        ..sort((a, b) {
          if (a.geometry.accepted != b.geometry.accepted) {
            return a.geometry.accepted ? -1 : 1;
          }
          return b.geometry.confidence.compareTo(a.geometry.confidence);
        });
      primary = ordered.first;
    }

    final classifier = primary?.classifier;
    final probabilities = classifier?.probabilities ?? const <double>[];
    final order = List<int>.generate(probabilities.length, (index) => index)
      ..sort((a, b) => probabilities[b].compareTo(probabilities[a]));
    final rawIndex = order.isEmpty ? -1 : order.first;
    final rawLabel = rawIndex < 0
        ? 'no_gesture'
        : staticGestureLabels[rawIndex];
    final geometry = primary?.geometry;
    final candidate = geometry?.accepted == true ? geometry!.label : null;
    final label = command?.label ?? candidate ?? 'no_gesture';
    final reason = command != null
        ? 'Confirmed ${displayGesture(command.label)} command emitted'
        : detections.isEmpty
        ? 'No complete hand landmarks'
        : geometry?.reason ?? 'Hand detected; gesture not qualified';
    final first = detections.isEmpty ? null : detections.first;
    final sortedLandmarks = first == null
        ? const <HandLandmark>[]
        : ([...first.landmarks]
            ..sort((a, b) => a.type.index.compareTo(b.type.index)));

    return GestureResult(
      status: command != null
          ? 'ok'
          : detections.isEmpty
          ? 'no_hand'
          : candidate != null
          ? 'candidate'
          : 'no_gesture',
      label: label,
      rawLabel: rawLabel,
      action: command?.action ?? staticGestureActions[label] ?? 'No action',
      reason: reason,
      confidence:
          geometry?.confidence ?? (rawIndex < 0 ? 0 : probabilities[rawIndex]),
      knownMass: classifier?.knownMass ?? 0,
      margin:
          geometry?.margin ??
          (order.length < 2
              ? 0
              : probabilities[order[0]] - probabilities[order[1]]),
      latencyMs: totalLatencyMs.round(),
      hands: detections.length,
      handedness: primary?.hand.name,
      landmarks: sortedLandmarks
          .map(
            (point) => Offset(
              frameWidth > 0 ? point.x / frameWidth : 0,
              frameHeight > 0 ? point.y / frameHeight : 0,
            ),
          )
          .toList(growable: false),
      predictions: [
        for (final index in order)
          GesturePrediction(
            index: index,
            label: staticGestureLabels[index],
            value: probabilities[index],
            disabled: staticGestureLabels[index] == 'dorsal',
          ),
      ],
      detections: List<Hand>.unmodifiable(detections.take(2)),
      frameWidth: frameWidth,
      frameHeight: frameHeight,
      commandEmitted: command != null,
    );
  }

  final String status;
  final String label;
  final String rawLabel;
  final String action;
  final String reason;
  final double confidence;
  final double knownMass;
  final double margin;
  final int latencyMs;
  final int hands;
  final double? angle;
  final bool? angleAllowed;
  final String? handedness;
  final List<Offset> landmarks;
  final List<GesturePrediction> predictions;
  final List<Hand> detections;
  final int frameWidth;
  final int frameHeight;
  final bool commandEmitted;

  bool get accepted =>
      commandEmitted || (status == 'ok' && label != 'no_gesture');
  bool get hasHand => hands > 0;
}

String displayGesture(String value) {
  if (value == 'no_gesture') return 'No gesture';
  return value
      .split('_')
      .map((word) => '${word[0].toUpperCase()}${word.substring(1)}')
      .join(' ');
}
