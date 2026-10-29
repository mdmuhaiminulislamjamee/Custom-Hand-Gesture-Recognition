import 'dart:typed_data';

import 'command_controller.dart';
import 'feature_extractor.dart';
import 'frame_scheduler.dart';
import 'geometry_policy.dart';

enum StaticPhysicalHand { left, right }

typedef StaticGeometryEvaluator = StaticGeometryDecision Function({
  required List<double> probabilities,
  required double knownMass,
  required List<List<double>> normalizedLandmarks,
  required String physicalHand,
  required double handednessConfidence,
  required double width,
  required double height,
  List<List<double>>? worldLandmarks,
  String? preferredDirection,
});

class StaticGestureHand {
  StaticGestureHand(
      {required this.hand,
      required List<List<double>> normalizedLandmarks,
      required this.handednessConfidence,
      List<List<double>>? worldLandmarks})
      : normalizedLandmarks = _copyPoints(normalizedLandmarks),
        worldLandmarks =
            worldLandmarks == null ? null : _copyPoints(worldLandmarks);

  final StaticPhysicalHand hand;
  final List<List<double>> normalizedLandmarks;
  final List<List<double>>? worldLandmarks;
  final double handednessConfidence;
}

class StaticGestureFrame {
  StaticGestureFrame(
      {required this.timestampMs,
      required this.frameWidth,
      required this.frameHeight,
      required List<StaticGestureHand> hands})
      : hands = List<StaticGestureHand>.unmodifiable(hands);
  final double timestampMs;
  final double frameWidth;
  final double frameHeight;
  final List<StaticGestureHand> hands;
}

class StaticClassifierOutput {
  StaticClassifierOutput(
      {required List<double> probabilities,
      required this.knownMass,
      this.inferenceMs = 0})
      : probabilities = List<double>.unmodifiable(probabilities);
  final List<double> probabilities;
  final double knownMass;
  final double inferenceMs;
}

abstract interface class StaticGestureClassifier {
  bool get isInitialized;
  Future<void> initialize();
  Future<StaticClassifierOutput> predict(Float32List features);
  Future<void> close();
}

class StaticGestureCommand {
  const StaticGestureCommand(
      {required this.label,
      required this.action,
      required this.hand,
      required this.timestampMs,
      required this.confidence,
      required this.margin,
      required this.knownMass});
  final String label;
  final String action;
  final StaticPhysicalHand hand;
  final double timestampMs;
  final double confidence;
  final double margin;
  final double knownMass;
}

class StaticHandResult {
  const StaticHandResult(
      {required this.hand,
      required this.geometry,
      required this.controller,
      required this.classifier});
  final StaticPhysicalHand hand;
  final StaticGeometryDecision geometry;
  final StaticControllerResult controller;
  final StaticClassifierOutput classifier;
}

class StaticGestureFrameResult {
  StaticGestureFrameResult(
      {required this.timestampMs,
      required List<StaticHandResult> hands,
      required this.command,
      required this.runtimeMs,
      required this.queueWaitMs,
      required this.replacedFrames})
      : hands = List<StaticHandResult>.unmodifiable(hands);
  final double timestampMs;
  final List<StaticHandResult> hands;
  final StaticGestureCommand? command;
  final double runtimeMs;
  final double queueWaitMs;
  final int replacedFrames;
}

/// Landmarks-in runtime. The detector/camera adapter owns camera normalization;
/// all hands must be physical identity in upright, unmirrored source space.
class StaticGestureRuntimeApi {
  StaticGestureRuntimeApi(
      {required this.classifier,
      this.onCommand,
      this.onResult,
      StaticGeometryEvaluator geometryEvaluator = StaticGeometryPolicy.evaluate,
      StaticConfirmationPolicy policy = const StaticConfirmationPolicy()})
      : _policy = policy,
        _geometry = geometryEvaluator {
    policy.validate();
    _states = {
      for (final hand in StaticPhysicalHand.values)
        hand: _HandState(StaticCommandController(policy: policy))
    };
    _scheduler =
        LatestFrameScheduler<_SubmittedFrame, StaticGestureFrameResult>(
            process: _process);
  }

  final StaticGestureClassifier classifier;
  final void Function(StaticGestureCommand event)? onCommand;
  final void Function(StaticGestureFrameResult result)? onResult;
  final StaticConfirmationPolicy _policy;
  final StaticGeometryEvaluator _geometry;
  late final Map<StaticPhysicalHand, _HandState> _states;
  late final LatestFrameScheduler<_SubmittedFrame, StaticGestureFrameResult>
      _scheduler;
  final Stopwatch _clock = Stopwatch()..start();
  bool _closed = false;
  int _generation = 0;
  double? _lastTimestampMs;
  Future<void>? _initializing;

  bool get isInitialized => !_closed && classifier.isInitialized;
  int get pendingFrames => _scheduler.pendingCount;
  int get replacedFrames => _scheduler.replacedFrames;

  Future<void> initialize() {
    if (_closed) throw StateError('Static gesture runtime is closed.');
    return _initializing ??= classifier.initialize();
  }

  /// Returns null if this frame was replaced or invalidated by reset/close.
  /// Calling once for every detector result, including empty results, cancels
  /// stale native results immediately when their hand disappears.
  Future<StaticGestureFrameResult?> processFrame(StaticGestureFrame frame) {
    if (!isInitialized)
      throw StateError('Initialize the static runtime first.');
    _validateFrame(frame);
    _lastTimestampMs = frame.timestampMs;
    final present = frame.hands.map((hand) => hand.hand).toSet();
    for (final entry in _states.entries) {
      final state = entry.value;
      if (present.contains(entry.key)) {
        if (state.lastSeenMs != null &&
            frame.timestampMs - state.lastSeenMs! >=
                _policy.missingHandResetMs) {
          state.controller.handLost();
          state.preferredDirection = null;
          state.version++;
        }
        state.present = true;
        state.lastSeenMs = frame.timestampMs;
      } else {
        if (state.present) state.version++;
        state.present = false;
        if (state.lastSeenMs != null &&
            frame.timestampMs - state.lastSeenMs! >=
                _policy.missingHandResetMs) {
          state.controller.handLost();
          state.preferredDirection = null;
        }
      }
    }
    return _scheduler.submit(_SubmittedFrame(
        frame,
        _generation,
        _clock.elapsedMicroseconds,
        {for (final entry in _states.entries) entry.key: entry.value.version}));
  }

  void _validateFrame(StaticGestureFrame frame) {
    if (!frame.timestampMs.isFinite ||
        frame.timestampMs < 0 ||
        (_lastTimestampMs != null && frame.timestampMs <= _lastTimestampMs!) ||
        !frame.frameWidth.isFinite ||
        frame.frameWidth <= 0 ||
        !frame.frameHeight.isFinite ||
        frame.frameHeight <= 0 ||
        frame.hands.length > 2 ||
        frame.hands.map((hand) => hand.hand).toSet().length !=
            frame.hands.length) {
      throw ArgumentError(
          'Invalid frame or non-increasing timestamp; reset after stream changes.');
    }
    for (final hand in frame.hands) {
      if (!hand.handednessConfidence.isFinite ||
          hand.handednessConfidence < 0 ||
          hand.handednessConfidence > 1 ||
          !_validPoints(hand.normalizedLandmarks) ||
          (hand.worldLandmarks != null &&
              !_validPoints(hand.worldLandmarks!))) {
        throw ArgumentError(
            'Expected 21 finite XYZ landmarks and valid handedness confidence.');
      }
    }
  }

  Future<StaticGestureFrameResult> _process(_SubmittedFrame submitted) async {
    final startedUs = _clock.elapsedMicroseconds;
    final frame = submitted.frame;
    final results = <StaticHandResult>[];
    StaticGestureCommand? command;
    for (final hand in frame.hands) {
      final state = _states[hand.hand]!;
      if (!_current(submitted, hand.hand)) continue;
      Float32List features;
      try {
        features = StaticFeatureExtractor.extract(hand.normalizedLandmarks);
      } on FormatException {
        state.controller.handLost();
        state.preferredDirection = null;
        continue;
      } on ArgumentError {
        state.controller.handLost();
        state.preferredDirection = null;
        continue;
      }
      final output = await classifier.predict(features);
      if (!_current(submitted, hand.hand)) continue;
      _validateOutput(output);
      final geometry = _geometry(
        probabilities: output.probabilities,
        knownMass: output.knownMass,
        normalizedLandmarks: hand.normalizedLandmarks,
        worldLandmarks: hand.worldLandmarks,
        physicalHand: hand.hand.name,
        handednessConfidence: hand.handednessConfidence,
        width: frame.frameWidth,
        height: frame.frameHeight,
        preferredDirection: state.preferredDirection,
      );
      final order = List<int>.generate(13, (i) => i)
        ..sort((a, b) =>
            output.probabilities[b].compareTo(output.probabilities[a]));
      final rawConfidence = output.probabilities[order[0]];
      final rawMargin = rawConfidence - output.probabilities[order[1]];
      var control = state.controller.process(
        label: geometry.accepted ? geometry.label : 'no_gesture',
        confidence: geometry.confidence,
        margin: geometry.margin,
        timestampMs: frame.timestampMs,
        minimumHoldMs: geometry.minimumHoldMs,
        allowStrongShortcut: output.knownMass >= 0.70 &&
            staticGestureLabels[order[0]] == geometry.label &&
            rawConfidence >= _policy.strongConfidence &&
            rawMargin >= _policy.strongMargin,
        dispatch: false,
      );
      if (geometry.accepted &&
          <String>{'left', 'right', 'up', 'down'}.contains(geometry.label)) {
        state.preferredDirection = geometry.label;
      } else {
        state.preferredDirection = null;
      }
      if (control.commandReady &&
          command == null &&
          _current(submitted, hand.hand)) {
        state.controller.markEmitted(control.label);
        control = StaticControllerResult(
          label: control.label,
          confirmed: true,
          emit: true,
          reason: 'confirmed command emitted',
          matches: control.matches,
          commandReady: true,
        );
        command = StaticGestureCommand(
            label: control.label,
            action: staticGestureActions[control.label]!,
            hand: hand.hand,
            timestampMs: frame.timestampMs,
            confidence: geometry.confidence,
            margin: geometry.margin,
            knownMass: output.knownMass);
        onCommand?.call(command);
      }
      results.add(StaticHandResult(
          hand: hand.hand,
          geometry: geometry,
          controller: control,
          classifier: output));
    }
    final result = StaticGestureFrameResult(
        timestampMs: frame.timestampMs,
        hands: results,
        command: command,
        runtimeMs: (_clock.elapsedMicroseconds - startedUs) / 1000,
        queueWaitMs: (startedUs - submitted.queuedUs) / 1000,
        replacedFrames: _scheduler.replacedFrames);
    if (!_closed && submitted.generation == _generation) onResult?.call(result);
    return result;
  }

  bool _current(_SubmittedFrame frame, StaticPhysicalHand hand) =>
      !_closed &&
      frame.generation == _generation &&
      _states[hand]!.present &&
      frame.versions[hand] == _states[hand]!.version;

  /// Call after reconnect, orientation/mirror changes, or timestamp reset.
  void reset() {
    _generation++;
    _lastTimestampMs = null;
    _scheduler.reset();
    for (final state in _states.values) {
      state.version++;
      state.present = false;
      state.lastSeenMs = null;
      state.preferredDirection = null;
      state.controller.reset();
    }
  }

  /// Cancels delayed results and clears temporal evidence after a bad detector
  /// frame, without treating an exception as a deliberate gesture release.
  void invalidateEvidence() {
    _generation++;
    _scheduler.reset();
    for (final state in _states.values) {
      state.version++;
      state.present = false;
      state.preferredDirection = null;
      state.controller.handLost();
    }
  }

  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    reset();
    await _scheduler.close();
    await _initializing;
    await classifier.close();
  }
}

class _HandState {
  _HandState(this.controller);
  final StaticCommandController controller;
  int version = 0;
  bool present = false;
  double? lastSeenMs;
  String? preferredDirection;
}

class _SubmittedFrame {
  const _SubmittedFrame(
      this.frame, this.generation, this.queuedUs, this.versions);
  final StaticGestureFrame frame;
  final int generation;
  final int queuedUs;
  final Map<StaticPhysicalHand, int> versions;
}

List<List<double>> _copyPoints(List<List<double>> points) =>
    List<List<double>>.unmodifiable(
        points.map((p) => List<double>.unmodifiable(p)));
bool _validPoints(List<List<double>> points) =>
    points.length == 21 &&
    points
        .every((point) => point.length == 3 && point.every((v) => v.isFinite));
void _validateOutput(StaticClassifierOutput output) {
  if (output.probabilities.length != 13 ||
      output.probabilities
          .any((value) => !value.isFinite || value < 0 || value > 1.00001) ||
      (output.probabilities.fold(0.0, (a, b) => a + b) - 1).abs() > 0.10 ||
      !output.knownMass.isFinite ||
      output.knownMass < 0 ||
      output.knownMass > 1.00001) {
    throw StateError('Invalid classifier output.');
  }
}
