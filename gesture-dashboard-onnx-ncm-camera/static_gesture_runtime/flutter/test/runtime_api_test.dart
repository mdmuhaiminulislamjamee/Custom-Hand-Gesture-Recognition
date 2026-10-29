import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:static_gesture_runtime/src/command_controller.dart';
import 'package:static_gesture_runtime/src/geometry_policy.dart';
import 'package:static_gesture_runtime/src/runtime_api.dart';

class _Classifier implements StaticGestureClassifier {
  bool ready = false;
  int calls = 0;
  Completer<StaticClassifierOutput>? blocked;
  Completer<void>? started;
  @override
  bool get isInitialized => ready;
  @override
  Future<void> initialize() async {
    ready = true;
  }

  @override
  Future<StaticClassifierOutput> predict(Float32List features) async {
    calls++;
    started?.complete();
    started = null;
    if (blocked != null) {
      final block = blocked!;
      blocked = null;
      return block.future;
    }
    return _output();
  }

  @override
  Future<void> close() async {
    ready = false;
  }
}

StaticClassifierOutput _output() => StaticClassifierOutput(
    probabilities: <double>[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    knownMass: 1);

StaticGeometryDecision _geometry(
        {required List<double> probabilities,
        required double knownMass,
        required List<List<double>> normalizedLandmarks,
        required String physicalHand,
        required double handednessConfidence,
        required double width,
        required double height,
        List<List<double>>? worldLandmarks,
        String? preferredDirection}) =>
    StaticGeometryDecision(
        label: 'left',
        confidence: 1,
        margin: 1,
        minimumHoldMs: 0,
        reason: 'test geometry passed',
        accepted: true);

StaticGestureHand _hand(StaticPhysicalHand hand, {bool degenerate = false}) =>
    StaticGestureHand(
        hand: hand,
        handednessConfidence: 1,
        normalizedLandmarks: List<List<double>>.generate(
            21,
            (i) => degenerate
                ? <double>[.5, .5, 0]
                : <double>[.2 + i * .01, .3 + i * .005, 0]));

StaticGestureFrame _frame(double time, List<StaticGestureHand> hands) =>
    StaticGestureFrame(
        timestampMs: time, frameWidth: 640, frameHeight: 480, hands: hands);

void main() {
  test('hands have independent latches and at most one command per timestamp',
      () async {
    final classifier = _Classifier();
    final commands = <StaticGestureCommand>[];
    final runtime = StaticGestureRuntimeApi(
        classifier: classifier,
        geometryEvaluator: _geometry,
        onCommand: commands.add);
    await runtime.initialize();
    final result = await runtime.processFrame(_frame(0, <StaticGestureHand>[
      _hand(StaticPhysicalHand.left),
      _hand(StaticPhysicalHand.right)
    ]));
    expect(result!.hands.length, 2);
    expect(result.hands.first.controller.emit, isTrue);
    expect(result.hands.last.controller.emit, isFalse);
    expect(commands.length, 1);
    expect(commands.single.hand, StaticPhysicalHand.left);
    await runtime.processFrame(
        _frame(100, <StaticGestureHand>[_hand(StaticPhysicalHand.right)]));
    expect(commands.length, 2);
    expect(commands.last.hand, StaticPhysicalHand.right);
    expect(commands.map((c) => c.timestampMs).toSet().length, 2);
    await runtime.close();
  });

  test('a missing hand invalidates an older native prediction immediately',
      () async {
    final classifier = _Classifier();
    final started = Completer<void>();
    final blocked = Completer<StaticClassifierOutput>();
    classifier.started = started;
    classifier.blocked = blocked;
    final commands = <StaticGestureCommand>[];
    final runtime = StaticGestureRuntimeApi(
        classifier: classifier,
        geometryEvaluator: _geometry,
        onCommand: commands.add);
    await runtime.initialize();
    final old = runtime.processFrame(
        _frame(0, <StaticGestureHand>[_hand(StaticPhysicalHand.left)]));
    await started.future;
    final missing = runtime.processFrame(_frame(100, <StaticGestureHand>[]));
    blocked.complete(_output());
    await old;
    await missing;
    expect(commands, isEmpty);
    await runtime.close();
  });

  test('a degenerate first hand cannot block the valid second hand', () async {
    final classifier = _Classifier();
    final runtime = StaticGestureRuntimeApi(
        classifier: classifier, geometryEvaluator: _geometry);
    await runtime.initialize();
    final result = await runtime.processFrame(_frame(0, <StaticGestureHand>[
      _hand(StaticPhysicalHand.left, degenerate: true),
      _hand(StaticPhysicalHand.right)
    ]));
    expect(classifier.calls, 1);
    expect(result!.command!.hand, StaticPhysicalHand.right);
    await runtime.close();
  });

  test('detector error and hand loss preserve duplicate suppression', () async {
    final classifier = _Classifier();
    final commands = <StaticGestureCommand>[];
    final runtime = StaticGestureRuntimeApi(
        classifier: classifier,
        geometryEvaluator: _geometry,
        onCommand: commands.add);
    await runtime.initialize();
    await runtime.processFrame(
        _frame(0, <StaticGestureHand>[_hand(StaticPhysicalHand.left)]));
    runtime.invalidateEvidence();
    await runtime.processFrame(
        _frame(100, <StaticGestureHand>[_hand(StaticPhysicalHand.left)]));
    await runtime.processFrame(_frame(500, <StaticGestureHand>[]));
    await runtime.processFrame(
        _frame(1000, <StaticGestureHand>[_hand(StaticPhysicalHand.left)]));
    expect(commands.length, 1);
    runtime.reset();
    await runtime.processFrame(
        _frame(0, <StaticGestureHand>[_hand(StaticPhysicalHand.left)]));
    expect(commands.length, 2);
    await runtime.close();
  });

  test('duplicate physical identities and backward timestamps are rejected',
      () async {
    final runtime = StaticGestureRuntimeApi(
        classifier: _Classifier(), geometryEvaluator: _geometry);
    await runtime.initialize();
    expect(
        () => runtime.processFrame(_frame(0, <StaticGestureHand>[
              _hand(StaticPhysicalHand.left),
              _hand(StaticPhysicalHand.left)
            ])),
        throwsArgumentError);
    await runtime.processFrame(_frame(100, <StaticGestureHand>[]));
    expect(() => runtime.processFrame(_frame(99, <StaticGestureHand>[])),
        throwsArgumentError);
    await runtime.close();
  });

  test('all twelve command mappings remain enabled and dorsal is absent', () {
    expect(staticGestureActions.length, 12);
    expect(staticGestureActions.containsKey('dorsal'), isFalse);
    expect(staticGestureActions['peace'], 'Turn 90 Degrees');
    expect(staticGestureActions['stop_inverted'], 'Return to main position');
  });
}
