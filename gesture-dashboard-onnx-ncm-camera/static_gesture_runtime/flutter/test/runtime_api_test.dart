import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:static_gesture_runtime/src/command_controller.dart';
import 'package:static_gesture_runtime/src/geometry_policy.dart';
import 'package:static_gesture_runtime/src/runtime_api.dart';

class _Classifier implements StaticGestureClassifier {
  bool ready = false;
  int calls = 0;
  StaticClassifierOutput output = _output();
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
    return output;
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
        label: staticGestureLabels[probabilities
            .indexOf(probabilities.reduce((a, b) => a > b ? a : b))],
        confidence: 1,
        margin: 1,
        minimumHoldMs: 0,
        reason: 'test geometry passed',
        accepted: true);

StaticGestureHand _hand(StaticPhysicalHand hand,
    {bool degenerate = false, double scale = 1, bool world = false}) {
  final x = hand == StaticPhysicalHand.left ? .10 : .65;
  final points = List<List<double>>.generate(
      21,
      (i) => degenerate
          ? <double>[.5, .5, 0]
          : <double>[x + i * .01 * scale, .3 + i * .005 * scale, 0]);
  return StaticGestureHand(
      hand: hand,
      handednessConfidence: 1,
      normalizedLandmarks: points,
      worldLandmarks:
          world ? List.generate(21, (i) => [i * .002, i * .001, 0.0]) : null);
}

StaticGestureFrame _frame(double time, List<StaticGestureHand> hands) =>
    StaticGestureFrame(
        timestampMs: time, frameWidth: 640, frameHeight: 480, hands: hands);

void main() {
  test(
      'all twelve single-hand commands preserve strong/ordinary confirmation and duplicate latches',
      () async {
    for (final side in StaticPhysicalHand.values) {
      for (final label in staticGestureActions.keys) {
        for (final strong in [false, true]) {
          final probabilities = List<double>.filled(13, 0);
          final index = staticGestureLabels.indexOf(label);
          probabilities[index] = strong ? 1 : .85;
          probabilities[(index + 1) % 13] = strong ? 0 : .15;
          final classifier = _Classifier()
            ..output = StaticClassifierOutput(
                probabilities: probabilities, knownMass: 1);
          final commands = <StaticGestureCommand>[];
          final runtime = StaticGestureRuntimeApi(
              classifier: classifier,
              geometryEvaluator: _geometry,
              onCommand: commands.add);
          await runtime.initialize();
          final first = (await runtime.processFrame(_frame(0, [_hand(side)])))!;
          expect(first.command != null, strong,
              reason: '$side $label strong=$strong');
          await runtime.processFrame(_frame(100, [_hand(side)]));
          await runtime.processFrame(_frame(200, [_hand(side)]));
          expect(commands.map((c) => c.label), [label]);
          expect(commands.single.hand, side);
          await runtime.close();
        }
      }
    }
  });

  test(
      'first owner survives second hand, reversed order and independent latches',
      () async {
    final classifier = _Classifier();
    final commands = <StaticGestureCommand>[];
    final runtime = StaticGestureRuntimeApi(
        classifier: classifier,
        geometryEvaluator: _geometry,
        onCommand: commands.add);
    await runtime.initialize();
    await runtime.processFrame(_frame(0, [_hand(StaticPhysicalHand.left)]));
    for (var time = 100; time <= 1000; time += 100) {
      final result = await runtime.processFrame(_frame(time.toDouble(),
          [_hand(StaticPhysicalHand.right), _hand(StaticPhysicalHand.left)]));
      expect(result!.hands.length, 2);
      expect(result.selectedHand, StaticPhysicalHand.left);
      expect(result.command, isNull);
      expect(result.switchPending, isFalse);
    }
    expect(commands.length, 1);
    expect(commands.single.hand, StaticPhysicalHand.left);
    await runtime.close();
  });

  for (final world in [false, true]) {
    test(
        'sustained closer challenger switches after confirmation (world=$world)',
        () async {
      final commands = <StaticGestureCommand>[];
      final runtime = StaticGestureRuntimeApi(
          classifier: _Classifier(),
          geometryEvaluator: _geometry,
          onCommand: commands.add);
      await runtime.initialize();
      await runtime.processFrame(
          _frame(0, [_hand(StaticPhysicalHand.left, world: world)]));
      for (final time in [100.0, 200.0, 300.0]) {
        final result = (await runtime.processFrame(_frame(time, [
          _hand(StaticPhysicalHand.right, scale: 1.8, world: world),
          _hand(StaticPhysicalHand.left, world: world)
        ])))!;
        expect(result.selectedHand, StaticPhysicalHand.left);
        expect(result.switchPending, isTrue);
        expect(result.command, isNull);
      }
      final handoff = (await runtime.processFrame(_frame(400, [
        _hand(StaticPhysicalHand.right, scale: 1.8, world: world),
        _hand(StaticPhysicalHand.left, world: world)
      ])))!;
      expect(handoff.selectedHand, StaticPhysicalHand.right);
      expect(handoff.selectionReason, 'hand_switched');
      expect(handoff.command, isNull,
          reason: 'a fresh observation must rebuild the new owner hold');
      for (final time in [500.0, 600.0, 700.0, 800.0]) {
        await runtime.processFrame(_frame(time, [
          _hand(StaticPhysicalHand.left, world: world),
          _hand(StaticPhysicalHand.right, scale: 1.8, world: world)
        ]));
      }
      expect(commands.map((c) => c.hand),
          [StaticPhysicalHand.left, StaticPhysicalHand.right]);
      expect(commands.last.timestampMs, 500);
      await runtime.close();
    });
  }

  test('brief distance bursts cannot transfer ownership', () async {
    final runtime = StaticGestureRuntimeApi(
        classifier: _Classifier(), geometryEvaluator: _geometry);
    await runtime.initialize();
    await runtime.processFrame(_frame(0, [_hand(StaticPhysicalHand.left)]));
    for (var step = 1; step <= 12; step++) {
      final result = (await runtime.processFrame(_frame(step * 100, [
        _hand(StaticPhysicalHand.right, scale: step % 3 == 0 ? 1 : 1.8),
        _hand(StaticPhysicalHand.left)
      ])))!;
      expect(result.selectedHand, StaticPhysicalHand.left);
      expect(result.command, isNull);
    }
    await runtime.close();
  });

  test('a confirmed handoff back to a previous owner rebuilds its latch once',
      () async {
    final commands = <StaticGestureCommand>[];
    final runtime = StaticGestureRuntimeApi(
        classifier: _Classifier(),
        geometryEvaluator: _geometry,
        onCommand: commands.add);
    await runtime.initialize();
    await runtime.processFrame(_frame(0, [_hand(StaticPhysicalHand.right)]));
    for (final time in [100.0, 200.0, 300.0, 400.0, 500.0]) {
      await runtime.processFrame(_frame(time, [
        _hand(StaticPhysicalHand.left, scale: 1.8),
        _hand(StaticPhysicalHand.right)
      ]));
    }
    for (var time = 600; time <= 1600; time += 100) {
      await runtime.processFrame(_frame(time.toDouble(), [
        _hand(StaticPhysicalHand.left),
        _hand(StaticPhysicalHand.right, scale: 1.8)
      ]));
    }
    expect(commands.map((c) => c.hand), [
      StaticPhysicalHand.right,
      StaticPhysicalHand.left,
      StaticPhysicalHand.right
    ]);
    expect(commands.map((c) => c.timestampMs).toSet().length, 3);
    await runtime.close();
  });

  test('only valid other gesture takes over invalid owner, even when farther',
      () async {
    var leftValid = true;
    final commands = <StaticGestureCommand>[];
    final runtime = StaticGestureRuntimeApi(
        classifier: _Classifier(),
        onCommand: commands.add,
        geometryEvaluator: (
                {required probabilities,
                required knownMass,
                required normalizedLandmarks,
                required physicalHand,
                required handednessConfidence,
                required width,
                required height,
                worldLandmarks,
                preferredDirection}) =>
            StaticGeometryDecision(
                label: physicalHand == 'left' && !leftValid
                    ? 'no_gesture'
                    : 'left',
                accepted: physicalHand != 'left' || leftValid,
                confidence: 1,
                margin: 1,
                minimumHoldMs: 0,
                reason: 'test validity'));
    await runtime.initialize();
    await runtime
        .processFrame(_frame(0, [_hand(StaticPhysicalHand.left, scale: 1.8)]));
    leftValid = false;
    final first = (await runtime.processFrame(_frame(100, [
      _hand(StaticPhysicalHand.left, scale: 1.8),
      _hand(StaticPhysicalHand.right)
    ])))!;
    expect(first.selectedHand, StaticPhysicalHand.left);
    expect(first.switchPending, isTrue);
    leftValid = true;
    final recovered = (await runtime.processFrame(_frame(200, [
      _hand(StaticPhysicalHand.left, scale: 1.8),
      _hand(StaticPhysicalHand.right)
    ])))!;
    expect(recovered.selectedHand, StaticPhysicalHand.left);
    expect(recovered.switchPending, isFalse);
    leftValid = false;
    for (final time in [300.0, 400.0, 500.0, 600.0, 700.0, 800.0]) {
      await runtime.processFrame(_frame(time, [
        _hand(StaticPhysicalHand.right),
        _hand(StaticPhysicalHand.left, scale: 1.8)
      ]));
    }
    expect(commands.map((c) => c.hand),
        [StaticPhysicalHand.left, StaticPhysicalHand.right]);
    expect(commands.last.timestampMs, 700);
    await runtime.close();
  });

  test('missing owner gets a grace period then the other hand confirms afresh',
      () async {
    final commands = <StaticGestureCommand>[];
    final runtime = StaticGestureRuntimeApi(
        classifier: _Classifier(),
        geometryEvaluator: _geometry,
        onCommand: commands.add);
    await runtime.initialize();
    await runtime.processFrame(_frame(0, [_hand(StaticPhysicalHand.left)]));
    for (final time in [100.0, 200.0, 300.0]) {
      final result = (await runtime
          .processFrame(_frame(time, [_hand(StaticPhysicalHand.right)])))!;
      expect(result.selectedHand, isNull);
      expect(result.selectionReason, 'active_hand_missing');
      expect(result.command, isNull);
    }
    for (final time in [400.0, 500.0, 600.0, 700.0, 800.0]) {
      await runtime
          .processFrame(_frame(time, [_hand(StaticPhysicalHand.right)]));
    }
    expect(commands.map((c) => c.hand),
        [StaticPhysicalHand.left, StaticPhysicalHand.right]);
    expect(commands.last.timestampMs, 800);
    runtime.reset();
    expect(
        (await runtime
                .processFrame(_frame(0, [_hand(StaticPhysicalHand.right)])))!
            .command,
        isNotNull);
    await runtime.close();
  });

  test('two simultaneous equal hands wait rather than choose by order',
      () async {
    final runtime = StaticGestureRuntimeApi(
        classifier: _Classifier(), geometryEvaluator: _geometry);
    await runtime.initialize();
    for (final time in [0.0, 100.0, 200.0, 300.0, 400.0]) {
      final result = (await runtime.processFrame(_frame(time,
          [_hand(StaticPhysicalHand.right), _hand(StaticPhysicalHand.left)])))!;
      expect(result.command, isNull);
      expect(result.selectedHand, isNull);
    }
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
