import 'package:flutter_test/flutter_test.dart';
import 'package:static_gesture_runtime/src/command_controller.dart';

void main() {
  test('strong valid evidence emits immediately and repeats require neutral',
      () {
    final controller = StaticCommandController();
    StaticControllerResult feed(String label, double time,
            {double confidence = .96}) =>
        controller.process(
            label: label,
            confidence: confidence,
            margin: .5,
            timestampMs: time);
    expect(feed('left', 0).emit, isTrue);
    expect(feed('left', 100).emit, isFalse);
    expect(feed('no_gesture', 200, confidence: 0).confirmed, isFalse);
    expect(feed('left', 300).emit, isFalse);
    feed('no_gesture', 400, confidence: 0);
    expect(feed('no_gesture', 500, confidence: 0).confirmed, isTrue);
    expect(feed('left', 600).emit, isTrue);
  });

  test('ordinary evidence uses two latest-three matches', () {
    final c = StaticCommandController();
    StaticControllerResult feed(String label, double time) =>
        c.process(label: label, confidence: .8, margin: .15, timestampMs: time);
    expect(feed('left', 0).emit, isFalse);
    expect(feed('right', 100).emit, isFalse);
    expect(feed('left', 200).emit, isTrue);
  });

  test('missing hand clears evidence but cannot rearm a held toggle', () {
    final c = StaticCommandController();
    c.process(label: 'open_palm', confidence: .99, margin: .9, timestampMs: 0);
    c.handLost();
    expect(
        c
            .process(
                label: 'open_palm',
                confidence: .99,
                margin: .9,
                timestampMs: 1000)
            .emit,
        isFalse);
    c.reset();
    expect(
        c
            .process(
                label: 'open_palm', confidence: .99, margin: .9, timestampMs: 0)
            .emit,
        isTrue);
  });

  test('low mass recovery never skips its required stable hold', () {
    final c = StaticCommandController();
    StaticControllerResult feed(double time) => c.process(
        label: 'left',
        confidence: .99,
        margin: .9,
        timestampMs: time,
        minimumHoldMs: 500,
        allowStrongShortcut: false);
    expect(feed(0).emit, isFalse);
    expect(feed(100).emit, isFalse);
    expect(feed(499).emit, isFalse);
    expect(feed(500).emit, isTrue);
  });

  test('a cross-hand loser is not silently marked as emitted', () {
    final c = StaticCommandController();
    final candidate = c.process(
        label: 'ok',
        confidence: .99,
        margin: .9,
        timestampMs: 0,
        dispatch: false);
    expect(candidate.commandReady, isTrue);
    expect(c.lastEmitted, isNull);
    c.markEmitted('ok');
    expect(
        c
            .process(label: 'ok', confidence: .99, margin: .9, timestampMs: 100)
            .commandReady,
        isFalse);
  });

  test('timestamps and disabled command labels are rejected', () {
    final c = StaticCommandController();
    c.process(label: 'left', confidence: .8, margin: .2, timestampMs: 100);
    expect(
        () => c.process(
            label: 'left', confidence: .8, margin: .2, timestampMs: 100),
        throwsArgumentError);
    expect(
        () => c.process(
            label: 'dorsal', confidence: .99, margin: .9, timestampMs: 200),
        throwsArgumentError);
  });
}
