import 'dart:async';
import 'package:flutter_test/flutter_test.dart';
import 'package:static_gesture_runtime/src/frame_scheduler.dart';

void main() {
  test('only the freshest pending frame runs while one frame is in flight',
      () async {
    final started = Completer<void>();
    final finish = Completer<int>();
    final seen = <int>[];
    final scheduler = LatestFrameScheduler<int, int>(process: (value) async {
      seen.add(value);
      if (value == 1) {
        started.complete();
        return finish.future;
      }
      return value;
    });
    final first = scheduler.submit(1);
    await started.future;
    final second = scheduler.submit(2);
    final third = scheduler.submit(3);
    expect(await second, isNull);
    expect(scheduler.pendingCount, 1);
    finish.complete(1);
    expect(await first, 1);
    expect(await third, 3);
    expect(seen, <int>[1, 3]);
    expect(scheduler.replacedFrames, 1);
    await scheduler.close();
  });

  test('reset discards pending and in-flight results', () async {
    final started = Completer<void>();
    final finish = Completer<int>();
    final scheduler = LatestFrameScheduler<int, int>(process: (_) {
      started.complete();
      return finish.future;
    });
    final first = scheduler.submit(1);
    await started.future;
    final pending = scheduler.submit(2);
    scheduler.reset();
    expect(await pending, isNull);
    finish.complete(1);
    expect(await first, isNull);
    await scheduler.close();
  });

  test('an inference error does not poison subsequent frames', () async {
    final scheduler = LatestFrameScheduler<int, int>(process: (value) async {
      if (value == 1) throw StateError('native failure');
      return value;
    });
    await expectLater(scheduler.submit(1), throwsStateError);
    expect(await scheduler.submit(2), 2);
    await scheduler.close();
  });
}
