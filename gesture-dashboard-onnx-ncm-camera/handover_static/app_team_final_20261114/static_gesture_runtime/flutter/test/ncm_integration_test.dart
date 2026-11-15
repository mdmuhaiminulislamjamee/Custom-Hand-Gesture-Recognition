import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/hand_detection.dart';
import 'package:static_gesture_runtime/ncm_static_gesture_runtime.dart';
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';

class _Classifier implements StaticGestureClassifier {
  bool ready = false;
  @override
  bool get isInitialized => ready;
  @override
  Future<void> initialize() async {
    ready = true;
  }

  @override
  Future<StaticClassifierOutput> predict(Float32List features) async =>
      StaticClassifierOutput(
          probabilities: [1, ...List<double>.filled(12, 0)], knownMass: 1);
  @override
  Future<void> close() async {
    ready = false;
  }
}

class _Detector implements NcmStaticDetector {
  final seen = <int>[];
  Completer<void>? initialization;
  Completer<List<Hand>>? next;
  bool closed = false;
  int resets = 0;
  @override
  String get backend => 'test detector';
  @override
  Future<void> initialize() async {
    await initialization?.future;
  }

  @override
  Future<List<Hand>> detect(Uint8List jpeg) async {
    seen.add(jpeg[6]);
    final pending = next;
    next = null;
    return pending == null ? [] : await pending.future;
  }

  @override
  Future<void> resetTracking() async {
    resets++;
  }

  @override
  Future<void> close() async {
    closed = true;
  }
}

Uint8List _jpeg(int tag) => Uint8List.fromList([
      255,
      216,
      255,
      224,
      0,
      3,
      tag,
      255,
      192,
      0,
      11,
      8,
      1,
      224,
      2,
      128,
      1,
      1,
      17,
      0,
      255,
      217
    ]);
Future<void> _until(bool Function() ready) async {
  final deadline = DateTime.now().add(const Duration(seconds: 3));
  while (!ready()) {
    if (DateTime.now().isAfter(deadline))
      fail('Timed out awaiting integration event');
    await Future<void>.delayed(const Duration(milliseconds: 5));
  }
}

void main() {
  test('failed detector from disconnected epoch cannot invalidate new stream',
      () async {
    final server = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
    final sockets = <Socket>[];
    final sub = server.listen(sockets.add);
    final detector = _Detector();
    final delayed = Completer<List<Hand>>();
    detector.next = delayed;
    final errors = <Object>[];
    final outputs = <int>[];
    final previews = <NcmStaticPreviewFrame>[];
    final engine = NcmStaticGestureRuntime(
        runtime: StaticGestureRuntimeApi(classifier: _Classifier()),
        detector: detector,
        host: '127.0.0.1',
        port: server.port,
        onError: errors.add,
        onPreview: previews.add,
        onFrame: (frame, result) => outputs.add(frame.sequence));
    addTearDown(() async {
      if (!delayed.isCompleted) delayed.complete([]);
      await engine.close();
      for (final socket in sockets) {
        socket.destroy();
      }
      await sub.cancel();
      await server.close();
    });
    await engine.start();
    await _until(() => sockets.isNotEmpty);
    sockets.first.add(JlipParser.packet(
        type: 16, sequence: 1, timestampMs: 0, payload: _jpeg(1)));
    await sockets.first.flush();
    await _until(() => detector.seen.length == 1);
    sockets.first.destroy();
    await _until(() => sockets.length == 2);
    errors.clear();
    sockets.last.add(JlipParser.packet(
        type: 16, sequence: 2, timestampMs: 0, payload: _jpeg(2)));
    await sockets.last.flush();
    delayed.completeError(StateError('old decode failure'));
    await _until(() => outputs.isNotEmpty);
    expect(outputs, [2]);
    expect(previews.single.source.sequence, 2);
    expect(previews.single.width, 640);
    expect(previews.single.height, 480);
    expect(previews.single.result.timestampMs,
        previews.single.source.receivedTimestampMs);
    expect(errors, isEmpty);
  });
  test(
      'JLIP client replies to heartbeat, reconnects and preserves clock metadata',
      () async {
    final server = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
    final sockets = <Socket>[];
    final serverSub = server.listen(sockets.add);
    final frames = <NcmJpegFrame>[];
    final states = <String>[];
    final client = NcmCameraClient(
        host: '127.0.0.1',
        port: server.port,
        onFrame: frames.add,
        onState: states.add,
        reconnectDelay: const Duration(milliseconds: 20));
    addTearDown(() async {
      await client.stop();
      for (final socket in sockets) {
        socket.destroy();
      }
      await serverSub.cancel();
      await server.close();
    });
    await client.start();
    await _until(() => sockets.isNotEmpty);
    final reply = Completer<JlipPacket>();
    final parser = JlipParser();
    sockets.first.listen((bytes) {
      final packets = parser.feed(bytes);
      if (packets.isNotEmpty && !reply.isCompleted)
        reply.complete(packets.first);
    });
    final bytes = JlipParser.packet(type: 3, sequence: 12, timestampMs: 99);
    sockets.first.add(bytes.sublist(0, 7));
    await sockets.first.flush();
    sockets.first.add([
      ...bytes.sublist(7),
      ...JlipParser.packet(
          type: 16, sequence: 13, timestampMs: 0xfffffff0, payload: _jpeg(1))
    ]);
    await sockets.first.flush();
    expect((await reply.future.timeout(const Duration(seconds: 3))).type, 3);
    await _until(() => frames.length == 1);
    expect(frames.single.sequence, 13);
    expect(frames.single.boardTimestampMs, 0xfffffff0);
    final firstTime = frames.single.receivedTimestampMs;
    sockets.first.destroy();
    await _until(() => sockets.length == 2);
    sockets.last.add(JlipParser.packet(
        type: 16, sequence: 1, timestampMs: 10, payload: _jpeg(2)));
    await sockets.last.flush();
    await _until(() => frames.length == 2);
    expect(frames.last.receivedTimestampMs, greaterThan(firstTime));
    expect(states, contains('reconnecting'));
  });

  test(
      'slow detector keeps newest pending frame and drops stopped epoch results',
      () async {
    final server = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
    final sockets = <Socket>[];
    final sub = server.listen(sockets.add);
    final detector = _Detector();
    final first = Completer<List<Hand>>();
    detector.next = first;
    final outputs = <int>[];
    final engine = NcmStaticGestureRuntime(
        runtime: StaticGestureRuntimeApi(classifier: _Classifier()),
        detector: detector,
        host: '127.0.0.1',
        port: server.port,
        onFrame: (frame, result) => outputs.add(frame.sequence));
    addTearDown(() async {
      if (!first.isCompleted) first.complete([]);
      await engine.close();
      for (final s in sockets) {
        s.destroy();
      }
      await sub.cancel();
      await server.close();
    });
    await engine.start();
    await _until(() => sockets.isNotEmpty);
    void send(int sequence) => sockets.last.add(JlipParser.packet(
        type: 16,
        sequence: sequence,
        timestampMs: sequence * 100,
        payload: _jpeg(sequence)));
    send(1);
    await sockets.last.flush();
    await _until(() => detector.seen.length == 1);
    send(2);
    send(3);
    await sockets.last.flush();
    await _until(() => engine.telemetry.replacedFrames == 1);
    first.complete([]);
    await _until(() => outputs.length == 2);
    expect(detector.seen, [1, 3]);
    expect(outputs, [1, 3]);
    final delayed = Completer<List<Hand>>();
    detector.next = delayed;
    send(4);
    await sockets.last.flush();
    await _until(() => detector.seen.length == 3);
    final stop = engine.stop();
    delayed.complete([]);
    await stop;
    expect(outputs, [1, 3]);
    await engine.start();
    await _until(() => sockets.length == 2);
    send(5);
    await sockets.last.flush();
    await _until(() => outputs.length == 3);
    expect(outputs.last, 5);
  });

  test('stop during initialization never connects a stale start', () async {
    final detector = _Detector()..initialization = Completer<void>();
    var results = 0;
    final engine = NcmStaticGestureRuntime(
        runtime: StaticGestureRuntimeApi(classifier: _Classifier()),
        detector: detector,
        onFrame: (_, __) => results++);
    final start = engine.start();
    await Future<void>.delayed(Duration.zero);
    final stop = engine.stop();
    detector.initialization!.complete();
    await start;
    await stop;
    await engine.close();
    expect(detector.closed, isTrue);
    expect(results, 0);
  });
}
