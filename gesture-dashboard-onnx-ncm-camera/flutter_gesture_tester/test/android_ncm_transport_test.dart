import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:gesture_model_tester/services/android_ncm_transport.dart';
import 'package:static_gesture_runtime/ncm_static_gesture_runtime.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final messenger =
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  const token =
      '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef';
  final servers = <ServerSocket>[];
  final sockets = <Socket>[];
  tearDown(() async {
    messenger.setMockMethodCallHandler(AndroidNcmTransport.channel, null);
    for (final socket in sockets) {
      socket.destroy();
    }
    for (final server in servers) {
      await server.close();
    }
    sockets.clear();
    servers.clear();
  });

  test(
    'relays JPEG bytes and heartbeat replies with an authenticated prefix',
    () async {
      final server = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
      servers.add(server);
      final heartbeat = Completer<void>();
      final jpeg = Uint8List.fromList([255, 216, 1, 2, 255, 217]);
      server.listen((socket) {
        sockets.add(socket);
        final pending = <int>[];
        final parser = JlipParser();
        var authenticated = false;
        socket.listen((bytes) {
          pending.addAll(bytes);
          if (!authenticated) {
            if (pending.length < 64) return;
            expect(ascii.decode(pending.take(64).toList()), token);
            pending.removeRange(0, 64);
            authenticated = true;
            socket.add(
              JlipParser.packet(type: 3, sequence: 1, timestampMs: 10),
            );
            socket.add(
              JlipParser.packet(
                type: 0x10,
                sequence: 2,
                timestampMs: 11,
                payload: jpeg,
              ),
            );
          }
          for (final packet in parser.feed(Uint8List.fromList(pending))) {
            if (packet.type == 3 && !heartbeat.isCompleted) {
              heartbeat.complete();
            }
          }
          pending.clear();
        });
      });
      var closes = 0;
      messenger.setMockMethodCallHandler(AndroidNcmTransport.channel, (
        call,
      ) async {
        if (call.method == 'close') {
          closes++;
          return null;
        }
        expect(call.arguments, {
          'host': '192.168.50.2',
          'port': 5000,
          'timeoutMs': 3000,
        });
        return {
          'port': server.port,
          'token': token,
          'interface': 'usb0',
          'addresses': ['192.168.50.1/30'],
        };
      });
      String? route;
      final transport = AndroidNcmTransport(onRoute: (value) => route = value);
      final frame = Completer<NcmJpegFrame>();
      final client = NcmCameraClient(
        socketConnector: transport.connect,
        closeTransport: transport.close,
        onFrame: (value) => frame.complete(value),
      );
      addTearDown(client.stop);
      await client.start();
      expect(
        (await frame.future.timeout(const Duration(seconds: 3))).jpeg,
        jpeg,
      );
      await heartbeat.future.timeout(const Duration(seconds: 3));
      expect(route, contains('usb0'));
      await client.stop();
      expect(closes, 1);
    },
  );

  test(
    'reconnect asks Android for a fresh network and preserves route errors',
    () async {
      var attempts = 0;
      final recovered = Completer<void>();
      final errors = <Object>[];
      final server = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
      servers.add(server);
      server.listen((socket) {
        sockets.add(socket);
      });
      messenger.setMockMethodCallHandler(AndroidNcmTransport.channel, (
        call,
      ) async {
        if (call.method == 'close') return null;
        if (++attempts == 1) {
          throw PlatformException(
            code: 'NCM_ROUTE_UNAVAILABLE',
            message: 'Ethernet has no board route',
          );
        }
        return {'port': server.port, 'token': token};
      });
      final transport = AndroidNcmTransport();
      final client = NcmCameraClient(
        socketConnector: transport.connect,
        closeTransport: transport.close,
        reconnectDelay: const Duration(milliseconds: 20),
        onError: errors.add,
        onFrame: (_) {},
        onState: (state) {
          if (state == 'connected' && !recovered.isCompleted) {
            recovered.complete();
          }
        },
      );
      addTearDown(client.stop);
      await client.start();
      await recovered.future.timeout(const Duration(seconds: 3));
      expect(attempts, 2);
      expect(
        (errors.single as PlatformException).code,
        'NCM_ROUTE_UNAVAILABLE',
      );
    },
  );

  test(
    'stop during Android connection prevents late local connection',
    () async {
      final endpoint = Completer<Map<String, Object?>>();
      final opened = Completer<void>();
      messenger.setMockMethodCallHandler(AndroidNcmTransport.channel, (
        call,
      ) async {
        if (call.method == 'close') return null;
        opened.complete();
        return endpoint.future;
      });
      final transport = AndroidNcmTransport();
      final pending = transport.connect(
        '192.168.50.2',
        5000,
        const Duration(seconds: 3),
      );
      final cancelled = expectLater(pending, throwsStateError);
      await opened.future;
      await transport.close();
      endpoint.complete({'port': 1, 'token': token});
      await cancelled;
    },
  );
}
