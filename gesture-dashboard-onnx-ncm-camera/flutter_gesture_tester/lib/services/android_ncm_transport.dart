import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';

/// Android binds the remote TCP socket to the board's Ethernet Network. The
/// local relay keeps parsing/heartbeat/reconnect logic in the shared Dart client.
class AndroidNcmTransport {
  static const channel = MethodChannel('gesture_lab/ncm_network');
  AndroidNcmTransport({this.onRoute});
  final void Function(String)? onRoute;
  int _generation = 0;

  static Future<List<String>> describeUsb() async {
    final rows = await channel.invokeListMethod<String>('describeUsb');
    return rows ?? const [];
  }

  Future<Socket> connect(String host, int port, Duration timeout) async {
    final epoch = _generation;
    final endpoint = await channel.invokeMapMethod<String, Object?>('open', {
      'host': host,
      'port': port,
      'timeoutMs': timeout.inMilliseconds,
    });
    if (epoch != _generation) throw StateError('NCM connection cancelled');
    if (endpoint == null ||
        endpoint['port'] is! int ||
        endpoint['token'] is! String ||
        (endpoint['token']! as String).length != 64) {
      throw StateError('Invalid Android NCM endpoint');
    }
    Socket? socket;
    try {
      socket = await Socket.connect(
        '127.0.0.1',
        endpoint['port']! as int,
        timeout: timeout,
      );
      if (epoch != _generation) throw StateError('NCM connection cancelled');
      socket.add(ascii.encode(endpoint['token']! as String));
      await socket.flush();
      onRoute?.call(
        '${endpoint['interface']} ${endpoint['addresses']} → $host:$port',
      );
      return socket;
    } catch (_) {
      socket?.destroy();
      rethrow;
    }
  }

  Future<void> close() async {
    _generation++;
    await channel.invokeMethod<void>('close');
  }
}
