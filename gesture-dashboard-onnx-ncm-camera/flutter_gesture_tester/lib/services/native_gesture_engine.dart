import 'package:camera/camera.dart';
import 'package:flutter/services.dart';

import '../models/gesture_result.dart';

class NativeGestureEngine {
  static const MethodChannel _channel = MethodChannel('gesture_lab/inference');
  static const EventChannel _ncmChannel = EventChannel(
    'gesture_lab/ncm_events',
  );
  Stream<Map<Object?, Object?>>? _ncmEvents;

  Stream<Map<Object?, Object?>> get ncmEvents => _ncmEvents ??= _ncmChannel
      .receiveBroadcastStream()
      .map((event) => event! as Map<Object?, Object?>)
      .asBroadcastStream();

  Future<Map<Object?, Object?>> initialize() async {
    final result = await _channel.invokeMethod<Map<Object?, Object?>>(
      'initialize',
    );
    return result ?? const {};
  }

  Future<GestureResult> analyze(CameraImage image, int rotation) async {
    final result = await _channel.invokeMethod<Map<Object?, Object?>>(
      'analyze',
      {
        'width': image.width,
        'height': image.height,
        'rotation': rotation,
        'format': image.format.raw,
        'planes': image.planes
            .map((plane) => plane.bytes)
            .toList(growable: false),
        'rowStrides': image.planes
            .map((plane) => plane.bytesPerRow)
            .toList(growable: false),
        'pixelStrides': image.planes
            .map((plane) => plane.bytesPerPixel ?? 1)
            .toList(growable: false),
      },
    );
    if (result == null) {
      throw StateError('The Android inference bridge returned no result.');
    }
    return GestureResult.fromMap(result);
  }

  Future<void> reset() => _channel.invokeMethod<void>('reset');

  Future<Map<Object?, Object?>> startNcm({
    String host = '192.168.50.2',
    int port = 5000,
  }) async {
    final result = await _channel.invokeMethod<Map<Object?, Object?>>(
      'startNcm',
      {'host': host, 'port': port},
    );
    return result ?? const {};
  }

  Future<void> stopNcm() => _channel.invokeMethod<void>('stopNcm');
}
