import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/hand_detection.dart';
import 'package:static_gesture_runtime/ncm_static_gesture_runtime.dart';
import 'package:static_gesture_runtime/static_gesture_preview.dart';
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';

import 'detector_adapter_test.dart' show detectedHand;

void main() {
  testWidgets(
      'paired preview draws both detector boxes, even without a command',
      (tester) async {
    late Uint8List png;
    await tester.runAsync(() async {
      final recorder = ui.PictureRecorder();
      Canvas(recorder).drawColor(const Color(0xff101010), BlendMode.src);
      final picture = recorder.endRecording();
      final image = await picture.toImage(640, 480);
      png = (await image.toByteData(format: ui.ImageByteFormat.png))!
          .buffer
          .asUint8List();
      image.dispose();
      picture.dispose();
    });
    NcmStaticPreviewFrame frame(List<Hand> hands, int sequence) =>
        NcmStaticPreviewFrame(
          source: NcmJpegFrame(
              jpeg: png,
              sequence: sequence,
              boardTimestampMs: sequence,
              receivedTimestampMs: sequence * 100),
          width: 640,
          height: 480,
          detections: hands,
          result: StaticGestureFrameResult(
              timestampMs: sequence * 100,
              hands: [],
              command: null,
              runtimeMs: 0,
              queueWaitMs: 0,
              replacedFrames: 0),
        );
    final boundaryKey = GlobalKey();
    Future<void> show(NcmStaticPreviewFrame preview) async {
      await tester.pumpWidget(MaterialApp(
          home: Center(
              child: SizedBox(
        width: 400,
        child: RepaintBoundary(
            key: boundaryKey, child: StaticNcmPreview(frame: preview)),
      ))));
      await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 40)));
      await tester.pumpAndSettle();
    }

    await show(frame([
      detectedHand(Handedness.left, 100),
      detectedHand(Handedness.right, 400),
    ], 1));
    final painter = tester
        .widgetList<CustomPaint>(find.byType(CustomPaint))
        .map((w) => w.foregroundPainter)
        .whereType<MultiOverlayPainter>()
        .single;
    expect(painter.results, hasLength(2));
    expect(tester.getSize(find.byType(StaticNcmPreview)), const Size(400, 300));
    final boundary = boundaryKey.currentContext!.findRenderObject()!
        as RenderRepaintBoundary;
    late ByteData pixels;
    await tester.runAsync(() async {
      final image = await boundary.toImage();
      pixels = (await image.toByteData())!;
      image.dispose();
    });
    // Both box top edges must be visible at their unmirrored source positions.
    for (final x in [65, 253]) {
      final offset = (63 * 400 + x) * 4;
      expect(pixels.getUint8(offset), greaterThan(100));
    }
    await show(frame([], 2));
    final emptyPainter = tester
        .widgetList<CustomPaint>(find.byType(CustomPaint))
        .map((w) => w.foregroundPainter)
        .whereType<MultiOverlayPainter>()
        .single;
    expect(emptyPainter.results, isEmpty);
    expect(tester.takeException(), isNull);
  });
}
