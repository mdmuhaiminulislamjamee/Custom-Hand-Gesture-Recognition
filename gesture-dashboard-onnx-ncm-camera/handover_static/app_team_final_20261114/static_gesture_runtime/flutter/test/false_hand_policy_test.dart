import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/src/shared/false_hand_policy.dart';
import 'package:flutter_litert/native.dart';

void main() {
  test('candidate policy matches Python across mirrors, sizes and world rescue',
      () {
    final data = jsonDecode(
        File('../validation/false_hand_policy.json').readAsStringSync()) as Map;
    List<List<double>> points(dynamic rows) => [
          for (final row in rows as List)
            [for (final v in row as List) (v as num).toDouble()]
        ];
    for (final c in data['cases']) {
      final faces = [
        for (final f in c['faces'])
          FaceBox((f[0] as num).toDouble(), (f[1] as num).toDouble(),
              (f[2] as num).toDouble(), (f[3] as num).toDouble())
      ];
      expect(
          falseHandReason(points(c['pixels']), c['width'], c['height'], faces,
              world: c['world'] == null ? null : points(c['world'])),
          c['reason'],
          reason: c['id']);
    }
    expect(data['cases'].length, 42);
  });
  test('BlazeFace anchors, letterboxing, logits and weighted NMS', () {
    final boxes = Float32List(896 * 16),
        scores = Float32List(896)..fillRange(0, 896, -100);
    // Two identical anchor centers; output represents a 32x32 tensor-space box.
    for (final i in [0, 1]) {
      scores[i] = 10;
      boxes[i * 16 + 2] = 32;
      boxes[i * 16 + 3] = 32;
    }
    final faces = decodeFaces(boxes, scores, 640, 480);
    expect(faces.length, 1);
    expect(faces.single.x, closeTo(-60, 1e-6));
    expect(faces.single.y, closeTo(-140, 1e-6));
    expect(faces.single.width, closeTo(160, 1e-6));
    scores.fillRange(0, 896, -100);
    expect(decodeFaces(boxes, scores, 640, 480), isEmpty);
  });
  test('face cache refresh, expiration, reset, rewind and stream isolation',
      () {
    final a = FaceBoxCache(), b = FaceBoxCache();
    expect(a.refresh(1000, 640, 480), isTrue);
    a.observe([const FaceBox(350, 80, 140, 210)], 1000);
    expect(a.refresh(1100, 640, 480), isFalse);
    expect(a.boxes.length, 1);
    expect(b.boxes, isEmpty);
    expect(a.refresh(1201, 640, 480), isTrue);
    a.observe([], 1201);
    a.refresh(1501, 640, 480);
    expect(a.boxes, isEmpty);
    a.observe([const FaceBox(1, 2, 3, 4)], 1501);
    a.refresh(1, 640, 480);
    expect(a.boxes, isEmpty);
    a.observe([const FaceBox(1, 2, 3, 4)], 1);
    a.refresh(2, 480, 640);
    expect(a.boxes, isEmpty);
  });
  test(
      'bundled real face model has the pinned contract and rejects blank input',
      () {
    final options = InterpreterOptions()..threads = 1;
    final model = Interpreter.fromBuffer(
        File('third_party/hand_detection/assets/models/blaze_face_short_range.tflite')
            .readAsBytesSync(),
        options: options);
    try {
      model.allocateTensors();
      expect(model.getInputTensor(0).shape, [1, 128, 128, 3]);
      final views = TensorFloat32Views.capture(model);
      views.inputs[0].fillRange(0, views.inputs[0].length, 0);
      model.invoke();
      final boxes = views.outputs.firstWhere((v) => v.length == 896 * 16);
      final scores = views.outputs.firstWhere((v) => v.length == 896);
      expect(decodeFaces(boxes, scores, 640, 480), isEmpty);
    } finally {
      model.close();
      options.delete();
    }
  });
}
