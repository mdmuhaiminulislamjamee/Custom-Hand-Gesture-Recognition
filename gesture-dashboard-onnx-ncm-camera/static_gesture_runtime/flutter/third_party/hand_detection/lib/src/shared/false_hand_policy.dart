import 'dart:math' as math;

class FaceBox {
  const FaceBox(this.x, this.y, this.width, this.height, [this.score = 1]);
  final double x, y, width, height, score;
}

double _distance(List<double> a, List<double> b) =>
    math.sqrt(List.generate(a.length, (i) => math.pow(a[i] - b[i], 2))
        .fold<double>(0, (a, b) => a + b));
double _median(List<double> values) {
  final sorted = [...values]..sort();
  final n = sorted.length;
  return n.isOdd ? sorted[n ~/ 2] : (sorted[n ~/ 2 - 1] + sorted[n ~/ 2]) / 2;
}

bool _collapsed(List<List<double>> p) {
  final palm = _median([
    for (final i in [5, 9, 13, 17]) _distance(p[i], p[0])
  ]);
  if (!palm.isFinite || palm < 1e-6) return true;
  return [5, 9, 13, 17]
          .where((s) =>
              List.generate(3, (i) => _distance(p[s + i], p[s + i + 1]))
                  .fold<double>(0, (a, b) => a + b) <
              .12 * palm)
          .length >=
      3;
}

/// Source-image pixels, before clamping or ROI magnification. Matches Python.
String? falseHandReason(
    List<List<double>> p, int width, int height, List<FaceBox> faces,
    {List<List<double>>? world}) {
  if (p.length != 21 ||
      p.any((v) => v.length != 2 || v.any((x) => !x.isFinite))) {
    return 'incomplete_hand';
  }
  final xs = p.map((v) => v[0]), ys = p.map((v) => v[1]);
  final dx = xs.reduce(math.max) - xs.reduce(math.min);
  final dy = ys.reduce(math.max) - ys.reduce(math.min);
  final palm = _median([
    for (final pair in [
      [0, 5],
      [0, 9],
      [0, 17],
      [5, 17]
    ])
      _distance(p[pair[0]], p[pair[1]])
  ]);
  String? reason;
  if (palm < 18 || math.max(dx, dy) < 40 || math.min(dx, dy) < 12) {
    reason = 'insufficient_hand_pixels';
  }
  final inside = p
          .where((v) =>
              v[0] >= -.05 * width &&
              v[0] <= 1.05 * width &&
              v[1] >= -.05 * height &&
              v[1] <= 1.05 * height)
          .length /
      21;
  if (inside < .9) reason ??= 'incomplete_hand';
  if (reason == null && _collapsed(p)) {
    final validWorld = world != null &&
        world.length == 21 &&
        world.every((v) => v.length == 3 && v.every((x) => x.isFinite));
    if (!validWorld || _collapsed(world)) reason = 'invalid_hand_geometry';
  }
  final center = [
    for (var axis = 0; axis < 2; axis++)
      [0, 5, 9, 13, 17].fold<double>(0, (v, i) => v + p[i][axis]) / 5
  ];
  for (final f in faces) {
    if (f.width <= 0 || f.height <= 0) continue;
    bool covers(List<double> v) =>
        v[0] >= f.x - .3 * f.width &&
        v[0] <= f.x + 1.3 * f.width &&
        v[1] >= f.y - .15 * f.height &&
        v[1] <= f.y + 1.05 * f.height;
    if (covers(center) &&
        math.max(dx, dy) < .7 * math.max(f.width, f.height) &&
        p.where(covers).length / 21 >= .75) return 'face_detected_as_hand';
  }
  return reason;
}

double faceOverlap(FaceBox a, FaceBox b) {
  final area = math.max(
          0.0, math.min(a.x + a.width, b.x + b.width) - math.max(a.x, b.x)) *
      math.max(
          0.0, math.min(a.y + a.height, b.y + b.height) - math.max(a.y, b.y));
  return area / math.max(1e-9, a.width * a.height + b.width * b.height - area);
}

/// BlazeFace short-range graph: fixed 896 anchors, reverse XYWH, weighted NMS.
/// https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection_short_range.pbtxt
List<FaceBox> decodeFaces(
    List<double> boxes, List<double> scores, int width, int height,
    {double threshold = .65}) {
  if (boxes.length != 896 * 16 || scores.length != 896)
    throw StateError('Invalid BlazeFace outputs');
  final candidates = <FaceBox>[];
  var index = 0;
  final newW = (128 * width / math.max(width, height)).floor();
  final newH = (128 * height / math.max(width, height)).floor();
  final padX = (128 - newW) ~/ 2, padY = (128 - newH) ~/ 2;
  for (final layer in [(16, 2), (8, 6)]) {
    for (var y = 0; y < layer.$1; y++) {
      for (var x = 0; x < layer.$1; x++) {
        for (var a = 0; a < layer.$2; a++, index++) {
          final logit = scores[index];
          if (!logit.isFinite) continue;
          final score = 1 / (1 + math.exp(-logit.clamp(-100, 100)));
          if (score < threshold) continue;
          final offset = index * 16;
          final cx = boxes[offset] + (x + .5) / layer.$1 * 128;
          final cy = boxes[offset + 1] + (y + .5) / layer.$1 * 128;
          final w = boxes[offset + 2], h = boxes[offset + 3];
          if (![cx, cy, w, h].every((v) => v.isFinite) || w <= 0 || h <= 0)
            continue;
          candidates.add(FaceBox(
              (cx - w / 2 - padX) * width / newW,
              (cy - h / 2 - padY) * height / newH,
              w * width / newW,
              h * height / newH,
              score));
        }
      }
    }
  }
  candidates.sort((a, b) => b.score.compareTo(a.score));
  final result = <FaceBox>[];
  while (candidates.isNotEmpty && result.length < 8) {
    final first = candidates.first;
    final group = candidates
        .where((b) => identical(b, first) || faceOverlap(first, b) > .3)
        .toList();
    final total = group.fold<double>(0, (v, b) => v + b.score);
    double weighted(double Function(FaceBox) get) =>
        group.fold<double>(0, (v, b) => v + get(b) * b.score) / total;
    result.add(FaceBox(weighted((b) => b.x), weighted((b) => b.y),
        weighted((b) => b.width), weighted((b) => b.height), first.score));
    candidates.removeWhere(group.contains);
  }
  return result;
}

class FaceBoxCache {
  final _recent = <(FaceBox, double)>[];
  double _attempt = double.negativeInfinity;
  int _width = 0, _height = 0;
  List<FaceBox> get boxes => [for (final row in _recent) row.$1];
  bool refresh(double nowMs, int width, int height) {
    if (width != _width || height != _height || nowMs < _attempt) reset();
    _width = width;
    _height = height;
    _recent.removeWhere((row) => nowMs - row.$2 > 500 || nowMs < row.$2);
    if (nowMs - _attempt < 200) return false;
    _attempt = nowMs;
    return true;
  }

  void observe(List<FaceBox> fresh, double nowMs) {
    _recent.removeWhere(
        (old) => fresh.any((box) => faceOverlap(old.$1, box) > .15));
    _recent.addAll(fresh.map((box) => (box, nowMs)));
  }

  void reset() {
    _recent.clear();
    _attempt = double.negativeInfinity;
    _width = _height = 0;
  }
}
