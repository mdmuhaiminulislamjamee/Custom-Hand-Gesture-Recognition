import 'dart:math' as math;
import 'dart:typed_data';

/// Exact 76-value XY feature contract of backend/geometry.py.
/// Image coordinates must be upright and unmirrored. Z is intentionally unused.
abstract final class StaticFeatureExtractor {
  static const classNames = <String>[
    'left',
    'right',
    'up',
    'down',
    'open_palm',
    'like',
    'dorsal',
    'ok',
    'fist',
    'thumb_down',
    'peace',
    'rock',
    'stop_inverted',
  ];

  static Float32List extract(List<List<double>> landmarks) {
    validateLandmarks(landmarks);
    final p = landmarks.map((v) => <double>[v[0], v[1]]).toList();
    final palmScale = gapScale(p);
    final extensions = <double>[
      for (final m in <int>[1, 5, 9, 13, 17]) extension(p, m),
    ];
    final shape = <double>[
      ...extensions,
      1 - mean(extensions.sublist(2)),
      distance(p[4], p[5]) / palmScale,
      distance(p[4], p[0]) / palmScale,
    ];
    final centered = p.map((v) => subtract(v, p[0])).toList();
    final direction = unit(subtract(p[8], p[5]));
    final wristIndex = unit(centered[8]);
    if (norm(centered[9]) < epsilon) {
      throw const FormatException('Degenerate wrist-to-middle geometry.');
    }
    final angle = -math.pi / 2 - math.atan2(centered[9][1], centered[9][0]);
    final c = math.cos(angle), s = math.sin(angle);
    var rotated = <List<double>>[
      for (final v in centered)
        <double>[c * v[0] - s * v[1], s * v[0] + c * v[1]],
    ];
    final scale = maxValue(rotated.map(norm));
    if (scale < epsilon) throw const FormatException('Degenerate hand scale.');
    rotated = rotated.map((v) => divide(v, scale)).toList();
    if (rotated[5][0] < rotated[17][0]) {
      rotated = rotated.map((v) => <double>[-v[0], v[1]]).toList();
    }
    return Float32List.fromList(<double>[
      for (final v in rotated) ...v,
      for (final v in rotated) norm(v),
      ...direction,
      ...wristIndex,
      distance(p[4], p[8]) / palmScale,
      ...shape,
    ]);
  }
}

const double epsilon = 1e-8;
void validateLandmarks(List<List<double>> points, {int dimensions = 2}) {
  if (points.length != 21 ||
      points.any(
        (p) =>
            p.length < dimensions || p.take(dimensions).any((v) => !v.isFinite),
      )) {
    throw FormatException('Expected 21 finite $dimensions-D landmarks.');
  }
}

List<double> subtract(List<double> a, List<double> b) => <double>[
      for (var i = 0; i < a.length; i++) a[i] - b[i],
    ];
List<double> add(List<double> a, List<double> b) => <double>[
      for (var i = 0; i < a.length; i++) a[i] + b[i],
    ];
List<double> multiply(List<double> a, double scale) =>
    a.map((x) => x * scale).toList();
List<double> divide(List<double> a, double scale) =>
    a.map((x) => x / scale).toList();
double dot(List<double> a, List<double> b) {
  var result = 0.0;
  for (var i = 0; i < a.length; i++) {
    result += a[i] * b[i];
  }
  return result;
}

double norm(List<double> a) => math.sqrt(dot(a, a));
double distance(List<double> a, List<double> b) => norm(subtract(a, b));
List<double> unit(List<double> a) {
  final length = norm(a);
  if (length < epsilon) throw const FormatException('Degenerate direction.');
  return divide(a, length);
}

double mean(Iterable<double> values) {
  final v = values.toList();
  return v.fold(0.0, (a, b) => a + b) / v.length;
}

double maxValue(Iterable<double> values) => values.reduce(math.max);
double minValue(Iterable<double> values) => values.reduce(math.min);
double clip(double x, [double low = 0, double high = 1]) =>
    x.clamp(low, high).toDouble();
double gapScale(List<List<double>> p) {
  final references = <double>[
    distance(p[5], p[0]),
    distance(p[9], p[0]),
    distance(p[17], p[0]),
    distance(p[17], p[5]),
  ].where((x) => x > epsilon).toList()
    ..sort();
  if (references.isEmpty) throw const FormatException('Degenerate palm scale.');
  final mid = references.length ~/ 2;
  return references.length.isOdd
      ? references[mid]
      : (references[mid - 1] + references[mid]) / 2;
}

double jointAngle(List<double> a, List<double> b, List<double> c) {
  final first = subtract(a, b), second = subtract(c, b);
  final denominator = norm(first) * norm(second);
  if (denominator < epsilon) return 0;
  return math.acos(clip(dot(first, second) / denominator, -1, 1)) *
      180 /
      math.pi;
}

double extension(List<List<double>> p, int m) => clip(
      ((jointAngle(p[m], p[m + 1], p[m + 2]) +
                      jointAngle(p[m + 1], p[m + 2], p[m + 3])) *
                  .5 -
              110) /
          60,
    );
List<double> averagePoints(List<List<double>> p, List<int> indices) => <double>[
      for (var axis = 0; axis < p.first.length; axis++)
        mean(indices.map((i) => p[i][axis])),
    ];
