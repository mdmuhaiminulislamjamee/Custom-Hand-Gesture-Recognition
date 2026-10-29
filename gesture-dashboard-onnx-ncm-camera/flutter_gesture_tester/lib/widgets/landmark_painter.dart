import 'package:flutter/material.dart';

class LandmarkPainter extends CustomPainter {
  const LandmarkPainter({
    required this.landmarks,
    required this.mirror,
    required this.active,
  });

  final List<Offset> landmarks;
  final bool mirror;
  final bool active;

  static const _connections = <(int, int)>[
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final guide = RRect.fromRectAndRadius(
      Rect.fromCenter(
        center: Offset(size.width / 2, size.height / 2),
        width: size.width * .76,
        height: size.height * .76,
      ),
      const Radius.circular(36),
    );
    canvas.drawRRect(
      guide,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5
        ..color = Colors.white.withValues(alpha: .36),
    );

    if (landmarks.length != 21) return;
    Offset mapPoint(Offset point) => Offset(
      (mirror ? 1 - point.dx : point.dx) * size.width,
      point.dy * size.height,
    );

    final points = landmarks.map(mapPoint).toList(growable: false);
    final glow = active ? const Color(0xFF7EF2CC) : const Color(0xFFFFC37A);
    final linePaint = Paint()
      ..color = glow.withValues(alpha: .75)
      ..strokeWidth = 2.4
      ..strokeCap = StrokeCap.round;
    final glowPaint = Paint()
      ..color = glow.withValues(alpha: .18)
      ..strokeWidth = 7
      ..strokeCap = StrokeCap.round;
    for (final connection in _connections) {
      canvas
        ..drawLine(points[connection.$1], points[connection.$2], glowPaint)
        ..drawLine(points[connection.$1], points[connection.$2], linePaint);
    }
    final pointPaint = Paint()..color = Colors.white;
    final pointGlow = Paint()..color = glow.withValues(alpha: .55);
    for (final point in points) {
      canvas
        ..drawCircle(point, 5, pointGlow)
        ..drawCircle(point, 2.25, pointPaint);
    }
  }

  @override
  bool shouldRepaint(covariant LandmarkPainter oldDelegate) {
    return oldDelegate.landmarks != landmarks ||
        oldDelegate.mirror != mirror ||
        oldDelegate.active != active;
  }
}
