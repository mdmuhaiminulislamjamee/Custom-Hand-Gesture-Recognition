part of 'runtime_api.dart';

class _OwnershipDecision {
  const _OwnershipDecision(this.hand, this.pending, this.changed, this.reason,
      {this.allowCommand = false});
  final StaticPhysicalHand? hand;
  final bool pending, changed, allowCommand;
  final String reason;
}

/// Relative foreground selection, using the desktop selector's 1.10 projected
/// world / 1.15 palm ratios, .50 smoothing, and 3-frame / 250ms confirmation.
/// Per-hand world Z is never treated as absolute camera distance.
class _HandOwnership {
  StaticPhysicalHand? _owner, _previousOwner, _pending;
  String? _pendingMode;
  double _pendingSince = 0, _pendingLast = 0;
  int _pendingFrames = 0;
  final Map<StaticPhysicalHand, _HandScale> _tracks = {};

  void reset() {
    _owner = _previousOwner = null;
    _tracks.clear();
    _clearPending();
  }

  void invalidateEvidence() {
    _tracks.clear();
    _clearPending();
  }

  void _clearPending() {
    _pending = null;
    _pendingMode = null;
    _pendingFrames = 0;
  }

  bool _confirm(StaticPhysicalHand side, double now, String mode) {
    if (_pending != side || _pendingMode != mode || now - _pendingLast >= 350) {
      _pending = side;
      _pendingMode = mode;
      _pendingSince = now;
      _pendingFrames = 0;
    }
    _pendingLast = now;
    _pendingFrames++;
    return _pendingFrames >= 3 && now - _pendingSince >= 250;
  }

  _OwnershipDecision update(
      StaticGestureFrame frame, List<StaticHandResult> results) {
    final now = frame.timestampMs;
    final available = {for (final result in results) result.hand: result};
    final raw = <StaticPhysicalHand, _HandScale>{};
    for (final hand in frame.hands) {
      if (!available.containsKey(hand.hand)) continue;
      final scale = _HandScale.measure(hand, frame);
      raw[hand.hand] = scale;
      final old = _tracks[hand.hand];
      _tracks[hand.hand] =
          old == null || now - old.time >= 350 ? scale : old.smooth(scale);
    }
    if (_owner != null && !available.containsKey(_owner)) {
      final last = _tracks[_owner];
      if (last == null || now - last.time >= 350) _owner = null;
    }
    _tracks.removeWhere((side, track) => now - track.time >= 350);
    final visibleOwner = available.containsKey(_owner) ? _owner : null;
    if (results.isEmpty) {
      _clearPending();
      return const _OwnershipDecision(null, false, false, 'no_current_hand');
    }
    if (_owner != null && visibleOwner == null) {
      _clearPending();
      return const _OwnershipDecision(
          null, false, false, 'active_hand_missing');
    }
    if (raw.length == 2) {
      final pair = raw.values.toList();
      final separation = math.sqrt(math.pow(pair[0].x - pair[1].x, 2) +
          math.pow(pair[0].y - pair[1].y, 2));
      if (separation < .50 * math.max(pair[0].palm, pair[1].palm)) {
        _clearPending();
        return _OwnershipDecision(visibleOwner, false, false, 'hands_overlap');
      }
    }
    final qualified = results
        .where((r) => r.geometry.accepted && r.controller.label != 'no_gesture')
        .toList();
    StaticPhysicalHand? candidate;
    var mode = 'gesture';
    if (qualified.length == 1) {
      candidate = qualified.single.hand;
    } else if (qualified.length == 2) {
      final near = _nearer(raw);
      if (near != null && near == _nearer(_tracks)) candidate = near;
      mode = 'distance';
    }
    if (candidate == _owner || candidate == null) {
      _clearPending();
      // An unowned single hand still supplies rejection/confirmation feedback.
      final feedback =
          visibleOwner ?? (results.length == 1 ? results.single.hand : null);
      return _OwnershipDecision(feedback, false, false,
          _owner != null ? 'active_hand_locked' : 'waiting_for_gesture',
          allowCommand: visibleOwner != null);
    }
    final challenger = available[candidate]!;
    final ready = challenger.controller.confirmed &&
        challenger.controller.label != 'no_gesture';
    final replacing = _owner != null ||
        (_previousOwner != null && _previousOwner != candidate);
    // Preserve the existing single-hand strong/ordinary confirmation. Every
    // switch, plus selection between two visible hands, needs a stable choice.
    final needsHold = replacing || results.length > 1;
    if (!ready || (needsHold && !_confirm(candidate, now, mode))) {
      if (!ready) _clearPending();
      return _OwnershipDecision(
          visibleOwner ??
              (results.length == 1 && !replacing ? candidate : null),
          replacing,
          false,
          mode == 'distance'
              ? 'confirming_closer_hand'
              : 'confirming_valid_hand');
    }
    final changed = replacing;
    _owner = _previousOwner = candidate;
    _clearPending();
    return _OwnershipDecision(
        candidate, false, changed, changed ? 'hand_switched' : 'hand_selected',
        allowCommand: !changed);
  }

  StaticPhysicalHand? _nearer(Map<StaticPhysicalHand, _HandScale> scales) {
    if (scales.length != 2) return null;
    final a = scales[StaticPhysicalHand.left]!;
    final b = scales[StaticPhysicalHand.right]!;
    if (a.projected != null && b.projected != null) {
      if (a.projected! >= b.projected! * 1.10) return StaticPhysicalHand.left;
      if (b.projected! >= a.projected! * 1.10) return StaticPhysicalHand.right;
      return null;
    }
    final ratios = [
      for (var i = 0; i < 4; i++) a.segments[i] / math.max(b.segments[i], 1)
    ];
    if (a.segments.any((v) => v < 8) ||
        b.segments.any((v) => v < 8) ||
        _percentile(ratios, .8) / math.max(_percentile(ratios, .2), .01) >
            1.40) {
      return null;
    }
    if (a.palm >= b.palm * 1.15) return StaticPhysicalHand.left;
    if (b.palm >= a.palm * 1.15) return StaticPhysicalHand.right;
    return null;
  }
}

class _HandScale {
  const _HandScale(this.time, this.x, this.y, this.segments, this.projected);
  final double time, x, y;
  final List<double> segments;
  final double? projected;
  double get palm => _percentile(segments, .5);

  factory _HandScale.measure(StaticGestureHand hand, StaticGestureFrame frame) {
    final points = hand.normalizedLandmarks;
    double distance(List<List<double>> p, int a, int b, double w, double h) =>
        math.sqrt(math.pow((p[a][0] - p[b][0]) * w, 2) +
            math.pow((p[a][1] - p[b][1]) * h, 2));
    final segments = [
      for (final pair in const [
        [0, 5],
        [0, 9],
        [0, 17],
        [5, 17]
      ])
        distance(points, pair[0], pair[1], frame.frameWidth, frame.frameHeight)
    ];
    final ratios = <double>[];
    final world = hand.worldLandmarks;
    if (world != null) {
      for (final pair in const [
        [0, 5],
        [0, 9],
        [0, 13],
        [0, 17],
        [5, 9],
        [9, 13],
        [13, 17],
        [5, 17],
        [5, 6],
        [6, 7],
        [7, 8],
        [9, 10],
        [10, 11],
        [11, 12],
        [13, 14],
        [14, 15],
        [17, 18]
      ]) {
        final metric = distance(world, pair[0], pair[1], 1, 1);
        final pixels = distance(
            points, pair[0], pair[1], frame.frameWidth, frame.frameHeight);
        if (metric >= .006 && pixels >= 5) ratios.add(pixels / metric);
      }
    }
    final projected = ratios.length >= 7 &&
            _percentile(ratios, .2) > 0 &&
            _percentile(ratios, .8) / _percentile(ratios, .2) <= 1.45
        ? _percentile(ratios, .5)
        : null;
    final palm = [
      for (final i in [0, 5, 9, 13, 17]) points[i]
    ];
    return _HandScale(
        frame.timestampMs,
        palm.fold(0.0, (v, p) => v + p[0]) / 5 * frame.frameWidth,
        palm.fold(0.0, (v, p) => v + p[1]) / 5 * frame.frameHeight,
        segments,
        projected);
  }

  _HandScale smooth(_HandScale next) {
    final alpha = 1 - math.pow(.50, math.min(next.time - time, 500) / 100);
    return _HandScale(
        next.time,
        next.x,
        next.y,
        [
          for (var i = 0; i < 4; i++)
            segments[i] + alpha * (next.segments[i] - segments[i])
        ],
        next.projected == null
            ? null
            : projected == null
                ? next.projected
                : projected! + alpha * (next.projected! - projected!));
  }
}

double _percentile(List<double> values, double fraction) {
  final sorted = [...values]..sort();
  final index = (sorted.length - 1) * fraction;
  final lo = index.floor(), hi = index.ceil();
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
}
