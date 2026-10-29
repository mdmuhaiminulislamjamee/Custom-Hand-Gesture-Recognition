import 'dart:math' as math;

import 'feature_extractor.dart';

/// A qualified per-frame candidate. Confirmation/rearming belongs to the runtime.
class StaticGeometryDecision {
  const StaticGeometryDecision({
    required this.label,
    required this.confidence,
    required this.margin,
    required this.minimumHoldMs,
    required this.reason,
    required this.accepted,
    this.probabilities = const <double>[],
    this.knownMass = 0,
    this.recovery = false,
    this.diagnostics = const <String, Object?>{},
  });
  final String label;
  final double confidence, margin, minimumHoldMs, knownMass;
  final bool accepted, recovery;
  final String reason;
  final List<double> probabilities;
  final Map<String, Object?> diagnostics;
}

/// Port of the production directional and static-shape policy.
///
/// Coordinates are the original upright, unmirrored detector UV coordinates;
/// image dimensions are required for source-pixel angle tests. World landmarks
/// are MediaPipe metric world landmarks, never normalized image Z substituted
/// for world Z. Fist requires world depth in live use.
abstract final class StaticGeometryPolicy {
  static const _names = StaticFeatureExtractor.classNames;
  static const _directions = <String>{'left', 'right', 'up', 'down'};

  static StaticGeometryDecision evaluate({
    required List<double> probabilities,
    required double knownMass,
    required List<List<double>> normalizedLandmarks,
    required String physicalHand,
    required double handednessConfidence,
    required double width,
    required double height,
    List<List<double>>? worldLandmarks,
    String? preferredDirection,
    bool requireFistDepth = true,
  }) {
    if (probabilities.length != 13 ||
        probabilities.any((x) => !x.isFinite || x < 0) ||
        probabilities.fold(0.0, (a, b) => a + b) <= 1e-12 ||
        !knownMass.isFinite ||
        knownMass < 0 ||
        knownMass > 1 ||
        !width.isFinite ||
        width <= 0 ||
        !height.isFinite ||
        height <= 0) {
      return const StaticGeometryDecision(
        label: 'no_gesture',
        confidence: 0,
        margin: 0,
        minimumHoldMs: 0,
        reason: 'invalid classifier/frame values',
        accepted: false,
      );
    }
    try {
      validateLandmarks(normalizedLandmarks);
      if (worldLandmarks != null)
        validateLandmarks(worldLandmarks, dimensions: 3);
      final p = normalizedLandmarks.map((x) => <double>[x[0], x[1]]).toList();
      final model = _normalize(probabilities);
      final direction = _directional(
        model,
        p,
        worldLandmarks,
        preferredDirection,
        width,
        height,
      );
      // The adapter has already converted the detector's selfie category to
      // physical identity. Surface winding also uses physical Left/Right;
      // swapping again reverses Open Palm and Stop Inverted.
      final shape = _shape(
        direction.row,
        model,
        p,
        worldLandmarks,
        physicalHand.toLowerCase(),
        handednessConfidence,
        width,
        height,
        requireFistDepth,
      );
      var row = shape.row;
      final d = direction.details, s = shape.details;
      if (_flag(d, 'prominent_index_down')) row = _floor(row, 'down', .98);
      for (final key in <String>[
        'expanded_thumb_pointing',
        'continued_direction',
        'horizontal_pointing',
      ]) {
        if (_flag(d, key)) row = _floor(row, d['gesture'] as String, .98);
      }
      var resolved = _top(row);
      var valid = _flag(s, 'valid');
      var reason = s['reason'] as String;
      if (_directions.contains(resolved)) {
        valid = _flag(d, 'valid');
        reason = d['reason'] as String? ?? 'direction geometry accepted';
      }
      if (_flag(s, 'reverse_closed_fist')) {
        valid = false;
        reason = 'fist fingers face away from camera';
      } else if (_flag(s, 'downward_closed_fist')) {
        valid = false;
        reason = 'fist points below horizontal';
      }
      if (_flag(s, 'open_hand_lower_arc')) {
        valid = false;
        reason = 'open hand points below horizontal';
      }
      final singleIndex = _flag(d, 'prominent_index_down') ||
          _flag(d, 'reverse_palm_edge_down') ||
          _flag(d, 'horizontal_pointing');
      final hardNegative = (_flag(s, 'dorsal_hard_negative') && !singleIndex) ||
          resolved == 'dorsal';
      if (hardNegative) {
        valid = false;
        reason = 'Dorsal is a disabled hard negative';
        resolved = 'no_gesture';
        row = List<double>.filled(13, 0)..[6] = 1;
      }
      if (_flag(s, 'stop_inverted_outside_angle') &&
          !singleIndex &&
          !hardNegative) {
        valid = false;
        reason = 'Stop inverted outside -145 to -35 degrees';
      }
      if (_flag(s, 'open_palm_outside_angle') &&
          !singleIndex &&
          !hardNegative) {
        valid = false;
        reason = 'Open Palm outside -160 to -20 degrees';
      }
      final supported = !hardNegative &&
          ((resolved == 'open_palm' && _flag(s, 'palm_geometry_supported')) ||
              (resolved == 'stop_inverted' &&
                  _flag(s, 'stop_inverted_geometry_supported')) ||
              (resolved == 'like' && _flag(s, 'like_geometry_supported')) ||
              (resolved == 'fist' && _flag(s, 'fist_geometry_supported')) ||
              (resolved == 'thumb_down' &&
                  _flag(s, 'thumb_down_geometry_supported')) ||
              (resolved == 'peace' && _flag(s, 'peace_geometry_supported')) ||
              (_directions.contains(resolved) && _flag(d, 'strong_geometry')));
      final expanded =
          supported && _flag(d, 'expanded_thumb_pointing') && knownMass >= .005;
      final horizontal = supported && _flag(d, 'horizontal_pointing');
      final continued = supported && _flag(d, 'continued_direction');
      final directionalRecovery = supported &&
          _directions.contains(resolved) &&
          (resolved == 'down' ||
              knownMass >= .15 ||
              expanded ||
              horizontal ||
              continued);
      final supportedHand = supported &&
          <String>{
            'open_palm',
            'like',
            'fist',
            'thumb_down',
          }.contains(resolved) &&
          (resolved == 'open_palm'
              ? knownMass >= .12 || _flag(s, 'sideways_palm_recovery')
              : knownMass >= .15);
      final peaceRecovery = supported &&
          resolved == 'peace' &&
          (knownMass >= .15 ||
              (_flag(s, 'peace_low_mass_pair_recovery') && knownMass >= .005));
      final stopRecovery = supported &&
          resolved == 'stop_inverted' &&
          _flag(s, 'stop_inverted_geometry_supported');
      final recovery =
          directionalRecovery || supportedHand || peaceRecovery || stopRecovery;
      final sorted = List<double>.of(row)..sort();
      final confidence = sorted.last,
          margin = sorted.last - sorted[sorted.length - 2];
      var hold = 0.0;
      if (knownMass < .70) {
        if (!recovery) {
          valid = false;
          reason = 'outside active gesture vocabulary';
        }
        if (directionalRecovery) hold = expanded || horizontal ? 500 : 400;
        if (supportedHand || stopRecovery) hold = math.max(hold, 450);
        if (peaceRecovery) hold = math.max(hold, 500);
      }
      if (confidence < .70) {
        valid = false;
        reason = 'below confidence floor';
      }
      if (margin < .08) {
        valid = false;
        reason = 'ambiguous probability margin';
      }
      return StaticGeometryDecision(
        label: valid ? resolved : 'no_gesture',
        confidence: confidence,
        margin: margin,
        minimumHoldMs: hold,
        reason: reason,
        accepted: valid,
        probabilities: row,
        knownMass: knownMass,
        recovery: recovery && knownMass < .70,
        diagnostics: <String, Object?>{
          'resolved_label': resolved,
          'geometry_valid': valid,
          'geometry_supported': supported,
          'directional': d,
          'hand_shape': s,
        },
      );
    } on FormatException catch (e) {
      return StaticGeometryDecision(
        label: 'no_gesture',
        confidence: 0,
        margin: 0,
        minimumHoldMs: 0,
        reason: e.message,
        accepted: false,
      );
    }
  }

  static _Resolution _directional(
    List<double> probabilities,
    List<List<double>> p,
    List<List<double>>? w,
    String? preferred,
    double width,
    double height,
  ) {
    final raw = _top(probabilities);
    var worldIndex = false, worldFolded = false, worldContradiction = w != null;
    var projection = 0.0;
    if (w != null) {
      final v = subtract(w[8], w[5]),
          reach = norm(v),
          palm = math.max(distance(w[9], w[0]), epsilon);
      projection = norm(v.sublist(0, 2)) / math.max(reach, epsilon);
      final straightness = reach / math.max(_path(w, 5), epsilon),
          ext = extension(w, 5);
      final others = <double>[
        for (final m in <int>[9, 13, 17]) extension(w, m),
      ];
      final reaches = <double>[
        for (final m in <int>[9, 13, 17]) distance(w[m + 3], w[m]) / palm,
      ];
      worldIndex = ext >= .75 &&
          straightness >= .88 &&
          reach / palm >= .55 &&
          maxValue(others) <= .55 &&
          maxValue(reaches) <= .65;
      final axis = divide(v, math.max(reach, epsilon));
      worldFolded = ext >= .75 &&
          straightness >= .88 &&
          <int>[9, 13, 17].every(
            (m) =>
                extension(w, m) <= .55 &&
                distance(w[m + 3], w[m]) <= .70 * reach &&
                dot(subtract(w[m + 3], w[m + 1]), axis) <= -.04 * reach,
          );
      worldContradiction = reach <= epsilon ||
          (ext < .35 && straightness < .65) ||
          <int>[9, 13, 17].any(
            (m) =>
                extension(w, m) >= .75 &&
                distance(w[m + 3], w[m]) >= .60 * reach &&
                distance(w[m + 3], w[m]) / math.max(_path(w, m), epsilon) >=
                    .90,
          );
    }
    final arched = _archedThumb(p), expanded = _expandedThumb(p) || arched;
    final horizontalShape = _horizontal(p, width, height);
    final horizontal = _flag(horizontalShape, 'valid') &&
        (w == null ||
            worldIndex ||
            (_flag(horizontalShape, 'folded_fingers') && worldFolded) ||
            (_flag(horizontalShape, 'clear_single_index') &&
                !worldContradiction));
    final continued =
        worldIndex && projection < .20 && _directions.contains(preferred);
    if (<String>{'peace', 'rock'}.contains(raw) &&
        !(expanded || worldIndex || horizontal)) {
      return _Resolution(probabilities, <String, Object?>{
        'valid': true,
        'reason': 'multi-finger command retains model label',
        'strong_geometry': false,
      });
    }
    final vector = subtract(p[8], p[5]), reach = norm(vector);
    if (reach < epsilon) {
      if (continued)
        return _Resolution(
          _floor(probabilities, preferred!, .98),
          <String, Object?>{
            'valid': true,
            'gesture': preferred,
            'strong_geometry': true,
            'continued_direction': true,
          },
        );
      return _Resolution(probabilities, <String, Object?>{
        'valid': !_directions.contains(raw),
        'reason': 'degenerate index direction',
      });
    }
    final others = <double>[
      for (final m in <int>[9, 13, 17]) extension(p, m),
    ];
    final indexExt = extension(p, 5);
    if (_directions.contains(raw) && !worldIndex) {
      final peace = indexExt >= .65 &&
          others[0] >= .65 &&
          maxValue(others.sublist(1)) <= .55 &&
          distance(p[12], p[9]) >= .60 * reach;
      final rock = indexExt >= .65 &&
          others[2] >= .65 &&
          maxValue(others.sublist(0, 2)) <= .55 &&
          distance(p[20], p[17]) >= .45 * reach;
      for (final name in <String>['peace', 'rock']) {
        if (name == 'peace' ? peace : rock) {
          final support = probabilities[_names.indexOf(name)];
          if (support >= .15)
            return _Resolution(
              _floor(probabilities, name, .92),
              <String, Object?>{
                'valid': true,
                'reason': 'multi-finger geometry resolves ambiguous direction',
                'strong_geometry': false,
              },
            );
          if (support >= .08)
            return _Resolution(probabilities, <String, Object?>{
              'valid': false,
              'reason': 'more than one finger extended',
              'strong_geometry': false,
            });
        }
      }
    }
    final screen = <double>[vector[0] * width, vector[1] * height];
    final dominance = maxValue(screen.map((x) => x.abs())) / norm(screen);
    final palm = math.max(distance(p[9], p[0]), 1e-6),
        axis = divide(vector, reach);
    final indexStraightness = reach / math.max(_path(p, 5), epsilon);
    final indexLead = minValue(
      <int>[12, 16, 20].map((i) => dot(subtract(p[8], p[i]), axis) / palm),
    );
    final indexReach = reach / palm;
    final otherProjected = maxValue(
      <int>[9, 13, 17].map((m) => dot(subtract(p[m + 3], p[m]), axis) / palm),
    );
    final relative = indexReach - otherProjected;
    final prominence = clip(
      ((distance(p[8], p[0]) -
                      maxValue(
                        <int>[12, 16, 20].map((i) => distance(p[i], p[0])),
                      )) /
                  palm +
              .10) /
          .70,
    );
    final poseScore = clip(
      .50 * indexExt + .35 * (1 - mean(others)) + .15 * prominence,
    );
    var gesture = screen[1].abs() >= screen[0].abs()
        ? (screen[1] > 0 ? 'down' : 'up')
        : (screen[0] > 0 ? 'left' : 'right');
    if (horizontal) gesture = screen[0] > 0 ? 'left' : 'right';
    if (continued) gesture = preferred!;
    final projected = _returnMain(p);
    final tips = <int>[8, 12, 16, 20], mcps = <int>[5, 9, 13, 17];
    final fingerVectors = <List<double>>[
      for (var i = 0; i < 4; i++) subtract(p[tips[i]], p[mcps[i]]),
    ];
    final reaches = fingerVectors.map((v) => norm(v) / palm).toList();
    final units = fingerVectors
        .map((v) => divide(v, math.max(norm(v), epsilon)))
        .toList();
    final exts = <double>[indexExt, ...others];
    final palmDown = (p[9][1] - p[0][1]) / palm;
    final descent = <double>[
      for (var i = 1; i < 4; i++) (p[tips[i]][1] - p[tips[i - 1]][1]) / palm,
    ];
    final innerY = tips.take(3).map((i) => p[i][1]).toList();
    final spread = (maxValue(innerY) - minValue(innerY)) / palm;
    final outerDrop = (p[20][1] - maxValue(innerY)) / palm;
    final outerAdv = reaches[3] / math.max(mean(reaches.take(3)), 1e-6);
    final ys = tips.map((i) => p[i][1]).toList();
    final deepest = ys.indexOf(maxValue(ys));
    final restY = <double>[
      for (var i = 0; i < 4; i++)
        if (i != deepest) ys[i],
    ];
    final edgeSpread = (maxValue(restY) - minValue(restY)) / palm;
    final edgeDrop = (ys[deepest] - maxValue(restY)) / palm;
    final edgeAdv = reaches[deepest] /
        math.max(
          mean(<double>[
            for (var i = 0; i < 4; i++)
              if (i != deepest) reaches[i],
          ]),
          1e-6,
        );
    final minDown = minValue(units.map((u) => u[1]));
    final edge = gesture == 'down' &&
        dominance >= .82 &&
        indexStraightness >= .76 &&
        indexReach >= .55 &&
        indexExt >= .35 &&
        projected['downward_finger_count'] == 3 &&
        _number(projected, 'downward_score') >= .90 &&
        _number(projected, 'parallel_score') >= .88 &&
        _number(projected, 'together_score') >= .50 &&
        _number(projected, 'score') < .76;
    final curled = gesture == 'down' &&
        dominance >= .88 &&
        indexReach >= .40 &&
        palmDown >= .85 &&
        minDown >= .86 &&
        _number(projected, 'parallel_score') >= .92 &&
        descent.every((v) => v >= 0) &&
        maxValue(exts.take(3)) <= .45 &&
        exts[3] >= .75 &&
        reaches[3] >= 1.45 &&
        projected['downward_finger_count'] == 1 &&
        _number(projected, 'score') < .55;
    final clustered = gesture == 'down' &&
        dominance >= .80 &&
        indexReach >= .55 &&
        palmDown >= .80 &&
        minDown >= .80 &&
        _number(projected, 'downward_score') >= .95 &&
        _number(projected, 'parallel_score') >= .84 &&
        _number(projected, 'downward_finger_count') <= 1 &&
        _number(projected, 'score') < .55 &&
        spread <= .45 &&
        outerDrop >= .75 &&
        maxValue(exts.take(3)) <= .45 &&
        reaches[3] >= 1.55 &&
        outerAdv >= 1.55;
    final reverse = gesture == 'down' &&
        dominance >= .78 &&
        palmDown <= .35 &&
        <int>[0, 3].contains(deepest) &&
        minDown >= .75 &&
        _number(projected, 'downward_score') >= .90 &&
        _number(projected, 'parallel_score') >= .80 &&
        edgeSpread <= .90 &&
        edgeDrop >= .55 &&
        reaches[deepest] >= 1.20 &&
        edgeAdv >= 1.30;
    final prominent = gesture == 'down' &&
        dominance >= .90 &&
        indexExt >= .75 &&
        indexStraightness >= .90 &&
        indexReach >= .90 &&
        indexLead >= .40 &&
        relative >= .20 &&
        palmDown >= .60 &&
        deepest == 0 &&
        edgeSpread <= .45 &&
        edgeDrop >= .45 &&
        edgeAdv >= 1.40;
    final sorted = List<double>.of(probabilities)..sort();
    final agrees = raw == gesture &&
        sorted.last >= .70 &&
        sorted.last - sorted[sorted.length - 2] >= .08;
    final shortIndex = indexStraightness >= .88 &&
        indexReach >= .27 &&
        math.max(indexLead, relative) >= .12 &&
        mean(others) <= .35;
    final indexOnly =
        (indexExt >= .45 && indexStraightness >= .70 && indexLead >= .04) ||
            shortIndex;
    final worldDirectional =
        worldIndex && projection >= .20 && indexReach >= .18;
    final supported = (agrees &&
            ((indexExt >= .45 &&
                    indexStraightness >= .80 &&
                    indexLead >= .20) ||
                shortIndex)) ||
        edge ||
        curled ||
        clustered ||
        reverse ||
        prominent ||
        expanded ||
        horizontal ||
        worldDirectional ||
        continued;
    final directional = (dominance >= .73 || horizontal) &&
        (!worldIndex ||
            projection >= .20 ||
            continued ||
            (horizontal && _flag(horizontalShape, 'clear_single_index'))) &&
        ((indexOnly &&
                ((poseScore >= .50 && mean(others) <= .70) || supported)) ||
            edge ||
            curled ||
            clustered ||
            reverse ||
            prominent ||
            horizontal ||
            worldDirectional ||
            continued);
    final strongShort = shortIndex &&
        gesture == 'down' &&
        agrees &&
        sorted.last >= .95 &&
        dominance >= .80;
    final strong = directional &&
        (edge ||
            curled ||
            clustered ||
            reverse ||
            prominent ||
            expanded ||
            horizontal ||
            worldDirectional ||
            continued ||
            (agrees &&
                sorted.last >= .95 &&
                ((indexExt >= .85 &&
                        indexStraightness >= .92 &&
                        indexLead >= .30 &&
                        dominance >= .86) ||
                    strongShort)));
    final details = <String, Object?>{
      'valid': _directions.contains(raw) ? directional : true,
      'gesture': gesture,
      'strong_geometry': strong,
      'model_supported_pose': supported,
      'expanded_thumb_pointing': expanded,
      'arched_thumb_pointing': arched,
      'horizontal_pointing': horizontal,
      'continued_direction': continued,
      'world_index_only': worldIndex,
      'prominent_index_down': prominent,
      'reverse_palm_edge_down': reverse,
      'pose_score': poseScore,
      'axis_dominance': dominance,
      'index_extension': indexExt,
      'index_straightness': indexStraightness,
      'edge_on_down': edge,
      'curled_edge_down': curled,
      'clustered_edge_down': clustered,
    };
    if (!directional) {
      if (_directions.contains(raw))
        details['reason'] = dominance < .73
            ? 'direction is diagonal'
            : 'extend index beyond other fingertips';
      return _Resolution(probabilities, details);
    }
    details['valid'] = true;
    return _Resolution(
      _floor(
        probabilities,
        gesture,
        math.min(.98, .92 + .06 * math.max(0, poseScore - .72) / .28),
      ),
      details,
    );
  }

  static _Resolution _shape(
    List<double> probabilities,
    List<double> model,
    List<List<double>> p,
    List<List<double>>? w,
    String side,
    double handednessScore,
    double width,
    double height,
    bool requireDepth,
  ) {
    var row = List<double>.of(probabilities), raw = _top(row);
    final modelRaw = _top(model);
    final palm = math.max(distance(p[9], p[0]), 1e-6);
    final exts = <double>[
          for (final m in <int>[1, 5, 9, 13, 17]) extension(p, m),
        ],
        non = <double>[
          for (final m in <int>[5, 9, 13, 17]) extension(p, m),
        ];
    final peaceImage = _peace(p), peaceWorld = w == null ? null : _peace(w);
    final peaceSupport = row[10];
    final peaceModel = raw == 'peace' &&
        peaceSupport >= .98 &&
        _number(peaceImage, 'index_extension') >= .75 &&
        _number(peaceImage, 'middle_extension') >= .65 &&
        _number(peaceImage, 'index_reach_ratio') >= .70 &&
        _number(peaceImage, 'index_reach_ratio') <= 1.45 &&
        _number(peaceImage, 'index_forward_ratio') >= .25 &&
        _number(peaceImage, 'tip_separation_ratio') >= .04 &&
        math.min(exts[3], exts[4]) <= .65 &&
        math.max(exts[3], exts[4]) <= .85;
    final peaceGeometry = peaceWorld != null
        ? _flag(peaceWorld, 'valid')
        : (_flag(peaceImage, 'valid') || peaceModel);
    final peacePair = _flag(peaceImage, 'strong_geometry') && peaceGeometry;
    final peaceLow = peacePair &&
        (exts[3] <= .55 || _number(peaceImage, 'ring_reach_ratio') <= .45) &&
        (exts[4] <= .55 || _number(peaceImage, 'pinky_reach_ratio') <= .45);
    final twoUp = modelRaw == 'up' &&
        _flag(peaceImage, 'valid') &&
        distance(p[6], p[10]) / palm >= .05 &&
        (peaceWorld == null || _flag(peaceWorld, 'valid'));
    if (peaceGeometry &&
        (raw == 'peace' ||
            peacePair ||
            twoUp ||
            peaceSupport >= (w != null ? .05 : .15))) {
      row = _floor(row, 'peace', .96);
      raw = 'peace';
    }
    final thumb = subtract(p[4], p[2]),
        thumbNorm = math.max(norm(thumb), epsilon);
    final thumbUp = -thumb[1] / thumbNorm, thumbDown = thumb[1] / thumbNorm;
    final gapPalm = gapScale(p), gap = distance(p[4], p[8]) / gapPalm;
    final otherGaps = <double>[
      for (final t in <int>[12, 16, 20]) distance(p[4], p[t]) / gapPalm,
    ];
    final closestGap = minValue(otherGaps);
    final otherReaches = <double>[
      for (final m in <int>[9, 13, 17]) distance(p[m + 3], p[m]) / gapPalm,
    ];
    final extendedOther = <int>[
      for (var i = 0; i < 3; i++)
        if (exts[i + 2] >= .58 && otherReaches[i] >= .60) i,
    ].length;
    final fist = _fist(p);
    final palmAxis = subtract(averagePoints(p, <int>[5, 9, 13, 17]), p[0]);
    final palmNorm = math.max(norm(palmAxis), epsilon),
        palmUp = -palmAxis[1] / palmNorm;
    final orientation = _openOrientation(p, width, height);
    final tilt = _number(orientation, 'tilt_from_up_degrees').abs();
    final openUpper = tilt <= 90.0001, palmAllowed = tilt <= 70.0001;
    final stop = _stop(p, width, height);
    final surface = _surface(p, side, handednessScore),
        palmar = surface == 'palmar',
        dorsal = surface == 'dorsal';
    final surfaceActive = <String>{'left', 'right'}.contains(side);
    final openSupport =
        <String>{'open_palm', 'stop_inverted'}.contains(modelRaw) &&
            maxValue(model) >= .70;
    final stopDepth = _stopDepth(w);
    final stopGeometry = _flag(stop, 'strong_geometry') && dorsal && stopDepth;
    final stopValid = (openSupport || stopGeometry) &&
        _flag(stop, 'valid') &&
        dorsal &&
        stopDepth;
    final rotation = math.pi / 2 - math.atan2(palmAxis[1], palmAxis[0]);
    final c = math.cos(rotation), sn = math.sin(rotation);
    final aligned = <List<double>>[
      for (final point in p)
        <double>[
          c * (point[0] - p[0][0]) - sn * (point[1] - p[0][1]),
          sn * (point[0] - p[0][0]) + c * (point[1] - p[0][1]),
        ],
    ];
    final dorsalShape = _returnMain(aligned);
    final indexAxis = subtract(p[8], p[5]),
        indexSquared = math.max(dot(indexAxis, indexAxis), epsilon);
    final thumbProjection = dot(subtract(p[4], p[5]), indexAxis) / indexSquared;
    final nearest = add(p[5], multiply(indexAxis, clip(thumbProjection)));
    final thumbGap =
        distance(p[4], nearest) / math.max(distance(p[9], p[0]), epsilon);
    final deliberate = _number(dorsalShape, 'minimum_extension_score') >= .45 &&
        minValue(non) >= .45 &&
        mean(non) >= .68 &&
        thumbGap <= 1.50 &&
        thumbProjection >= 0;
    final palmValid = non.where((x) => x >= .58).length == 4 &&
        mean(non) >= .78 &&
        palmAllowed &&
        (!surfaceActive || palmar);
    final palmGeometry = (palmValid &&
            palmar &&
            minValue(non) >= .65 &&
            mean(non) >= .82 &&
            exts[0] >= .45) ||
        (openSupport &&
            modelRaw == 'stop_inverted' &&
            palmValid &&
            palmar &&
            _flag(stop, 'extended_fingers') &&
            palmAllowed);
    final sidewaysPalm = palmGeometry && palmUp <= .75;
    final thumbLead =
        minValue(<int>[8, 12, 16, 20].map((t) => p[t][1] - p[4][1])) / palm;
    final screen = <List<double>>[
      for (final point in p) <double>[point[0] * width, point[1] * height],
    ];
    final screenPalm = math.max(distance(screen[9], screen[0]), 1e-6);
    final screenThumb = subtract(screen[4], screen[2]);
    final screenThumbUp =
        -screenThumb[1] / math.max(norm(screenThumb), epsilon);
    final screenThumbLead =
        minValue(<int>[8, 12, 16, 20].map((t) => screen[t][1] - screen[4][1])) /
            screenPalm;
    final screenThumbDownLead =
        minValue(<int>[8, 12, 16, 20].map((t) => screen[4][1] - screen[t][1])) /
            screenPalm;
    final nonReach = <double>[
      for (final m in <int>[5, 9, 13, 17]) distance(p[m + 3], p[m]) / palm,
    ];
    final raised = <int>[
      for (var i = 0; i < 4; i++)
        if (non[i] >= .58 && nonReach[i] >= .70) i,
    ].length;
    final likeFinger = thumbLead >= .10 && raised == 0;
    var worldThumb = false, worldFolded = false, worldClosed = false;
    List<double>? worldAxis;
    double? depth, sideAlign, reportedSide, tiltedSide;
    if (w != null) {
      final worldPalm = math.max(distance(w[9], w[0]), epsilon),
          worldThumbVector = subtract(w[4], w[2]);
      final thumbReach = norm(worldThumbVector),
          thumbPath = distance(w[3], w[2]) + distance(w[4], w[3]);
      final thumbUnit = divide(worldThumbVector, math.max(thumbReach, epsilon));
      final worldExt = <double>[
        for (final m in <int>[5, 9, 13, 17]) extension(w, m),
      ];
      final worldReach = <double>[
        for (final m in <int>[5, 9, 13, 17])
          distance(w[m + 3], w[m]) / worldPalm,
      ];
      final worldLead = minValue(
        <int>[
          8,
          12,
          16,
          20,
        ].map((t) => dot(subtract(w[4], w[t]), thumbUnit) / worldPalm),
      );
      worldThumb = extension(w, 1) >= .70 &&
          thumbReach / math.max(thumbPath, epsilon) >= .90 &&
          thumbReach / worldPalm >= .50 &&
          worldLead >= .35 &&
          maxValue(worldExt) <= .45 &&
          maxValue(worldReach) <= .50;
      worldFolded = maxValue(worldExt) <= .80 && mean(worldExt) <= .55;
      worldClosed = maxValue(worldExt) <= .55 && mean(worldExt) <= .38;
      final axis = subtract(averagePoints(w, <int>[5, 9, 13, 17]), w[0]);
      if (norm(axis) >= epsilon) {
        worldAxis = unit(axis);
        depth = worldAxis[2];
        var ref = <double>[
          math.tan(72 * math.pi / 180),
          1 / math.tan(68 * math.pi / 180),
          1,
        ];
        ref = unit(ref);
        sideAlign = math.max(
          dot(worldAxis, ref),
          dot(worldAxis, <double>[-ref[0], ref[1], ref[2]]),
        );
        reportedSide = _viewAlignment(worldAxis, -5, 80);
        tiltedSide = _viewAlignment(worldAxis, -3, 107);
      }
    }
    final sorted = List<double>.of(row)..sort();
    final rotatedThumb = worldThumb &&
        <String>{'like', 'thumb_down'}.contains(raw) &&
        sorted.last >= .60 &&
        sorted.last - sorted[sorted.length - 2] >= .25;
    final rotatedThumbUp =
        rotatedThumb && screenThumbUp >= .25 && screenThumbLead >= .10;
    final rotatedThumbDown =
        rotatedThumb && -screenThumbUp >= .25 && screenThumbDownLead >= .15;
    var likeValid = likeFinger &&
        ((exts[0] >= .60 && mean(non) <= .72 && thumbUp >= .55) ||
            (raw == 'like' &&
                maxValue(row) >= .90 &&
                exts[0] >= .40 &&
                thumbUp >= .20));
    likeValid = likeValid || rotatedThumbUp;
    final compactLike = (maxValue(non) <= .50 && mean(non) <= .38) ||
        (_number(fist, 'maximum_tip_radius_ratio') <= 1.40 &&
            raised == 0 &&
            mean(non) <= .72) ||
        (_number(fist, 'maximum_tip_radius_ratio') <= 1.50 && worldFolded);
    final likeGeometry = (exts[0] >= .70 &&
            _number(fist, 'thumb_radius_ratio') >= 1.10 &&
            compactLike &&
            thumbUp >= .70 &&
            thumbLead >= .25 &&
            palmUp >= -.35) ||
        rotatedThumbUp;
    likeValid = likeValid || likeGeometry;
    final downLead =
        minValue(<int>[8, 12, 16, 20].map((t) => p[4][1] - p[t][1])) / palm;
    final downValid = (exts[0] >= .55 &&
            mean(non) <= .72 &&
            thumbDown >= .30 &&
            downLead >= .15) ||
        (raw == 'thumb_down' &&
            maxValue(row) >= .85 &&
            exts[0] >= .40 &&
            thumbDown >= .25 &&
            downLead >= .15) ||
        rotatedThumbDown;
    final downGeometry = (downValid &&
            exts[0] >= .70 &&
            mean(non) <= .45 &&
            thumbDown >= .70 &&
            downLead >= .25) ||
        rotatedThumbDown;
    final compactFist =
        maxValue(non) <= .55 && mean(non) <= .38 && exts[0] <= .85;
    final worldCompact = compactFist &&
        worldClosed &&
        exts[0] <= .45 &&
        _number(fist, 'maximum_tip_radius_ratio') <= .85 &&
        _number(fist, 'thumb_radius_ratio') <= .85;
    final reportedView = (reportedSide != null &&
            reportedSide >= math.cos(18 * math.pi / 180)) ||
        (tiltedSide != null && tiltedSide >= math.cos(10 * math.pi / 180));
    final reportedGeometry = reportedView &&
        _number(fist, 'maximum_tip_radius_ratio') <= .95 &&
        _number(fist, 'thumb_radius_ratio') <= 1.10 &&
        maxValue(<int>[5, 9, 13, 17].map((m) => distance(p[m + 3], p[m]))) /
                palmNorm <=
            .70 &&
        maxValue(non) <= .62 &&
        mean(non) <= .42 &&
        exts[0] <= .85 &&
        !downValid &&
        !likeValid;
    final sideGeometry = sideAlign != null &&
        sideAlign >= math.cos(18 * math.pi / 180) - 1e-6 &&
        _number(fist, 'maximum_tip_radius_ratio') <= .85 &&
        _number(fist, 'thumb_radius_ratio') <= .85 &&
        maxValue(non) <= .55 &&
        mean(non) <= .38 &&
        !downValid &&
        !likeValid;
    final dorsalDepth = depth != null &&
        depth < -.500001 &&
        dorsal &&
        worldFolded &&
        (_flag(fist, 'strong_geometry') ||
            (compactFist &&
                maxValue(non) <= .42 &&
                mean(non) <= .28 &&
                _number(fist, 'thumb_radius_ratio') <= .85));
    final depthAllowed = (depth != null || !requireDepth) &&
        (depth == null || depth >= -.500001 || dorsalDepth);
    final screenAllowed =
        norm(palmAxis) >= epsilon && -palmAxis[1] / palmNorm >= -1e-6;
    final fistAllowed = depthAllowed && screenAllowed;
    final reverseFist = depth != null &&
        !depthAllowed &&
        (compactFist || _flag(fist, 'strong_geometry')) &&
        worldFolded &&
        _number(fist, 'thumb_radius_ratio') <= .85;
    final downwardFist = !screenAllowed &&
        (compactFist || _flag(fist, 'strong_geometry')) &&
        worldFolded &&
        _number(fist, 'thumb_radius_ratio') <= .85;
    final fistValid = fistAllowed &&
        (compactFist ||
            _flag(fist, 'strong_geometry') ||
            sideGeometry ||
            reportedGeometry);
    final fistGeometry = fistAllowed &&
        (worldCompact ||
            reportedGeometry ||
            sideGeometry ||
            _flag(fist, 'strong_geometry') ||
            (fistValid && maxValue(non) <= .42 && mean(non) <= .28));
    final tucked = fistGeometry &&
        _flag(fist, 'strong_geometry') &&
        _number(fist, 'thumb_radius_ratio') <= .85 &&
        probabilities[8] >= .08;
    final fistRelabel = fistAllowed &&
        (worldCompact ||
            _flag(fist, 'strong_geometry') ||
            sideGeometry ||
            reportedGeometry ||
            (probabilities[8] >= .08 &&
                fistGeometry &&
                (worldFolded ||
                    (maxValue(non) <= .38 &&
                        mean(non) <= .25 &&
                        _number(fist, 'thumb_radius_ratio') <= .85))) ||
            (<String>{'like', 'thumb_down'}.contains(raw) &&
                fistGeometry &&
                exts[0] <= .45 &&
                _number(fist, 'thumb_radius_ratio') >= .75 &&
                probabilities[8] >= .08)) &&
        (raw != 'like' || !likeValid) &&
        (raw != 'thumb_down' || !downValid || tucked);
    final okValid = gap <= .58 &&
        closestGap >= .70 &&
        closestGap - gap >= .25 &&
        extendedOther >= 2 &&
        exts[1] <= .82;
    final validators = <String, bool>{
      'stop_inverted': stopValid,
      'open_palm': palmValid,
      'like': likeValid,
      'dorsal': false,
      'ok': okValid,
      'fist': fistValid,
      'thumb_down': downValid,
      'peace': peaceGeometry,
    };
    if (stopValid) {
      row = List<double>.of(model);
      if (modelRaw == 'open_palm' || stopGeometry)
        row = _floor(row, 'stop_inverted', .96);
      raw = 'stop_inverted';
    } else if (likeGeometry) {
      row = _floor(row, 'like', .96);
      raw = 'like';
    } else if (<String>{'like', 'thumb_down'}.contains(raw) && downGeometry) {
      row = _floor(row, 'thumb_down', .96);
      raw = 'thumb_down';
    } else if (fistRelabel) {
      row = _floor(row, 'fist', .96);
      raw = 'fist';
    } else if (palmGeometry || (palmValid && raw == 'open_palm')) {
      row = _floor(row, 'open_palm', .96);
      raw = 'open_palm';
    }
    final valid = validators[raw] ?? true;
    final reason = valid
        ? 'pose geometry accepted'
        : raw == 'fist' && requireDepth && depth == null
            ? 'fist depth landmarks unavailable'
            : raw == 'fist' && !depthAllowed
                ? 'fist fingers face away from camera'
                : raw == 'fist' && !screenAllowed
                    ? 'fist points below horizontal'
                    : raw == 'stop_inverted'
                        ? 'Stop inverted shape, surface or angle rejected'
                        : '$raw hand shape failed geometry checks';
    return _Resolution(row, <String, Object?>{
      'valid': valid,
      'reason': reason,
      'resolved_gesture': raw,
      'palm_geometry_supported': palmGeometry,
      'sideways_palm_recovery': sidewaysPalm,
      'like_geometry_supported': likeGeometry,
      'thumb_down_geometry_supported': downGeometry,
      'world_thumb_only': worldThumb,
      'rotated_thumb_up_support': rotatedThumbUp,
      'rotated_thumb_down_support': rotatedThumbDown,
      'screen_thumb_up_score': screenThumbUp,
      'screen_thumb_lead_ratio': screenThumbLead,
      'screen_thumb_down_lead_ratio': screenThumbDownLead,
      'fist_geometry_supported': fistGeometry,
      'peace_geometry_supported': peaceGeometry,
      'peace_pair_recovery': peacePair,
      'peace_low_mass_pair_recovery': peaceLow,
      'stop_inverted_geometry_supported': stopGeometry,
      'stop_inverted_valid': stopValid,
      'reverse_closed_fist': reverseFist,
      'downward_closed_fist': downwardFist,
      'open_hand_lower_arc': _flag(stop, 'extended_fingers') && !openUpper,
      'open_palm_outside_angle': _flag(stop, 'extended_fingers') &&
          !dorsal &&
          !worldThumb &&
          !worldFolded &&
          openUpper &&
          !palmAllowed,
      'stop_inverted_outside_angle': _flag(stop, 'extended_fingers') &&
          dorsal &&
          openUpper &&
          !_flag(stop, 'allowed_direction'),
      'dorsal_hard_negative': ((deliberate && !palmar) ||
              (dorsal && minValue(non) >= .45 && mean(non) >= .68)) &&
          !worldThumb &&
          !worldFolded &&
          !stopValid,
      'hand_surface': surface,
      'open_hand_xy_degrees': orientation['xy_degrees'],
      'fist_depth_alignment': depth,
      'fist_depth_allowed': depthAllowed,
      'fist_screen_direction_allowed': screenAllowed,
      'thumb_extension': exts[0],
      'index_extension': exts[1],
      'middle_extension': exts[2],
      'ring_extension': exts[3],
      'pinky_extension': exts[4],
    });
  }

  static Map<String, Object?> _openOrientation(
    List<List<double>> p,
    double width,
    double height,
  ) {
    final vector = subtract(averagePoints(p, <int>[5, 9, 13, 17]), p[0]);
    final axis = <double>[vector[0] * width, vector[1] * height];
    final tilt = math.atan2(axis[0], -axis[1]) * 180 / math.pi;
    var xy = math.atan2(axis[1], axis[0]) * 180 / math.pi;
    if ((xy - 180).abs() <= 1e-4) {
      xy = -180;
    } else if (xy.abs() <= 1e-4) {
      xy = 0;
    }
    return <String, Object?>{
      'xy_degrees': xy,
      'tilt_from_up_degrees': tilt,
      'nondegenerate': norm(axis) > epsilon,
    };
  }

  static Map<String, Object?> _stop(
    List<List<double>> uv,
    double width,
    double height,
  ) {
    final orientation = _openOrientation(uv, width, height);
    final allowed = _flag(orientation, 'nondegenerate') &&
        _number(orientation, 'tilt_from_up_degrees').abs() <= 55.0001;
    final p = <List<double>>[
      for (final point in uv) <double>[point[0] * width, point[1] * height],
    ];
    final exts = <double>[
      for (final m in <int>[5, 9, 13, 17]) extension(p, m),
    ];
    final scale = math.max(distance(p[9], p[0]), epsilon);
    final reaches = <double>[
      for (final m in <int>[5, 9, 13, 17]) distance(p[m + 3], p[m]) / scale,
    ];
    final extended =
        minValue(exts) >= .60 && mean(exts) >= .80 && minValue(reaches) >= .40;
    final strong = extended &&
        minValue(exts) >= .85 &&
        mean(exts) >= .92 &&
        minValue(reaches.take(3)) >= .60 &&
        mean(reaches) >= .70 &&
        extension(p, 1) >= .75 &&
        distance(p[4], p[8]) / scale >= .50;
    return <String, Object?>{
      'allowed_direction': allowed,
      'extended_fingers': extended,
      'valid': extended && allowed,
      'strong_geometry': strong && allowed,
    };
  }

  static bool _stopDepth(List<List<double>>? w) {
    if (w == null) return true;
    for (final m in <int>[5, 9, 13, 17]) {
      final reach = distance(w[m + 3], w[m]), path = _path(w, m);
      if (reach <= epsilon ||
          path <= epsilon ||
          (extension(w, m) < .35 && reach / math.max(path, epsilon) < .65))
        return false;
    }
    return true;
  }

  static String? _surface(
    List<List<double>> p,
    String side,
    double confidence,
  ) {
    if (!<String>{'left', 'right'}.contains(side) ||
        !confidence.isFinite ||
        confidence < .75) return null;
    final a = subtract(p[5], p[0]),
        b = subtract(p[17], p[0]),
        denom = norm(a) * norm(b);
    if (denom < epsilon) return null;
    final score =
        (a[0] * b[1] - a[1] * b[0]) / denom * (side == 'right' ? 1 : -1);
    return score >= .25
        ? 'dorsal'
        : score <= -.25
            ? 'palmar'
            : null;
  }

  static double _viewAlignment(
    List<double> axis,
    double xyDegrees,
    double zxDegrees,
  ) {
    final xy = xyDegrees * math.pi / 180, zx = zxDegrees * math.pi / 180;
    final ref = unit(<double>[
      math.cos(xy).abs(),
      math.sin(xy),
      math.cos(xy) / math.tan(zx),
    ]);
    return dot(<double>[axis[0].abs(), axis[1], axis[2]], ref);
  }

  static bool _expandedThumb(List<List<double>> p) {
    final scale = gapScale(p),
        index = subtract(p[8], p[5]),
        reach = norm(index);
    final axis = divide(index, math.max(reach, epsilon));
    final thumb = subtract(p[4], p[2]), thumbNorm = norm(thumb);
    final otherExt = <double>[
      for (final m in <int>[9, 13, 17]) extension(p, m),
    ];
    final otherReach = <double>[
      for (final m in <int>[9, 13, 17]) distance(p[m + 3], p[m]) / scale,
    ];
    final lead = minValue(
      <int>[12, 16, 20].map((t) => dot(subtract(p[8], p[t]), axis) / scale),
    );
    return axis[0].abs() >= .94 &&
        extension(p, 5) >= .90 &&
        reach / math.max(_path(p, 5), epsilon) >= .96 &&
        reach / scale >= .75 &&
        lead >= .60 &&
        maxValue(otherExt) <= .30 &&
        maxValue(otherReach) <= .50 &&
        extension(p, 1) >= .85 &&
        thumbNorm /
                math.max(
                  distance(p[3], p[2]) + distance(p[4], p[3]),
                  epsilon,
                ) >=
            .90 &&
        distance(p[4], p[0]) / scale >= 1.35 &&
        dot(divide(thumb, math.max(thumbNorm, epsilon)), axis).abs() <= .65 &&
        distance(p[4], p[8]) / scale >= .75;
  }

  static bool _archedThumb(List<List<double>> p) {
    final scale = gapScale(p),
        index = subtract(p[8], p[5]),
        reach = norm(index);
    final axis = divide(index, math.max(reach, epsilon));
    final otherExt = <double>[
      for (final m in <int>[9, 13, 17]) extension(p, m),
    ];
    final otherReach = <double>[
      for (final m in <int>[9, 13, 17]) distance(p[m + 3], p[m]) / scale,
    ];
    final lead = minValue(
      <int>[12, 16, 20].map((t) => dot(subtract(p[8], p[t]), axis) / scale),
    );
    double side(List<double> point) {
      final v = subtract(point, p[5]);
      return (axis[0] * v[1] - axis[1] * v[0]) / scale;
    }

    final base = side(p[2]), sign = base >= 0 ? 1.0 : -1.0;
    final thumb = subtract(p[4], p[2]), thumbNorm = norm(thumb);
    final thumbExt = extension(p, 1),
        thumbStraight = thumbNorm /
            math.max(distance(p[3], p[2]) + distance(p[4], p[3]), epsilon);
    final thumbReach = distance(p[4], p[0]) / scale,
        align = dot(divide(thumb, math.max(thumbNorm, epsilon)), axis).abs();
    final gap = distance(p[4], p[8]) / scale;
    final long = reach / scale >= 1.20 &&
        thumbExt >= .30 &&
        thumbStraight >= .88 &&
        thumbReach >= 1.30 &&
        align <= .87 &&
        gap >= .80 &&
        base.abs() >= .40 &&
        <int>[12, 16, 20].every((t) => sign * side(p[t]) <= -.20);
    final compact = reach / scale >= .95 &&
        maxValue(otherReach) <= .60 &&
        thumbExt >= .10 &&
        thumbStraight >= .78 &&
        thumbReach >= 1.20 &&
        align <= .95 &&
        gap >= .75 &&
        base.abs() >= .30 &&
        <int>[12, 16, 20].every((t) => sign * side(p[t]) <= -.15);
    final sideView = reach / scale >= .70 &&
        maxValue(otherReach) <= .70 &&
        maxValue(
              <int>[
                9,
                13,
                17,
              ].map((m) => dot(subtract(p[m + 3], p[m]), axis) / scale),
            ) <=
            .18 &&
        <int>[9, 13, 17].every(
          (m) => dot(subtract(p[m + 3], p[m + 1]), axis) / scale <= -.04,
        ) &&
        thumbExt >= .10 &&
        thumbStraight >= .75 &&
        thumbReach >= 1.20 &&
        align <= .99 &&
        gap >= .60 &&
        base.abs() >= .18 &&
        sign * side(p[4]) >= .10 &&
        <int>[12, 16, 20].every((t) => sign * side(p[t]) <= -.15);
    return axis[0].abs() >= .94 &&
        extension(p, 5) >= .90 &&
        reach / math.max(_path(p, 5), epsilon) >= .97 &&
        (long || compact || sideView) &&
        lead >= .70 &&
        (maxValue(otherExt) <= .40 || sideView) &&
        maxValue(otherReach) <= .90 &&
        sign * side(p[4]) >= .06;
  }

  static Map<String, Object?> _horizontal(
    List<List<double>> uv,
    double width,
    double height,
  ) {
    final p = <List<double>>[
      for (final x in uv)
        <double>[(x[0] - uv[0][0]) * width, (x[1] - uv[0][1]) * height],
    ];
    final scale = math.max(distance(p[9], p[0]), epsilon),
        vector = subtract(p[8], p[5]),
        reach = norm(vector);
    final axis = divide(vector, math.max(reach, epsilon)),
        normal = <double>[-axis[1], axis[0]];
    final exts = <double>[
      for (final m in <int>[9, 13, 17]) extension(p, m),
    ];
    final reaches = <double>[
      for (final m in <int>[9, 13, 17]) distance(p[m + 3], p[m]) / scale,
    ];
    final returns = <double>[
      for (final m in <int>[9, 13, 17])
        dot(subtract(p[m + 3], p[m + 1]), axis) / scale,
    ];
    final lead = minValue(
      <int>[12, 16, 20].map((t) => dot(subtract(p[8], p[t]), axis) / scale),
    );
    final folded = maxValue(reaches) <= .75 &&
        maxValue(exts) <= .55 &&
        maxValue(returns) <= -.04 &&
        lead >= .55;
    final bases = <double>[
      for (final m in <int>[9, 13, 17])
        dot(subtract(p[m], p[5]), normal) / scale,
    ];
    final tips = <double>[
      for (final m in <int>[9, 13, 17])
        dot(subtract(p[m + 3], p[5]), normal) / scale,
    ];
    final convergence = <double>[
      for (var i = 0; i < 3; i++) bases[i].abs() - tips[i].abs(),
    ];
    final converging =
        (bases.every((x) => x > .15) || bases.every((x) => x < -.15)) &&
            minValue(convergence) >= .18 &&
            maxValue(tips) - minValue(tips) <= .60 &&
            maxValue(reaches) * scale <= .90 * reach &&
            lead >= .25;
    final span = maxValue(<double>[
          for (final a in <int>[5, 9, 13, 17])
            for (final b in <int>[5, 9, 13, 17]) distance(p[a], p[b]),
        ]) /
        scale;
    final collapsed = span <= .20 &&
        exts.where((x) => x <= .45).length >= 2 &&
        maxValue(reaches) * scale <= .90 * reach &&
        lead >= .20;
    final valid = axis[0].abs() >= .94 &&
        reach / scale >= .70 &&
        extension(p, 5) >= .90 &&
        reach / math.max(_path(p, 5), epsilon) >= .97 &&
        (folded || converging || collapsed);
    final clear = valid &&
        folded &&
        reach / scale >= .80 &&
        lead >= .75 &&
        reach / scale >= .70 * span &&
        maxValue(reaches) <= .65 &&
        distance(p[4], p[0]) / scale >= .90 &&
        distance(p[4], p[8]) / scale >= .50;
    return <String, Object?>{
      'valid': valid,
      'folded_fingers': folded,
      'clear_single_index': clear,
    };
  }

  static Map<String, Object?> _returnMain(List<List<double>> p) {
    final mcps = <int>[5, 9, 13, 17];
    final exts = <double>[for (final m in mcps) extension(p, m)];
    final vectors = <List<double>>[
      for (final m in mcps) subtract(p[m + 3], p[m]),
    ];
    final lengths = vectors.map(norm).toList();
    if (lengths.any((x) => x < epsilon))
      return <String, Object?>{
        'score': 0.0,
        'strong_geometry': false,
        'downward_finger_count': 0,
      };
    final units = <List<double>>[
      for (var i = 0; i < 4; i++) divide(vectors[i], lengths[i]),
    ];
    final down = <int>[
      for (var i = 0; i < 4; i++)
        if (units[i][1] >= .50 && exts[i] >= .50) i,
    ].length;
    final meanAxis = <double>[
      for (var a = 0; a < p.first.length; a++) mean(units.map((x) => x[a])),
    ];
    final length = norm(meanAxis);
    final downward =
        length < epsilon ? 0.0 : clip((meanAxis[1] / length - .30) / .60);
    final parallel = length < epsilon ? 0.0 : clip((length - .70) / .30);
    final palm = math.max(distance(p[9], p[0]), 1e-6);
    final gap = mean(<double>[
          distance(p[8], p[12]),
          distance(p[12], p[16]),
          distance(p[16], p[20]),
        ]) /
        palm;
    final together = clip((.78 - gap) / .52);
    final minimum = minValue(exts), average = mean(exts);
    final score = clip(
      (.48 * (.70 * average + .30 * minimum) +
              .27 * downward +
              .15 * together +
              .10 * parallel) *
          (.75 + .25 * together),
    );
    return <String, Object?>{
      'score': score,
      'strong_geometry': score >= .76 && down == 4 && minimum >= .50,
      'downward_finger_count': down,
      'minimum_extension_score': minimum,
      'downward_score': downward,
      'together_score': together,
      'parallel_score': parallel,
    };
  }

  static Map<String, Object?> _peace(List<List<double>> p) {
    final index = subtract(p[8], p[5]), middle = subtract(p[12], p[9]);
    final indexReach = norm(index),
        middleReach = norm(middle),
        palm = math.max(distance(p[9], p[0]), epsilon);
    final midAxis = divide(middle, math.max(middleReach, epsilon));
    final ratio = indexReach / math.max(middleReach, epsilon),
        forward = dot(index, midAxis) / math.max(middleReach, epsilon);
    final separation = distance(p[8], p[12]) / palm;
    final exts = <double>[
      for (final m in <int>[5, 9, 13, 17]) extension(p, m),
    ];
    final ringReach = distance(p[16], p[13]) / math.max(middleReach, epsilon),
        pinkyReach = distance(p[20], p[17]) / math.max(middleReach, epsilon);
    final near = separation <= .35;
    final raised = (exts[2] > .65 && (ringReach >= .65 || !near)) ||
        (exts[3] > .65 && (pinkyReach >= .65 || !near));
    var pair = add(divide(index, math.max(indexReach, epsilon)), midAxis);
    pair = divide(pair, math.max(norm(pair), epsilon));
    final tips = <double>[
      for (final t in <int>[8, 12, 16, 20])
        dot(subtract(p[t], p[0]), pair) / palm,
    ];
    final lead = minValue(tips.take(2)) - maxValue(tips.skip(2));
    final otherForward = <double>[
      dot(subtract(p[16], p[13]), pair) / palm,
      dot(subtract(p[20], p[17]), pair) / palm,
    ];
    final strong = exts[0] >= .80 &&
        exts[1] >= .80 &&
        ratio >= .70 &&
        ratio <= 1.35 &&
        forward >= .65 &&
        math.min(indexReach, middleReach) / palm >= .80 &&
        separation >= .015 &&
        separation <= .55 &&
        distance(p[5], p[9]) / palm >= .05 &&
        distance(p[6], p[10]) / palm >= .05 &&
        lead >= .65 &&
        maxValue(otherForward) <= .60;
    final valid = exts[0] >= .55 &&
        exts[1] >= .45 &&
        !raised &&
        ratio >= .62 &&
        ratio <= 1.45 &&
        forward >= .42 &&
        separation >= .04;
    return <String, Object?>{
      'valid': valid || strong,
      'strong_geometry': strong,
      'index_extension': exts[0],
      'middle_extension': exts[1],
      'index_reach_ratio': ratio,
      'index_forward_ratio': forward,
      'tip_separation_ratio': separation,
      'ring_reach_ratio': ringReach,
      'pinky_reach_ratio': pinkyReach,
    };
  }

  static Map<String, Object?> _fist(List<List<double>> p) {
    final center = averagePoints(p, <int>[5, 9, 13, 17]),
        vector = subtract(averagePoints(p, <int>[5, 9, 13, 17]), p[0]),
        scale = norm(vector);
    if (scale < epsilon)
      return <String, Object?>{
        'strong_geometry': false,
        'maximum_tip_radius_ratio': double.infinity,
        'thumb_radius_ratio': double.infinity,
      };
    final axis = divide(vector, scale);
    final curl = <double>[
      for (final m in <int>[5, 9, 13, 17])
        (dot(subtract(p[m], p[0]), axis) -
                dot(subtract(p[m + 3], p[0]), axis)) /
            scale,
    ];
    final radii = <double>[
      for (final t in <int>[8, 12, 16, 20]) distance(p[t], center) / scale,
    ];
    final thumb = distance(p[4], center) / scale;
    return <String, Object?>{
      'strong_geometry': curl.where((x) => x >= .05).length == 4 &&
          mean(curl) >= .16 &&
          maxValue(radii) <= .85 &&
          thumb <= .85,
      'maximum_tip_radius_ratio': maxValue(radii),
      'thumb_radius_ratio': thumb,
    };
  }

  static double _path(List<List<double>> p, int m) =>
      distance(p[m], p[m + 1]) +
      distance(p[m + 1], p[m + 2]) +
      distance(p[m + 2], p[m + 3]);
  static String _top(List<double> p) => _names[p.indexOf(maxValue(p))];
  static List<double> _normalize(List<double> p) =>
      divide(p, p.fold(0.0, (a, b) => a + b));
  static List<double> _floor(List<double> values, String name, double floor) {
    final i = _names.indexOf(name), p = List<double>.of(values);
    if (p[i] < floor) {
      final other = p.fold(0.0, (a, b) => a + b) - p[i];
      for (var j = 0; j < p.length; j++)
        p[j] = other > 1e-12
            ? p[j] * (1 - floor) / other
            : (1 - floor) / (p.length - 1);
      p[i] = floor;
    }
    return _normalize(p);
  }

  static bool _flag(Map<String, Object?> d, String key) => d[key] == true;
  static double _number(Map<String, Object?> d, String key) =>
      (d[key] as num?)?.toDouble() ?? 0;
}

class _Resolution {
  _Resolution(this.row, this.details);
  final List<double> row;
  final Map<String, Object?> details;
}
