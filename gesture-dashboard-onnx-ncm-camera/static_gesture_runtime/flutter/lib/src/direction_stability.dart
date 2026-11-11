/// Bounded continuation for an already confirmed direction. It never authorizes
/// a new command and never refreshes its deadline using a marginal frame.
class DirectionStability {
  static const directions = <String>{'left', 'right', 'up', 'down'};
  String? label;
  double confirmedAtMs = 0;
  int invalidFrames = 0;

  void reset() {
    label = null;
    confirmedAtMs = 0;
    invalidFrames = 0;
  }

  bool boundaryCandidate(
      String candidate, bool qualified, Map<String, Object?> directional) {
    final axes = directional['axis_alignment'] as Map?;
    final alignment = (axes?[label] as num?)?.toDouble() ?? -1;
    final dominance = (directional['axis_dominance'] as num?)?.toDouble() ?? 1;
    return label != null &&
        qualified &&
        directions.contains(candidate) &&
        candidate != label &&
        alignment >= .65 &&
        dominance < .80;
  }

  String? update({
    required String candidate,
    required bool qualified,
    required bool confirmed,
    required double nowMs,
    required Map<String, Object?> directional,
    required bool hardRejection,
    required double confidence,
    required double margin,
    required double knownMass,
    required String rawLabel,
  }) {
    final previous = label;
    final axes = directional['axis_alignment'] as Map?;
    final alignment = (axes?[previous] as num?)?.toDouble() ?? -1;
    final boundary = boundaryCandidate(candidate, qualified, directional);
    final projected = directional['continued_direction'] == true;
    if (projected &&
        qualified &&
        directional['world_index_only'] == true &&
        nowMs - confirmedAtMs >= 0 &&
        nowMs - confirmedAtMs <= 1000.001) {
      invalidFrames = 0;
      return null;
    }
    if (qualified &&
        directions.contains(candidate) &&
        !boundary &&
        !projected) {
      invalidFrames = 0;
      if (confirmed) {
        label = candidate;
        confirmedAtMs = nowMs;
      }
      return null;
    }
    if (hardRejection || (qualified && !directions.contains(candidate))) {
      reset();
      return null;
    }
    if (previous == null) return null;
    invalidFrames++;
    final elapsed = nowMs - confirmedAtMs;
    final withinGrace = elapsed >= 0 && elapsed <= 200.001 && invalidFrames < 2;
    final safe = directional['continuation_safe'] == true &&
        alignment >= .65 &&
        confidence >= .65 &&
        margin >= .05 &&
        knownMass >= .60 &&
        directions.contains(rawLabel);
    if (withinGrace && safe) return previous;
    if (!withinGrace) reset();
    return boundary || projected ? 'no_gesture' : null;
  }
}
