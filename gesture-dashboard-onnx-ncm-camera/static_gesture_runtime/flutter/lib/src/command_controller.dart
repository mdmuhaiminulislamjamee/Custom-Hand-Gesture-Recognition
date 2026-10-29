/// Canonical classifier order. Index 6 is retained to preserve model parity but
/// must never be dispatched as a command.
const staticGestureLabels = <String>[
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

const staticGestureActions = <String, String>{
  'left': 'Move Left',
  'right': 'Move Right',
  'up': 'Move Up',
  'down': 'Move Down',
  'open_palm': 'Enable / Disable Object Tracking',
  'like': 'Play / Pause',
  'ok': 'Start / Stop Recording',
  'fist': 'Mute',
  'thumb_down': 'Volume Down',
  'peace': 'Turn 90 Degrees',
  'rock': 'Backup',
  'stop_inverted': 'Return to main position',
};

class StaticConfirmationPolicy {
  const StaticConfirmationPolicy({
    this.requiredMatches = 2,
    this.evidenceSize = 3,
    this.strongConfidence = 0.92,
    this.strongMargin = 0.25,
    this.missingHandResetMs = 350,
  });

  final int requiredMatches;
  final int evidenceSize;
  final double strongConfidence;
  final double strongMargin;
  final double missingHandResetMs;

  void validate() {
    if (requiredMatches < 2 ||
        evidenceSize < requiredMatches ||
        !strongConfidence.isFinite ||
        strongConfidence < 0 ||
        strongConfidence > 1 ||
        !strongMargin.isFinite ||
        strongMargin < 0 ||
        strongMargin > 1 ||
        !missingHandResetMs.isFinite ||
        missingHandResetMs <= 0) {
      throw ArgumentError('Invalid static confirmation policy.');
    }
  }
}

class StaticControllerResult {
  const StaticControllerResult({
    required this.label,
    required this.confirmed,
    required this.emit,
    required this.reason,
    required this.matches,
    this.commandReady = false,
  });
  final String label;
  final bool confirmed;
  final bool emit;
  final String reason;
  final int matches;
  final bool commandReady;
}

/// State belongs to one physical hand. Detection loss clears confirmation
/// evidence but intentionally does not rearm the duplicate latch.
class StaticCommandController {
  StaticCommandController({this.policy = const StaticConfirmationPolicy()}) {
    policy.validate();
  }
  final StaticConfirmationPolicy policy;
  final List<String> _evidence = <String>[];
  String? _lastEmitted;
  String? _currentLabel;
  double? _currentSinceMs;
  double? _lastTimestampMs;

  String? get lastEmitted => _lastEmitted;

  StaticControllerResult process({
    required String label,
    required double confidence,
    required double margin,
    required double timestampMs,
    double minimumHoldMs = 0,
    bool allowStrongShortcut = true,
    bool dispatch = true,
  }) {
    if (!timestampMs.isFinite ||
        (_lastTimestampMs != null && timestampMs <= _lastTimestampMs!) ||
        !confidence.isFinite ||
        !margin.isFinite ||
        !minimumHoldMs.isFinite ||
        minimumHoldMs < 0 ||
        (label != 'no_gesture' && !staticGestureActions.containsKey(label))) {
      throw ArgumentError(
          'Invalid command evidence or non-increasing timestamp.');
    }
    _lastTimestampMs = timestampMs;
    if (_currentLabel != label) {
      _currentLabel = label;
      _currentSinceMs = timestampMs;
    }
    _evidence.add(label);
    if (_evidence.length > policy.evidenceSize) _evidence.removeAt(0);
    final matches = _evidence.where((value) => value == label).length;
    final strong = label != 'no_gesture' &&
        allowStrongShortcut &&
        confidence >= policy.strongConfidence &&
        margin >= policy.strongMargin;
    final held = timestampMs - _currentSinceMs! >= minimumHoldMs;
    final confirmed = held && (strong || matches >= policy.requiredMatches);
    if (!confirmed) {
      return StaticControllerResult(
          label: label,
          confirmed: false,
          emit: false,
          reason: held ? 'collecting evidence' : 'geometry recovery hold',
          matches: matches);
    }
    if (label == 'no_gesture') {
      _lastEmitted = null;
      return StaticControllerResult(
          label: label,
          confirmed: true,
          emit: false,
          reason: 'confirmed no_gesture; duplicate latch rearmed',
          matches: matches);
    }
    if (_lastEmitted == label) {
      return StaticControllerResult(
          label: label,
          confirmed: true,
          emit: false,
          reason: 'waiting for confirmed no_gesture',
          matches: matches);
    }
    if (dispatch) _lastEmitted = label;
    return StaticControllerResult(
        label: label,
        confirmed: true,
        emit: dispatch,
        commandReady: true,
        reason: dispatch
            ? (strong ? 'strong confirmation' : 'confirmed consensus')
            : 'cross-hand command suppressed',
        matches: matches);
  }

  /// Claims a confirmed candidate only after cross-hand arbitration selected it.
  void markEmitted(String label) {
    if (!staticGestureActions.containsKey(label))
      throw ArgumentError.value(label);
    _lastEmitted = label;
  }

  void handLost() {
    _evidence.clear();
    _currentLabel = null;
    _currentSinceMs = null;
    _lastTimestampMs = null;
  }

  void reset() {
    handLost();
    _lastEmitted = null;
  }
}
