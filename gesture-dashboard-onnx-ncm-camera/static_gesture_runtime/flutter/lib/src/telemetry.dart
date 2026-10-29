/// A bounded window of timings. Contains aggregate milliseconds only.
class TimingWindow {
  TimingWindow({this.capacity = 300});
  final int capacity;
  final List<double> _values = [];
  void add(double milliseconds) {
    if (!milliseconds.isFinite || milliseconds < 0) return;
    _values.add(milliseconds);
    if (_values.length > capacity) _values.removeAt(0);
  }

  void clear() => _values.clear();
  Map<String, Object> get snapshot {
    if (_values.isEmpty) return {'samples': 0};
    final sorted = [..._values]..sort();
    double percentile(double fraction) =>
        sorted[((sorted.length - 1) * fraction).round()];
    return {
      'samples': sorted.length,
      'p50_ms': percentile(.50),
      'p95_ms': percentile(.95),
      'max_ms': sorted.last
    };
  }
}

class NcmRuntimeTelemetry {
  final detectorAndDecode = TimingWindow();
  final classificationAndDecision = TimingWindow();
  final receivedFrameAge = TimingWindow();
  final endToEnd = TimingWindow();
  int processedFrames = 0, replacedFrames = 0, invalidFrames = 0;
  final Stopwatch _session = Stopwatch();
  void startSession() {
    processedFrames = 0;
    replacedFrames = 0;
    invalidFrames = 0;
    detectorAndDecode.clear();
    classificationAndDecision.clear();
    receivedFrameAge.clear();
    endToEnd.clear();
    _session
      ..reset()
      ..start();
  }

  void stopSession() => _session.stop();
  Map<String, Object> get snapshot => {
        'processed_frames': processedFrames,
        'replaced_frames': replacedFrames,
        'invalid_frames': invalidFrames,
        'achieved_fps': _session.elapsedMilliseconds == 0
            ? 0.0
            : processedFrames * 1000 / _session.elapsedMilliseconds,
        'decode_and_detector': detectorAndDecode.snapshot,
        'classification_and_decision': classificationAndDecision.snapshot,
        'receive_to_detector_start': receivedFrameAge.snapshot,
        'receive_to_result': endToEnd.snapshot,
        'scope': 'receiver timings; board capture/USB transit not measured',
      };
}
