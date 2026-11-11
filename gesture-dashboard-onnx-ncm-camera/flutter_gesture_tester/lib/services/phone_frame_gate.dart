/// Drops busy camera callbacks instead of queueing them. Admission uses a
/// monotonic clock so a device wall-clock adjustment cannot stall detection.
class PhoneFrameGate {
  PhoneFrameGate({int Function()? nowMicroseconds}) {
    final clock = Stopwatch()..start();
    _nowMicroseconds = nowMicroseconds ?? () => clock.elapsedMicroseconds;
  }

  static const frameIntervalMicroseconds = 100000;
  late final int Function() _nowMicroseconds;
  int? _lastAdmission;
  int _generation = 0;
  bool _busy = false;

  bool get isProcessing => _busy;

  int? begin() {
    final now = _nowMicroseconds();
    if (_busy ||
        (_lastAdmission != null &&
            now - _lastAdmission! < frameIntervalMicroseconds)) {
      return null;
    }
    _lastAdmission = now;
    _busy = true;
    return _generation;
  }

  bool isCurrent(int ticket) => ticket == _generation;

  /// Invalidates the result already in flight without admitting concurrent
  /// detector work. The next fresh frame can start as soon as that work exits.
  void invalidate() {
    _generation++;
    _lastAdmission = null;
  }

  void finish() => _busy = false;
}
