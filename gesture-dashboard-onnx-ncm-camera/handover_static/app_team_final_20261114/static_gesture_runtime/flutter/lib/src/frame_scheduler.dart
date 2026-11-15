import 'dart:async';

/// One active item and at most one pending item. New submissions replace the
/// pending item, and the displaced caller completes with null immediately.
class LatestFrameScheduler<T, R> {
  LatestFrameScheduler({required this.process});

  final Future<R> Function(T value) process;
  _Pending<T, R>? _pending;
  Future<void>? _drainFuture;
  bool _closed = false;
  int _generation = 0;
  int replacedFrames = 0;

  int get pendingCount => _pending == null ? 0 : 1;
  bool get isProcessing => _drainFuture != null;

  Future<R?> submit(T value) {
    if (_closed) throw StateError('Frame scheduler is closed.');
    final previous = _pending;
    if (previous != null) {
      replacedFrames++;
      previous.completer.complete(null);
    }
    final item = _Pending<T, R>(value);
    _pending = item;
    _drainFuture ??= _drain();
    return item.completer.future;
  }

  void reset() {
    _generation++;
    _pending?.completer.complete(null);
    _pending = null;
  }

  Future<void> _drain() async {
    // Avoid assigning an already-completed future to _drainFuture and coalesce
    // bursts of camera callbacks before launching native work.
    await Future<void>.delayed(Duration.zero);
    try {
      while (!_closed && _pending != null) {
        final item = _pending!;
        _pending = null;
        final generation = _generation;
        try {
          final result = await process(item.value);
          item.completer.complete(
            !_closed && generation == _generation ? result : null,
          );
        } catch (error, stack) {
          if (!_closed && generation == _generation) {
            item.completer.completeError(error, stack);
          } else {
            item.completer.complete(null);
          }
        }
      }
    } finally {
      _drainFuture = null;
    }
  }

  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    reset();
    await _drainFuture;
  }
}

class _Pending<T, R> {
  _Pending(this.value);
  final T value;
  final Completer<R?> completer = Completer<R?>();
}
