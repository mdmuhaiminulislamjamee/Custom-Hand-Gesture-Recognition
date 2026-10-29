import 'dart:async';
import 'dart:typed_data';

import 'package:hand_detection/hand_detection.dart';

import 'detector/hand_detector_factory.dart';
import 'src/detector_adapter.dart';
import 'src/ncm_client.dart';
import 'src/runtime_api.dart';
import 'src/telemetry.dart';

export 'src/ncm_client.dart'
    show NcmCameraClient, NcmJpegFrame, JlipParser, JlipPacket, jpegDimensions;
export 'src/telemetry.dart';

/// Injectable for deterministic integration tests or an already-owned detector.
abstract interface class NcmStaticDetector {
  String get backend;
  Future<void> initialize();
  Future<List<Hand>> detect(Uint8List jpeg);
  Future<void> resetTracking();
  Future<void> close();
}

/// The source JPEG, raw detector overlay data and gesture decision from one
/// completed frame. Raw detections remain visible when a gesture is rejected.
class NcmStaticPreviewFrame {
  NcmStaticPreviewFrame({
    required this.source,
    required this.width,
    required this.height,
    required List<Hand> detections,
    required this.result,
  }) : detections = List<Hand>.unmodifiable(detections.take(2));

  final NcmJpegFrame source;
  final int width;
  final int height;
  final List<Hand> detections;
  final StaticGestureFrameResult result;
}

class _BundledDetector implements NcmStaticDetector {
  _BundledDetector(this.requestedBackend);
  final MobileInferenceBackend requestedBackend;
  MobileHandDetectorSession? _session;
  @override
  String get backend => _session?.backend ?? 'initializing';
  @override
  Future<void> initialize() async {
    _session ??= await MobileHandDetectorFactory.create(
        requestedBackend: requestedBackend,
        requestedLandmarkModel: MobileLandmarkModelVariant.full,
        detectionConfidence: .50,
        landmarkConfidence: .60);
  }

  @override
  Future<List<Hand>> detect(Uint8List jpeg) => _session!.detector.detect(jpeg);
  @override
  Future<void> resetTracking() async => _session?.detector.resetTracking();
  @override
  Future<void> close() async {
    await _session?.detector.dispose();
    _session = null;
  }
}

/// NCM front end for StaticGestureRuntimeApi. The socket receives continuously;
/// only the newest pending JPEG is retained. JPEG decoding, palm detection and
/// landmarks execute in the bundled detector isolate. Classifier work is awaited
/// before the next detector admission, so no downstream frame queue can grow.
class NcmStaticGestureRuntime {
  NcmStaticGestureRuntime(
      {required this.runtime,
      this.onState,
      this.onError,
      this.onFrame,
      this.onPreview,
      this.onTelemetry,
      this.host = '192.168.50.2',
      this.port = 5000,
      this.sourceAddress,
      this.backend = MobileInferenceBackend.gpu,
      NcmStaticDetector? detector})
      : _detector = detector;
  final StaticGestureRuntimeApi runtime;
  final void Function(String)? onState;
  final void Function(Object)? onError;
  final void Function(NcmJpegFrame, StaticGestureFrameResult)? onFrame;
  final void Function(NcmStaticPreviewFrame)? onPreview;
  final void Function(Map<String, Object>)? onTelemetry;
  final String host;
  final int port;
  final String? sourceAddress;
  final MobileInferenceBackend backend;
  final telemetry = NcmRuntimeTelemetry();
  final _adapter = StaticDetectorAdapter();
  NcmStaticDetector? _detector;
  NcmCameraClient? _client;
  NcmJpegFrame? _pending;
  Timer? _timer;
  Future<void>? _drainFuture;
  bool _running = false;
  bool _closed = false;
  Future<void>? _initializing;
  int _generation = 0;
  double _lastAdmissionMs = double.negativeInfinity;
  String? get detectorBackend => _detector?.backend;

  Future<void> initialize() {
    if (_closed) throw StateError('NCM runtime closed');
    return _initializing ??= _initialize();
  }

  Future<void> _initialize() async {
    await runtime.initialize();
    _detector ??= _BundledDetector(backend);
    await _detector!.initialize();
  }

  Future<void> start() async {
    if (_closed) throw StateError('NCM runtime closed');
    if (_running) return;
    _running = true;
    final epoch = ++_generation;
    try {
      await initialize();
    } catch (error) {
      if (epoch == _generation) _running = false;
      rethrow;
    }
    if (!_running || epoch != _generation || _closed) return;
    runtime.reset();
    _adapter.reset();
    telemetry.startSession();
    _lastAdmissionMs = double.negativeInfinity;
    final client = NcmCameraClient(
        host: host,
        port: port,
        sourceAddress: sourceAddress,
        onState: (state) {
          if (_running) onState?.call(state);
        },
        onError: (error) {
          if (_running) onError?.call(error);
        },
        onDisconnect: () {
          if (_running) _resetStream();
        },
        onFrame: (frame) {
          if (!_running || epoch != _generation) return;
          if (_pending != null) telemetry.replacedFrames++;
          _pending = frame;
          _schedule();
        });
    _client = client;
    await client.start();
  }

  void _resetStream() {
    _pending = null;
    _timer?.cancel();
    _timer = null;
    runtime.reset();
    _adapter.reset();
    // The detector serializes its own isolate messages, so tracking reset waits
    // for an already running detection. That result is discarded via epoch.
    _streamEpoch++;
    unawaited(_resetDetectorTracking());
  }

  int _streamEpoch = 0;

  void _schedule() {
    if (!_running || _drainFuture != null || _timer != null || _pending == null)
      return;
    final now = _client!.monotonicMs;
    final wait = 100 - (now - _lastAdmissionMs);
    if (wait > 0) {
      _timer = Timer(Duration(microseconds: (wait * 1000).ceil()), () {
        _timer = null;
        _schedule();
      });
      return;
    }
    _drainFuture = _drain().whenComplete(() {
      _drainFuture = null;
      _schedule();
    });
  }

  Future<void> _drain() async {
    final frame = _pending;
    _pending = null;
    if (frame == null || !_running) return;
    final epoch = _generation, streamEpoch = _streamEpoch;
    final client = _client!;
    _lastAdmissionMs = client.monotonicMs;
    telemetry.receivedFrameAge
        .add(_lastAdmissionMs - frame.receivedTimestampMs);
    final watch = Stopwatch()..start();
    try {
      final dimensions = jpegDimensions(frame.jpeg);
      final hands = await _detector!.detect(frame.jpeg);
      final detectedMs = watch.elapsedMicroseconds / 1000;
      if (!_running || epoch != _generation || streamEpoch != _streamEpoch)
        return;
      telemetry.detectorAndDecode.add(detectedMs);
      final adapted = _adapter.frame(
          timestampMs: frame.receivedTimestampMs,
          width: dimensions.width,
          height: dimensions.height,
          detections: hands);
      final result = await runtime.processFrame(adapted);
      if (!_running ||
          epoch != _generation ||
          streamEpoch != _streamEpoch ||
          result == null) return;
      telemetry.classificationAndDecision
          .add(watch.elapsedMicroseconds / 1000 - detectedMs);
      telemetry.endToEnd.add(client.monotonicMs - frame.receivedTimestampMs);
      telemetry.processedFrames++;
      onPreview?.call(NcmStaticPreviewFrame(
          source: frame,
          width: dimensions.width,
          height: dimensions.height,
          detections: hands,
          result: result));
      onFrame?.call(frame, result);
      onTelemetry?.call({
        ...telemetry.snapshot,
        'detector_backend': detectorBackend ?? 'unknown',
        'pending_frames': _pending == null ? 0 : 1,
        'crc_errors': client.crcErrors,
        'header_errors': client.headerErrors
      });
    } catch (error) {
      if (_running && epoch == _generation && streamEpoch == _streamEpoch) {
        telemetry.invalidFrames++;
        runtime.invalidateEvidence();
        _adapter.reset();
        onError?.call(error);
      }
    }
  }

  Future<void> stop() async {
    _running = false;
    telemetry.stopSession();
    final epoch = ++_generation;
    _streamEpoch++;
    _timer?.cancel();
    _timer = null;
    _pending = null;
    runtime.reset();
    _adapter.reset();
    final client = _client;
    _client = null;
    await client?.stop();
    await _drainFuture;
    try {
      await _initializing;
    } catch (_) {
      // Closing a partially initialized detector still releases its resources.
    }
    if (epoch != _generation) return;
    await _resetDetectorTracking();
    onState?.call('stopped');
  }

  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    await stop();
    await _detector?.close();
    _detector = null;
    await runtime.close();
  }

  Future<void> _resetDetectorTracking() async {
    try {
      await _detector?.resetTracking();
    } catch (error) {
      if (_running) onError?.call(error);
    }
  }
}
