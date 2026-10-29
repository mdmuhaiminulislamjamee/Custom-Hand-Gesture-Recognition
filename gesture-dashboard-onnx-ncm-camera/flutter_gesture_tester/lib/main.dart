import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:hand_detection/hand_detection.dart'
    show CameraHandOverlayPainter;
import 'package:static_gesture_runtime/detector/hand_detector_factory.dart';
import 'package:static_gesture_runtime/ncm_static_gesture_runtime.dart';
import 'package:static_gesture_runtime/static_gesture_preview.dart';
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';

import 'models/gesture_result.dart';
import 'services/phone_static_gesture_engine.dart';

enum _VideoSource { phone, ncm }

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  await SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: Color(0xFF07110F),
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );
  runApp(const GestureLabApp());
}

class GestureLabApp extends StatelessWidget {
  const GestureLabApp({super.key});

  @override
  Widget build(BuildContext context) {
    const background = Color(0xFF07110F);
    const mint = Color(0xFF7EF2CC);
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Gesture Lab',
      theme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: mint,
          brightness: Brightness.dark,
          surface: const Color(0xFF10201C),
        ),
        scaffoldBackgroundColor: background,
        fontFamily: 'sans-serif',
        useMaterial3: true,
      ),
      home: const GestureLabScreen(),
    );
  }
}

class GestureLabScreen extends StatefulWidget {
  const GestureLabScreen({super.key});

  @override
  State<GestureLabScreen> createState() => _GestureLabScreenState();
}

class _GestureLabScreenState extends State<GestureLabScreen>
    with WidgetsBindingObserver {
  static const _mint = Color(0xFF7EF2CC);
  static const _amber = Color(0xFFFFC37A);
  static const _frameGap = Duration(milliseconds: 100);

  final List<_GestureEvent> _history = [];

  List<CameraDescription> _cameras = const [];
  CameraController? _camera;
  PhoneStaticGestureEngine? _phoneEngine;
  NcmStaticGestureRuntime? _ncmEngine;
  NcmStaticPreviewFrame? _ncmPreview;
  GestureResult? _result;
  DateTime _lastSubmitted = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime? _lastCompleted;
  String? _error;
  String _phase = 'Loading model';
  String _ncmState = 'stopped';
  String _detectorBackend = 'initializing';
  Map<String, Object> _ncmTelemetry = const {};
  bool _initializing = true;
  bool _switchingSource = false;
  bool _paused = false;
  bool _processing = false;
  double _fps = 0;
  int _cameraIndex = 0;
  int _generation = 0;
  int? _boardSequence;
  _VideoSource _source = _VideoSource.phone;
  String? _lastHistoryLabel;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    unawaited(_boot());
  }

  Future<void> _boot() async {
    try {
      setState(() => _phase = 'Loading TFLite model and hand detector');
      await _preparePhoneEngine();
      if (!mounted) return;
      setState(() => _phase = 'Starting camera');
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        throw StateError('No camera was found on this device.');
      }
      final front = _cameras.indexWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
      );
      _cameraIndex = front >= 0 ? front : 0;
      await _openCamera(_cameraIndex);
      if (!mounted) return;
      setState(() => _initializing = false);
    } on CameraException catch (error) {
      _showFatal(_cameraMessage(error));
    } catch (error) {
      _showFatal(error.toString().replaceFirst('Exception: ', ''));
    }
  }

  Future<void> _preparePhoneEngine() async {
    final previous = _phoneEngine;
    if (previous != null) await previous.close();
    final engine = PhoneStaticGestureEngine(
      backend: MobileInferenceBackend.gpu,
    );
    _phoneEngine = engine;
    await engine.initialize();
    if (_phoneEngine != engine) {
      await engine.close();
      return;
    }
    _detectorBackend = engine.detectorBackend;
  }

  void _showFatal(String message) {
    if (!mounted) return;
    setState(() {
      _error = message;
      _initializing = false;
    });
  }

  Future<void> _openCamera(int index) async {
    final generation = ++_generation;
    final oldCamera = _camera;
    _camera = null;
    if (mounted) setState(() {});
    if (oldCamera != null) {
      if (oldCamera.value.isStreamingImages) await oldCamera.stopImageStream();
      await oldCamera.dispose();
    }

    final controller = CameraController(
      _cameras[index],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.nv21,
    );
    await controller.initialize();
    if (!mounted || generation != _generation) {
      await controller.dispose();
      return;
    }
    _camera = controller;
    _paused = false;
    await _phoneEngine?.reset();
    await controller.startImageStream((image) => _analyze(image, generation));
    if (mounted) setState(() {});
  }

  Future<void> _analyze(CameraImage image, int generation) async {
    final now = DateTime.now();
    if (_paused || _processing || generation != _generation) return;
    if (now.difference(_lastSubmitted) < _frameGap) return;
    _lastSubmitted = now;
    _processing = true;
    try {
      final controller = _camera;
      final engine = _phoneEngine;
      if (controller == null || engine == null) return;
      final frame = await engine.analyze(
        image,
        rotationDegrees: _rotationDegrees(),
        modelInputMirrored:
            Platform.isIOS &&
            controller.description.lensDirection == CameraLensDirection.front,
      );
      if (!mounted || generation != _generation) return;
      _detectorBackend = engine.detectorBackend;
      _commitResult(
        GestureResult.fromStatic(
          frame: frame.result,
          detections: frame.detections,
          frameWidth: frame.width,
          frameHeight: frame.height,
          totalLatencyMs: frame.totalMs,
        ),
      );
    } catch (error) {
      if (!mounted || generation != _generation) return;
      setState(() => _error = error.toString());
    } finally {
      _processing = false;
    }
  }

  int _rotationDegrees() {
    final controller = _camera!;
    const orientationDegrees = <DeviceOrientation, int>{
      DeviceOrientation.portraitUp: 0,
      DeviceOrientation.landscapeLeft: 90,
      DeviceOrientation.portraitDown: 180,
      DeviceOrientation.landscapeRight: 270,
    };
    final device = orientationDegrees[controller.value.deviceOrientation] ?? 0;
    final sensor = controller.description.sensorOrientation;
    if (controller.description.lensDirection == CameraLensDirection.front) {
      return (sensor + device) % 360;
    }
    return (sensor - device + 360) % 360;
  }

  void _commitResult(GestureResult result) {
    final completed = DateTime.now();
    final lastCompleted = _lastCompleted;
    if (lastCompleted != null) {
      final instantaneous =
          1000 /
          completed.difference(lastCompleted).inMilliseconds.clamp(1, 10000);
      _fps = _fps == 0 ? instantaneous : (_fps * .78) + (instantaneous * .22);
    }
    _lastCompleted = completed;
    _remember(result, completed);
    setState(() {
      _result = result;
      _error = null;
    });
  }

  Future<void> _selectSource(_VideoSource source) async {
    if (source == _source || _switchingSource) return;
    setState(() {
      _source = source;
      _switchingSource = true;
      _phase = source == _VideoSource.ncm
          ? 'Connecting to NCM camera'
          : 'Starting phone camera';
      _error = null;
      _result = null;
      _ncmPreview = null;
      _fps = 0;
      _lastCompleted = null;
    });
    try {
      if (source == _VideoSource.ncm) {
        ++_generation;
        final camera = _camera;
        _camera = null;
        if (camera != null) {
          if (camera.value.isStreamingImages) await camera.stopImageStream();
          await camera.dispose();
        }
        await _phoneEngine?.close();
        _phoneEngine = null;
        await _startNcm();
      } else {
        await _closeNcm();
        _ncmState = 'stopped';
        await _preparePhoneEngine();
        await _openCamera(_cameraIndex);
      }
    } catch (error) {
      if (mounted) {
        setState(() {
          _error = error is PlatformException
              ? error.message ?? error.code
              : error.toString();
          _ncmState = source == _VideoSource.ncm ? 'error' : _ncmState;
        });
      }
    } finally {
      if (mounted) setState(() => _switchingSource = false);
    }
  }

  Future<void> _startNcm() async {
    if (!mounted) return;
    setState(() {
      _ncmState = 'connecting';
      _paused = false;
      _error = null;
      _ncmPreview = null;
      _result = null;
    });
    var engine = _ncmEngine;
    if (engine == null) {
      late final NcmStaticGestureRuntime created;
      final runtime = StaticGestureRuntimeApi(
        classifier: StaticTfliteClassifier(model: StaticTfliteModel.float32),
      );
      created = NcmStaticGestureRuntime(
        runtime: runtime,
        backend: MobileInferenceBackend.gpu,
        onState: (state) {
          if (!mounted || _ncmEngine != created) return;
          setState(() {
            _ncmState = state;
            if (state == 'connected') {
              _error = null;
            } else {
              _ncmPreview = null;
              _result = null;
            }
          });
        },
        onError: (error) {
          if (!mounted || _ncmEngine != created) return;
          setState(() {
            _ncmState = 'reconnecting';
            _error = _friendlyNcmError(error);
            _ncmPreview = null;
            _result = null;
          });
        },
        onPreview: (preview) {
          if (!mounted || _ncmEngine != created) return;
          final latency = created.telemetry.endToEnd.snapshot['p50_ms'];
          _boardSequence = preview.source.sequence;
          _detectorBackend = created.detectorBackend ?? 'initializing';
          _ncmPreview = preview;
          _commitResult(
            GestureResult.fromStatic(
              frame: preview.result,
              detections: preview.detections,
              frameWidth: preview.width,
              frameHeight: preview.height,
              totalLatencyMs: latency is num
                  ? latency.toDouble()
                  : preview.result.runtimeMs,
            ),
          );
        },
        onTelemetry: (values) {
          if (!mounted || _ncmEngine != created) return;
          setState(() {
            _ncmTelemetry = values;
            final achieved = values['achieved_fps'];
            if (achieved is num) _fps = achieved.toDouble();
          });
        },
      );
      engine = created;
      _ncmEngine = created;
    }
    await engine.start();
    _detectorBackend = engine.detectorBackend ?? 'initializing';
  }

  Future<void> _closeNcm() async {
    final engine = _ncmEngine;
    _ncmEngine = null;
    _ncmPreview = null;
    _ncmTelemetry = const {};
    await engine?.close();
  }

  Future<void> _restartNcm() async {
    await _ncmEngine?.stop();
    if (mounted && _source == _VideoSource.ncm) await _startNcm();
  }

  String _friendlyNcmError(Object error) {
    final message = error.toString().replaceFirst('SocketException: ', '');
    final lower = message.toLowerCase();
    if (lower.contains('network is unreachable') ||
        lower.contains('no route to host')) {
      return 'Android cannot route to 192.168.50.2 yet. Keep the USB-NCM '
          'camera connected, enable USB Ethernet if the phone asks, then retry. '
          'The app now uses Android\'s real route and no longer requires an '
          'exact 192.168.50.1/30 pre-check.';
    }
    if (lower.contains('connection refused')) {
      return 'The NCM link is visible, but the camera is not listening at '
          '192.168.50.2:5000. Restart the board/camera service, then retry.';
    }
    if (lower.contains('timed out')) {
      return 'NCM connection to 192.168.50.2:5000 timed out. Check the USB '
          'data cable and camera firmware, then retry.';
    }
    return message;
  }

  void _remember(GestureResult result, DateTime time) {
    if (!result.accepted || result.label == _lastHistoryLabel) return;
    _lastHistoryLabel = result.label;
    _history.insert(0, _GestureEvent(result.label, result.confidence, time));
    if (_history.length > 8) _history.removeLast();
  }

  Future<void> _togglePause() async {
    if (_source == _VideoSource.ncm) {
      if (_paused) {
        await _startNcm();
      } else {
        await _ncmEngine?.stop();
        if (mounted) {
          setState(() {
            _paused = true;
            _ncmState = 'stopped';
            _ncmPreview = null;
            _result = null;
          });
        }
      }
      return;
    }
    final camera = _camera;
    if (camera == null) return;
    if (_paused) {
      final generation = _generation;
      await camera.startImageStream((image) => _analyze(image, generation));
    } else if (camera.value.isStreamingImages) {
      await camera.stopImageStream();
    }
    if (mounted) setState(() => _paused = !_paused);
  }

  Future<void> _switchCamera() async {
    if (_cameras.length < 2) return;
    setState(() => _phase = 'Switching camera');
    _cameraIndex = (_cameraIndex + 1) % _cameras.length;
    try {
      await _openCamera(_cameraIndex);
    } on CameraException catch (error) {
      _showFatal(_cameraMessage(error));
    }
  }

  Future<void> _retry() async {
    setState(() {
      _initializing = true;
      _error = null;
      _phase = 'Restarting';
    });
    try {
      ++_generation;
      final camera = _camera;
      _camera = null;
      if (camera != null) {
        if (camera.value.isStreamingImages) await camera.stopImageStream();
        await camera.dispose();
      }
      await _phoneEngine?.close();
      _phoneEngine = null;
      await _closeNcm();
      _source = _VideoSource.phone;
      await _boot();
    } catch (error) {
      _showFatal(error.toString());
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_source == _VideoSource.ncm) {
      if (state == AppLifecycleState.inactive ||
          state == AppLifecycleState.paused) {
        unawaited(_ncmEngine?.stop());
      } else if (state == AppLifecycleState.resumed && !_paused) {
        unawaited(_startNcm());
      }
      return;
    }
    final camera = _camera;
    if (camera == null || !camera.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      ++_generation;
      _camera = null;
      unawaited(camera.dispose());
    } else if (state == AppLifecycleState.resumed && _cameras.isNotEmpty) {
      unawaited(_openCamera(_cameraIndex));
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    ++_generation;
    unawaited(_camera?.dispose());
    unawaited(_phoneEngine?.close());
    unawaited(_ncmEngine?.close());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_initializing || _switchingSource) return _buildLoading();
    if (_source == _VideoSource.phone &&
        (_camera == null || !_camera!.value.isInitialized)) {
      return _buildError();
    }
    return Scaffold(
      body: DecoratedBox(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF0B1C18), Color(0xFF07110F)],
          ),
        ),
        child: SafeArea(
          bottom: false,
          child: CustomScrollView(
            physics: const BouncingScrollPhysics(),
            slivers: [
              SliverToBoxAdapter(child: _buildHeader()),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(16, 7, 16, 4),
                sliver: SliverToBoxAdapter(child: _buildSourcePicker()),
              ),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
                sliver: SliverToBoxAdapter(child: _buildCameraCard()),
              ),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(16, 14, 16, 0),
                sliver: SliverToBoxAdapter(child: _buildResultCard()),
              ),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
                sliver: SliverToBoxAdapter(child: _buildMetrics()),
              ),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
                sliver: SliverToBoxAdapter(child: _buildPredictions()),
              ),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
                sliver: SliverToBoxAdapter(child: _buildHistory()),
              ),
              SliverPadding(
                padding: EdgeInsets.fromLTRB(
                  22,
                  16,
                  22,
                  MediaQuery.paddingOf(context).bottom + 28,
                ),
                sliver: const SliverToBoxAdapter(child: _ScopeNote()),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLoading() {
    return Scaffold(
      body: Container(
        width: double.infinity,
        decoration: const BoxDecoration(
          gradient: RadialGradient(
            center: Alignment(0, -.2),
            radius: .8,
            colors: [Color(0xFF183B31), Color(0xFF07110F)],
          ),
        ),
        child: SafeArea(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 78,
                height: 78,
                decoration: BoxDecoration(
                  color: _mint.withValues(alpha: .12),
                  border: Border.all(color: _mint.withValues(alpha: .35)),
                  borderRadius: BorderRadius.circular(24),
                ),
                child: const Icon(
                  Icons.sign_language_rounded,
                  color: _mint,
                  size: 39,
                ),
              ),
              const SizedBox(height: 24),
              const Text(
                'Gesture Lab',
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.w800),
              ),
              const SizedBox(height: 10),
              Text(_phase, style: const TextStyle(color: Colors.white60)),
              const SizedBox(height: 28),
              const SizedBox(
                width: 148,
                child: LinearProgressIndicator(
                  minHeight: 3,
                  color: _mint,
                  backgroundColor: Color(0xFF1C332D),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildError() {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.camera_alt_outlined, color: _amber, size: 52),
              const SizedBox(height: 20),
              const Text(
                'Camera could not start',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800),
              ),
              const SizedBox(height: 12),
              Text(
                _error ?? 'Check camera permission and try again.',
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white60, height: 1.5),
              ),
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: _retry,
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('Try again'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    final isLive =
        !_paused &&
        _error == null &&
        (_source == _VideoSource.phone || _ncmState == 'connected');
    return Padding(
      padding: const EdgeInsets.fromLTRB(18, 12, 18, 8),
      child: Row(
        children: [
          Container(
            width: 43,
            height: 43,
            decoration: BoxDecoration(
              color: _mint.withValues(alpha: .12),
              borderRadius: BorderRadius.circular(14),
            ),
            child: const Icon(
              Icons.sign_language_rounded,
              color: _mint,
              size: 24,
            ),
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'GESTURE LAB',
                  style: TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w900,
                    letterSpacing: 1.2,
                  ),
                ),
                SizedBox(height: 2),
                Text(
                  'LITERT TWO-HAND TESTER',
                  style: TextStyle(
                    fontSize: 10,
                    color: Colors.white38,
                    letterSpacing: 1.4,
                  ),
                ),
              ],
            ),
          ),
          _StatusPill(
            label: isLive
                ? 'LIVE'
                : _source == _VideoSource.ncm
                ? _ncmState.toUpperCase()
                : 'PAUSED',
            active: isLive,
          ),
        ],
      ),
    );
  }

  Widget _buildSourcePicker() {
    return _Panel(
      child: Padding(
        padding: const EdgeInsets.all(5),
        child: Row(
          children: [
            Expanded(
              child: _SourceButton(
                icon: Icons.phone_android_rounded,
                label: 'Phone camera',
                selected: _source == _VideoSource.phone,
                onTap: () => _selectSource(_VideoSource.phone),
              ),
            ),
            const SizedBox(width: 5),
            Expanded(
              child: _SourceButton(
                icon: Icons.usb_rounded,
                label: 'NCM camera',
                selected: _source == _VideoSource.ncm,
                onTap: () => _selectSource(_VideoSource.ncm),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraCard() {
    final camera = _camera;
    final result = _result;
    final isFront =
        camera?.description.lensDirection == CameraLensDirection.front;
    return AspectRatio(
      aspectRatio: 1,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(28),
        child: DecoratedBox(
          decoration: const BoxDecoration(color: Colors.black),
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (_source == _VideoSource.phone && camera != null)
                FittedBox(
                  fit: BoxFit.cover,
                  clipBehavior: Clip.hardEdge,
                  child: SizedBox(
                    width: camera.value.previewSize!.height,
                    height: camera.value.previewSize!.width,
                    child: CameraPreview(camera),
                  ),
                )
              else
                _buildNcmPreview(),
              DecoratedBox(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    stops: const [0, .35, 1],
                    colors: [
                      Colors.black.withValues(alpha: .5),
                      Colors.transparent,
                      Colors.black.withValues(alpha: .62),
                    ],
                  ),
                ),
              ),
              if (_source == _VideoSource.phone &&
                  result != null &&
                  result.frameWidth > 0 &&
                  result.frameHeight > 0)
                CustomPaint(
                  painter: CameraHandOverlayPainter(
                    hands: result.detections,
                    imageSize: Size(
                      result.frameWidth.toDouble(),
                      result.frameHeight.toDouble(),
                    ),
                    mirrorHorizontally: isFront == true,
                  ),
                ),
              Positioned(
                left: 16,
                top: 15,
                child: _GlassLabel(
                  icon: Icons.center_focus_strong_rounded,
                  text: result?.hasHand == true
                      ? '${result!.hands} HAND${result.hands == 1 ? '' : 'S'} TRACKED'
                      : _source == _VideoSource.ncm
                      ? 'NCM ${_ncmState.toUpperCase()}'
                      : 'CENTER YOUR HAND',
                ),
              ),
              if (_processing)
                const Positioned(
                  right: 17,
                  top: 18,
                  child: SizedBox(
                    width: 15,
                    height: 15,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: _mint,
                    ),
                  ),
                ),
              Positioned(
                left: 14,
                right: 14,
                bottom: 14,
                child: Row(
                  children: [
                    _CameraControl(
                      tooltip: _paused ? 'Resume inference' : 'Pause inference',
                      icon: _paused
                          ? Icons.play_arrow_rounded
                          : Icons.pause_rounded,
                      onTap: _togglePause,
                    ),
                    const SizedBox(width: 9),
                    _GlassLabel(
                      icon: Icons.speed_rounded,
                      text: '${_fps.toStringAsFixed(1)} FPS',
                    ),
                    const Spacer(),
                    if (_source == _VideoSource.ncm)
                      _GlassLabel(
                        icon: Icons.usb_rounded,
                        text: _boardSequence == null
                            ? 'JLIP'
                            : 'SEQ $_boardSequence · DROP ${_ncmTelemetry['replaced_frames'] ?? 0}',
                      )
                    else if (_cameras.length > 1)
                      _CameraControl(
                        tooltip: 'Switch camera',
                        icon: Icons.cameraswitch_rounded,
                        onTap: _switchCamera,
                      ),
                  ],
                ),
              ),
              if (_paused)
                ColoredBox(
                  color: Colors.black.withValues(alpha: .38),
                  child: const Center(
                    child: _GlassLabel(
                      icon: Icons.pause_rounded,
                      text: 'PAUSED',
                    ),
                  ),
                ),
              if (_source == _VideoSource.ncm && _error != null && !_paused)
                Positioned(
                  left: 22,
                  right: 22,
                  top: 112,
                  child: _NcmError(message: _error!, onRetry: _restartNcm),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildNcmPreview() {
    final preview = _ncmPreview;
    if (preview != null) {
      return FittedBox(
        fit: BoxFit.cover,
        clipBehavior: Clip.hardEdge,
        child: SizedBox(
          width: preview.width.toDouble(),
          height: preview.height.toDouble(),
          child: StaticNcmPreview(frame: preview),
        ),
      );
    }
    return ColoredBox(
      color: const Color(0xFF08120F),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.usb_rounded, color: _mint, size: 42),
            const SizedBox(height: 15),
            Text(
              _ncmState == 'connected'
                  ? 'Waiting for JPEG frames'
                  : 'Connecting to 192.168.50.2:5000',
              style: const TextStyle(
                color: Colors.white70,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 7),
            Text(
              'Automatic Android route · $_detectorBackend',
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.white30, fontSize: 11),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    final result = _result;
    final accepted = result?.accepted ?? false;
    final label = result == null
        ? 'Ready'
        : result.hasHand
        ? displayGesture(result.label)
        : 'No hand';
    final accent = accepted ? _mint : _amber;
    return _Panel(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 58,
              height: 58,
              decoration: BoxDecoration(
                color: accent.withValues(alpha: .12),
                border: Border.all(color: accent.withValues(alpha: .22)),
                borderRadius: BorderRadius.circular(18),
              ),
              child: Icon(_gestureIcon(result?.label), color: accent, size: 29),
            ),
            const SizedBox(width: 15),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Flexible(
                        child: Text(
                          label,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.w900,
                            letterSpacing: -.4,
                          ),
                        ),
                      ),
                      if (accepted) ...[
                        const SizedBox(width: 8),
                        const Icon(
                          Icons.check_circle_rounded,
                          color: _mint,
                          size: 18,
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    result == null
                        ? 'Show a static pose to begin'
                        : result.reason,
                    style: TextStyle(
                      color: accent.withValues(alpha: .82),
                      fontSize: 12.5,
                    ),
                  ),
                  if (result?.hasHand == true) ...[
                    const SizedBox(height: 9),
                    Wrap(
                      spacing: 8,
                      runSpacing: 6,
                      children: [
                        _MiniTag(
                          text: 'RAW  ${displayGesture(result!.rawLabel)}',
                        ),
                        if (result.handedness != null)
                          _MiniTag(text: result.handedness!.toUpperCase()),
                        if (result.angle != null)
                          _MiniTag(text: '${result.angle!.round()}°'),
                      ],
                    ),
                  ],
                  if (accepted) ...[
                    const SizedBox(height: 10),
                    Text(
                      result!.action.toUpperCase(),
                      style: const TextStyle(
                        fontSize: 10,
                        color: Colors.white54,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 1.2,
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetrics() {
    final result = _result;
    return Row(
      children: [
        Expanded(
          child: _Metric(
            label: 'CONFIDENCE',
            value: _percent(result?.confidence),
            progress: result?.confidence ?? 0,
          ),
        ),
        const SizedBox(width: 9),
        Expanded(
          child: _Metric(
            label: 'KNOWN MASS',
            value: _percent(result?.knownMass),
            progress: result?.knownMass ?? 0,
          ),
        ),
        const SizedBox(width: 9),
        Expanded(
          child: _Metric(
            label: 'LATENCY',
            value: result == null ? '—' : '${result.latencyMs} ms',
            progress: result == null ? 0 : (result.latencyMs / 250).clamp(0, 1),
          ),
        ),
      ],
    );
  }

  Widget _buildPredictions() {
    final predictions = _result?.predictions.take(4).toList() ?? const [];
    return _Panel(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(18, 17, 18, 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const _SectionTitle(
              title: 'TOP PREDICTIONS',
              subtitle: 'TFLITE FP32 OUTPUT',
            ),
            const SizedBox(height: 14),
            if (predictions.isEmpty)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 12),
                child: Text(
                  'Waiting for a complete hand…',
                  style: TextStyle(color: Colors.white38, fontSize: 13),
                ),
              )
            else
              for (var i = 0; i < predictions.length; i++) ...[
                _PredictionBar(prediction: predictions[i], rank: i + 1),
                if (i != predictions.length - 1) const SizedBox(height: 13),
              ],
          ],
        ),
      ),
    );
  }

  Widget _buildHistory() {
    return _Panel(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(18, 17, 18, 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _SectionTitle(
              title: 'ACCEPTED EVENTS',
              subtitle: '${_history.length} THIS SESSION',
            ),
            const SizedBox(height: 13),
            if (_history.isEmpty)
              const Text(
                'Accepted poses will appear here.',
                style: TextStyle(color: Colors.white38, fontSize: 13),
              )
            else
              SizedBox(
                height: 70,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  itemCount: _history.length,
                  separatorBuilder: (_, _) => const SizedBox(width: 9),
                  itemBuilder: (_, index) =>
                      _HistoryChip(event: _history[index]),
                ),
              ),
          ],
        ),
      ),
    );
  }

  String _percent(double? value) =>
      value == null ? '—' : '${(value * 100).round()}%';

  String _cameraMessage(CameraException error) {
    if (error.code == 'CameraAccessDenied' ||
        error.code == 'CameraAccessDeniedWithoutPrompt') {
      return 'Camera permission is required. Enable it in Android Settings, then retry.';
    }
    return error.description ?? error.code;
  }
}

class _Panel extends StatelessWidget {
  const _Panel({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xFF10201C).withValues(alpha: .92),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: Colors.white.withValues(alpha: .065)),
      ),
      child: child,
    );
  }
}

class _SourceButton extends StatelessWidget {
  const _SourceButton({
    required this.icon,
    required this.label,
    required this.selected,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    const accent = Color(0xFF7EF2CC);
    return Material(
      color: selected ? accent.withValues(alpha: .12) : Colors.transparent,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 11),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 17, color: selected ? accent : Colors.white38),
              const SizedBox(width: 8),
              Text(
                label,
                style: TextStyle(
                  color: selected ? Colors.white : Colors.white54,
                  fontSize: 12,
                  fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _NcmError extends StatelessWidget {
  const _NcmError({required this.message, required this.onRetry});

  final String message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: const Color(0xFF281510).withValues(alpha: .94),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: const Color(0xFFFF9B78).withValues(alpha: .35),
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            message,
            maxLines: 3,
            overflow: TextOverflow.ellipsis,
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 11,
              height: 1.35,
            ),
          ),
          const SizedBox(height: 8),
          TextButton.icon(
            onPressed: onRetry,
            icon: const Icon(Icons.refresh_rounded, size: 16),
            label: const Text('Reconnect'),
          ),
        ],
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  const _StatusPill({required this.label, required this.active});
  final String label;
  final bool active;

  @override
  Widget build(BuildContext context) {
    final color = active ? const Color(0xFF7EF2CC) : const Color(0xFFFFC37A);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 7),
      decoration: BoxDecoration(
        color: color.withValues(alpha: .1),
        borderRadius: BorderRadius.circular(30),
        border: Border.all(color: color.withValues(alpha: .24)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontSize: 10,
              fontWeight: FontWeight.w800,
              letterSpacing: 1,
            ),
          ),
        ],
      ),
    );
  }
}

class _GlassLabel extends StatelessWidget {
  const _GlassLabel({required this.icon, required this.text});
  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: .46),
        borderRadius: BorderRadius.circular(30),
        border: Border.all(color: Colors.white.withValues(alpha: .12)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 13, color: Colors.white70),
          const SizedBox(width: 6),
          Text(
            text,
            style: const TextStyle(
              fontSize: 9.5,
              fontWeight: FontWeight.w700,
              letterSpacing: .7,
            ),
          ),
        ],
      ),
    );
  }
}

class _CameraControl extends StatelessWidget {
  const _CameraControl({
    required this.tooltip,
    required this.icon,
    required this.onTap,
  });
  final String tooltip;
  final IconData icon;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: tooltip,
      child: Material(
        color: Colors.black.withValues(alpha: .52),
        shape: const CircleBorder(side: BorderSide(color: Color(0x24FFFFFF))),
        child: InkWell(
          customBorder: const CircleBorder(),
          onTap: onTap,
          child: SizedBox(
            width: 38,
            height: 38,
            child: Icon(icon, color: Colors.white, size: 20),
          ),
        ),
      ),
    );
  }
}

class _MiniTag extends StatelessWidget {
  const _MiniTag({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: .05),
        borderRadius: BorderRadius.circular(7),
      ),
      child: Text(
        text,
        style: const TextStyle(
          color: Colors.white54,
          fontSize: 9,
          fontWeight: FontWeight.w700,
          letterSpacing: .65,
        ),
      ),
    );
  }
}

class _Metric extends StatelessWidget {
  const _Metric({
    required this.label,
    required this.value,
    required this.progress,
  });
  final String label;
  final String value;
  final double progress;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 12, 12, 11),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              maxLines: 1,
              style: const TextStyle(
                fontSize: 8.5,
                color: Colors.white38,
                fontWeight: FontWeight.w700,
                letterSpacing: .75,
              ),
            ),
            const SizedBox(height: 7),
            Text(
              value,
              maxLines: 1,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 9),
            ClipRRect(
              borderRadius: BorderRadius.circular(3),
              child: LinearProgressIndicator(
                value: progress,
                minHeight: 3,
                backgroundColor: Colors.white.withValues(alpha: .07),
                color: const Color(0xFF7EF2CC),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  const _SectionTitle({required this.title, required this.subtitle});
  final String title;
  final String subtitle;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(
          title,
          style: const TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w800,
            letterSpacing: .9,
          ),
        ),
        const Spacer(),
        Text(
          subtitle,
          style: const TextStyle(
            fontSize: 8.5,
            color: Colors.white30,
            letterSpacing: .8,
          ),
        ),
      ],
    );
  }
}

class _PredictionBar extends StatelessWidget {
  const _PredictionBar({required this.prediction, required this.rank});
  final GesturePrediction prediction;
  final int rank;

  @override
  Widget build(BuildContext context) {
    final color = prediction.disabled
        ? const Color(0xFFFF8E8E)
        : rank == 1
        ? const Color(0xFF7EF2CC)
        : Colors.white38;
    return Column(
      children: [
        Row(
          children: [
            SizedBox(
              width: 19,
              child: Text(
                '$rank',
                style: const TextStyle(color: Colors.white24, fontSize: 10),
              ),
            ),
            Expanded(
              child: Text(
                displayGesture(prediction.label),
                style: TextStyle(
                  color: prediction.disabled ? color : Colors.white70,
                  fontSize: 12,
                  fontWeight: rank == 1 ? FontWeight.w700 : FontWeight.w500,
                ),
              ),
            ),
            if (prediction.disabled)
              const Padding(
                padding: EdgeInsets.only(right: 7),
                child: Text(
                  'DISABLED',
                  style: TextStyle(
                    color: Color(0xFFFF8E8E),
                    fontSize: 7.5,
                    letterSpacing: .6,
                  ),
                ),
              ),
            Text(
              '${(prediction.value * 100).toStringAsFixed(1)}%',
              style: TextStyle(
                color: color,
                fontSize: 11,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        Row(
          children: [
            const SizedBox(width: 19),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(3),
                child: LinearProgressIndicator(
                  value: prediction.value,
                  minHeight: 4,
                  color: color,
                  backgroundColor: Colors.white.withValues(alpha: .055),
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _GestureEvent {
  const _GestureEvent(this.label, this.confidence, this.time);
  final String label;
  final double confidence;
  final DateTime time;
}

class _HistoryChip extends StatelessWidget {
  const _HistoryChip({required this.event});
  final _GestureEvent event;

  @override
  Widget build(BuildContext context) {
    final minute = event.time.minute.toString().padLeft(2, '0');
    return Container(
      width: 132,
      padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF7EF2CC).withValues(alpha: .075),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: const Color(0xFF7EF2CC).withValues(alpha: .12),
        ),
      ),
      child: Row(
        children: [
          Icon(
            _gestureIcon(event.label),
            color: const Color(0xFF7EF2CC),
            size: 20,
          ),
          const SizedBox(width: 9),
          Expanded(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  displayGesture(event.label),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  '${(event.confidence * 100).round()}% · ${event.time.hour}:$minute',
                  style: const TextStyle(color: Colors.white38, fontSize: 9),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ScopeNote extends StatelessWidget {
  const _ScopeNote();

  @override
  Widget build(BuildContext context) {
    return const Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(Icons.science_outlined, color: Colors.white24, size: 17),
        SizedBox(width: 9),
        Expanded(
          child: Text(
            'Full static runtime tester: dynamic-style two-hand tracking, '
            'qualified FP32 TFLite classification, geometry checks and temporal '
            'command latches. Commands are displayed only; this tester does not '
            'control product actions.',
            style: TextStyle(
              color: Colors.white30,
              fontSize: 10.5,
              height: 1.45,
            ),
          ),
        ),
      ],
    );
  }
}

IconData _gestureIcon(String? label) {
  return switch (label) {
    'left' => Icons.arrow_back_rounded,
    'right' => Icons.arrow_forward_rounded,
    'up' => Icons.arrow_upward_rounded,
    'down' => Icons.arrow_downward_rounded,
    'open_palm' => Icons.pan_tool_alt_rounded,
    'like' => Icons.thumb_up_alt_rounded,
    'ok' => Icons.radio_button_checked_rounded,
    'fist' => Icons.back_hand_rounded,
    'thumb_down' => Icons.thumb_down_alt_rounded,
    'peace' => Icons.content_cut_rounded,
    'rock' => Icons.music_note_rounded,
    'stop_inverted' => Icons.front_hand_rounded,
    'no_gesture' => Icons.blur_off_rounded,
    _ => Icons.sign_language_rounded,
  };
}
