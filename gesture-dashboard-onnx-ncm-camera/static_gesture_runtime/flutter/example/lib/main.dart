import 'dart:async';

import 'package:flutter/material.dart';
import 'package:static_gesture_runtime/ncm_static_gesture_runtime.dart';
import 'package:static_gesture_runtime/static_gesture_runtime_api.dart';
import 'package:static_gesture_runtime/static_gesture_preview.dart';

void main() => runApp(const MaterialApp(home: NcmTester()));

class NcmTester extends StatefulWidget {
  const NcmTester({super.key});
  @override
  State<NcmTester> createState() => _NcmTesterState();
}

class _NcmTesterState extends State<NcmTester> with WidgetsBindingObserver {
  NcmStaticGestureRuntime? _engine;
  String _state = 'Initializing', _command = 'No command', _error = '';
  NcmStaticPreviewFrame? _preview;
  Map<String, Object> _telemetry = {};
  StaticGestureFrameResult? _result;
  bool _started = false, _busy = false;
  int _generation = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    const int8 = bool.fromEnvironment('STATIC_INT8');
    final runtime = StaticGestureRuntimeApi(
        classifier: StaticTfliteClassifier(
            model: int8 ? StaticTfliteModel.int8 : StaticTfliteModel.float32),
        onCommand: (event) {
          if (mounted)
            setState(() => _command = '${event.hand.name}: ${event.action}');
        });
    _engine = NcmStaticGestureRuntime(
        runtime: runtime,
        host: const String.fromEnvironment('NCM_HOST',
            defaultValue: '192.168.50.2'),
        onState: (state) {
          if (mounted) {
            setState(() {
              _state = state;
              if (state != 'connected') {
                _preview = null;
                _result = null;
              }
            });
          }
        },
        onError: (error) {
          if (mounted) {
            setState(() {
              _error = '$error';
              _preview = null;
              _result = null;
            });
          }
        },
        onPreview: (frame) {
          if (mounted)
            setState(() {
              _preview = frame;
              _result = frame.result;
              _error = '';
            });
        },
        onTelemetry: (values) {
          if (mounted) setState(() => _telemetry = values);
        });
    unawaited(_start());
  }

  Future<void> _start() async {
    if (_busy || _started) return;
    final epoch = ++_generation;
    _busy = true;
    try {
      await _engine!.start();
      if (epoch == _generation) _started = true;
    } catch (error) {
      if (mounted) setState(() => _error = '$error');
    } finally {
      _busy = false;
      if (mounted) setState(() {});
    }
  }

  Future<void> _stop() async {
    _generation++;
    _started = false;
    await _engine?.stop();
    if (mounted)
      setState(() {
        _state = 'stopped';
        _preview = null;
        _result = null;
      });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      unawaited(_stop());
    }
  }

  @override
  void dispose() {
    _generation++;
    WidgetsBinding.instance.removeObserver(this);
    unawaited(_engine?.close());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(title: const Text('Static Gesture · NCM')),
      body: SafeArea(
          child: ListView(padding: const EdgeInsets.all(16), children: [
        Text('$_state · ${_engine?.detectorBackend ?? "loading"}'),
        const SizedBox(height: 12),
        if (_preview != null)
          StaticNcmPreview(frame: _preview!)
        else
          const SizedBox(
              height: 220,
              child: Center(child: Text('Waiting for NCM frames'))),
        Text(_command, style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 8),
        for (final hand in _result?.hands ?? <StaticHandResult>[])
          Text(
              '${hand.hand.name}: ${hand.geometry.label} · ${hand.geometry.reason}'),
        if (_error.isNotEmpty)
          Text(_error, style: const TextStyle(color: Colors.red)),
        const SizedBox(height: 12),
        Text(
            'FPS: ${(_telemetry["achieved_fps"] as num?)?.toStringAsFixed(1) ?? "—"}'),
        Text('Decode + detector: ${_telemetry["decode_and_detector"] ?? "—"}'),
        Text(
            'Classifier + decision: ${_telemetry["classification_and_decision"] ?? "—"}'),
        Text('Receiver → result: ${_telemetry["receive_to_result"] ?? "—"}'),
        Text('Replaced pending frames: ${_telemetry["replaced_frames"] ?? 0}'),
        const SizedBox(height: 12),
        Wrap(spacing: 12, children: [
          FilledButton(
              onPressed: _busy || _started ? null : _start,
              child: const Text('Start')),
          OutlinedButton(
              onPressed: _started ? _stop : null, child: const Text('Stop')),
        ]),
      ])));
}
