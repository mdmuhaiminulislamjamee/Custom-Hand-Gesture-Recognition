"""Compare old/new FP32 warm invoke time on identical inputs on this host.

This deliberately excludes JPEG transport, detection, Dart/isolate overhead,
display and confirmation. It cannot establish physical-phone camera latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compare(baseline_archive, output):
    with zipfile.ZipFile(baseline_archive) as archive:
        candidates = [name for name in archive.namelist()
                      if name.endswith('/assets/models/gesture_static_float32.tflite')]
        if len(candidates) != 1:
            raise ValueError('Expected exactly one previous FP32 classifier')
        previous_bytes = archive.read(candidates[0])
    current_bytes = (ROOT / 'static_gesture_runtime/flutter/assets/models/gesture_static_float32.tflite').read_bytes()
    with np.load(ROOT / 'models/gesture_online_validation_cache.npz', allow_pickle=False) as cache:
        rows = cache['X'][:100].astype(np.float32)
    runtimes = {}
    for name, data in [('previous', previous_bytes), ('current', current_bytes)]:
        interpreter = Interpreter(model_content=data, num_threads=1)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        if inp['shape'].tolist() != [1, 76]:
            raise ValueError('Unexpected benchmark input')
        for row in rows:
            interpreter.set_tensor(inp['index'], row[None])
            interpreter.invoke()
        runtimes[name] = (interpreter, inp)
    timings = {name: [] for name in runtimes}
    # Alternate order each round to avoid consistently favoring one model.
    for round_index in range(40):
        for name in (('previous', 'current') if round_index % 2 == 0 else ('current', 'previous')):
            interpreter, inp = runtimes[name]
            for row in rows:
                interpreter.set_tensor(inp['index'], row[None])
                start = time.perf_counter_ns()
                interpreter.invoke()
                timings[name].append((time.perf_counter_ns() - start) / 1e6)
    report = dict(schema_version=1,
        scope='Same-host one-thread warm native CPU invoke only; excludes camera, transport, detector, isolate, display and confirmation.',
        physical_device_tested=False, baseline_archive=str(baseline_archive.relative_to(ROOT)),
        baseline_archive_sha256=digest(baseline_archive.read_bytes()),
        previous_model_sha256=digest(previous_bytes), current_model_sha256=digest(current_bytes),
        previous_model_bytes=len(previous_bytes), current_model_bytes=len(current_bytes),
        methodology='100 warmup invokes each; 40 alternating-order rounds of the same 100 feature rows; set_tensor excluded from invoke timer.',
        models={name: dict(samples=len(values), p50_ms=float(np.median(values)),
                           p95_ms=float(np.percentile(values, 95))) for name, values in timings.items()},
        retained_runtime_policy=dict(target_fps=10, admission_interval_ms=100,
            pending_frames_maximum=1, interpreter_initialization='persistent worker',
            strong_confirmation='immediate at confidence >= 0.92 and margin >= 0.25 after mass/geometry qualification',
            ordinary_confirmation='two matching results among latest three',
            fixed_action_cooldown_ms=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'handover_static/static_gesture_runtime_20261031_false_hands.zip')
    parser.add_argument('--output', type=Path, default=ROOT / 'static_gesture_runtime/validation/latency_comparison.json')
    arguments = parser.parse_args()
    print(json.dumps(compare(arguments.baseline.resolve(), arguments.output), indent=2))
