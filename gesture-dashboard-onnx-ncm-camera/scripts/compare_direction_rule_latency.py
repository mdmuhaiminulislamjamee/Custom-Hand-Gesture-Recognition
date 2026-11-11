"""Same-host old/new rule timing; excludes camera and physical-phone latency."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import types
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.config import load_runtime_config
from backend.geometry import GeometryResolver
from backend.direction_stability import DirectionStability


def main():
    prefix = subprocess.check_output(['git', 'rev-parse', '--show-prefix'], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(['git', 'show', 'HEAD:' + prefix + 'backend/geometry.py'], cwd=ROOT)
    old = types.ModuleType('backend.geometry_timing_baseline')
    old.__package__ = 'backend'
    sys.modules[old.__name__] = old
    exec(compile(source, '<baseline rules>', 'exec'), old.__dict__)
    reference = json.loads((ROOT / 'static_gesture_runtime/validation/python_reference.json').read_text())
    samples = reference['geometry_cases'][::max(1, len(reference['geometry_cases']) // 80)]
    cases = [(np.asarray(c['probabilities']), np.asarray(c['landmarks']), dict(
        handedness=c['handedness'], handedness_confidence=c['handedness_confidence'],
        world_landmarks=None if c['world_landmarks'] is None else np.asarray(c['world_landmarks']),
        require_fist_depth=c['require_fist_depth'], image_shape=c['image_shape'])) for c in samples]
    config = load_runtime_config()
    resolvers = {'previous': old.GeometryResolver(config), 'current': GeometryResolver(config)}
    for resolver in resolvers.values():
        for p, points, kwargs in cases:
            resolver.resolve(p, points, **kwargs)
    timings = {name: [] for name in resolvers}
    for iteration in range(20):
        for name in (('previous', 'current') if iteration % 2 == 0 else ('current', 'previous')):
            for p, points, kwargs in cases:
                started = time.perf_counter_ns()
                resolvers[name].resolve(p, points, **kwargs)
                timings[name].append((time.perf_counter_ns() - started) / 1e6)
    state = DirectionStability()
    state_times = []
    for n in range(4000):
        started = time.perf_counter_ns()
        state.update(label='left', qualified=True, confirmed=True, now=n*.1,
            directional={}, hard_rejection=False, confidence=.99, margin=.98,
            known_mass=1., raw_label='left')
        state_times.append((time.perf_counter_ns() - started) / 1e6)
    summary = {name: dict(samples=len(values), p50_ms=float(np.median(values)),
                          p95_ms=float(np.percentile(values, 95))) for name, values in timings.items()}
    delta = summary['current']['p95_ms'] - summary['previous']['p95_ms']
    report = dict(scope='Alternating same-host CPU landmark geometry; excludes camera, detector, Flutter and physical-phone latency',
        baseline_source_sha256=hashlib.sha256(source).hexdigest(),
        current_source_sha256=hashlib.sha256((ROOT/'backend/geometry.py').read_bytes()).hexdigest(),
        rules=summary, state_update=dict(p50_ms=float(np.median(state_times)), p95_ms=float(np.percentile(state_times, 95))),
        p95_delta_ms=delta, frame_budget_ms=100, additional_p95_below_one_ms=delta < 1.,
        scheduling_changed=False, extra_detector_or_classifier_passes=0, physical_device_tested=False)
    output = ROOT / 'static_gesture_runtime/validation/direction_rule_latency.json'
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
