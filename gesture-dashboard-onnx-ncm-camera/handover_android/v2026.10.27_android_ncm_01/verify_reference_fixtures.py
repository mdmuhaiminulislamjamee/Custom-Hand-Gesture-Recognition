"""Compare exported mobile vectors against an extracted current Python reference."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vectors', type=Path)
    parser.add_argument('--reference-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.reference_root.resolve()
    vectors = json.loads(args.vectors.read_text(encoding='utf-8'))
    for name, expected in vectors['source_sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Fixture/source mismatch: ' + name)
    sys.path.insert(0, str(root))
    import numpy as np
    from backend.config import load_runtime_config
    from backend.geometry import GeometryResolver, landmarks_to_feature
    from backend.hand_selection import gate_arguments
    from backend.runtime import TemporalGate
    config = load_runtime_config()
    resolver = GeometryResolver(config)
    outputs = {}

    def compare(actual, expected):
        for key, value in expected.items():
            if isinstance(value, float):
                assert abs(actual[key] - value) <= 1e-6, (key, actual[key], value)
            else:
                assert actual[key] == value, (key, actual[key], value)

    decisions = 0
    for case in vectors['geometry_cases']:
        points = np.asarray(case['landmarks'], np.float32)
        world = None if case['world_landmarks'] is None else np.asarray(case['world_landmarks'], np.float32)
        row, details = resolver.resolve(np.asarray(case['probabilities']), points,
            handedness=case['handedness'], handedness_confidence=case['handedness_confidence'],
            world_landmarks=world, require_fist_depth=case['require_fist_depth'], image_shape=case['image_shape'])
        np.testing.assert_allclose(landmarks_to_feature(points), case['features'], atol=.0001, rtol=0)
        np.testing.assert_allclose(row, case['expected']['probabilities'], atol=.0001, rtol=0)
        assert details['pose_validation'] == case['expected']['pose_validation'], case['id']
        for temporal in case['expected']['temporal']:
            gate = TemporalGate(config)
            arguments = gate_arguments(details, {'known_gesture_mass': temporal['mass']})
            assert arguments == temporal['gate_arguments']
            for tick, expected in enumerate(temporal['decisions']):
                compare(asdict(gate.update(row, now=10. + tick*.1, **arguments)), expected)
                decisions += 1
        outputs[case['id']] = row
    for sequence in vectors['temporal_transition_cases']:
        gate = TemporalGate(config)
        for step in sequence['steps']:
            result = gate.update(outputs[step['case_id']], now=step['timestamp'], **step['gate_arguments'])
            compare(asdict(result), step['expected'])
            decisions += 1
    print(json.dumps(dict(geometry_cases=len(outputs), temporal_decisions=decisions,
                          temporal_transitions=len(vectors['temporal_transition_cases']), passed=True), indent=2))


if __name__ == '__main__':
    main()
