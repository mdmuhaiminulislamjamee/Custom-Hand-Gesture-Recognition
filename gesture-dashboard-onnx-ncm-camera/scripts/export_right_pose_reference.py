"""Export Right side-view regressions with mass and competing-finger checks."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.config import load_runtime_config
from backend.geometry import GeometryResolver
from backend.hand_selection import gate_arguments
from backend.runtime import TemporalGate


def main():
    report = json.loads((ROOT / 'artifacts/right_stability_20261111/diagnostic_rules.json').read_text())
    config = load_runtime_config()
    resolver = GeometryResolver(config)
    cases = []
    for record in report['image_replay']['rows']:
        if not (record.get('resolvers') or {}).get('directional', {}).get('right_retracted_index'):
            continue
        points, world = np.asarray(record['landmarks']), np.asarray(record['world_landmarks'])
        side = record['quality']['detector']['selected_handedness']
        for variant in ('right', 'no_world', 'competing_middle', 'left_model', 'peace_model', 'rock_model', 'weak_model'):
            depth = world.copy()
            target = {'left_model': 'left', 'peace_model': 'peace', 'rock_model': 'rock'}.get(variant, 'right')
            probability = np.full(13, .04 / 12)
            probability[config.class_to_idx[target]] = .96
            if variant == 'weak_model':
                probability[:] = .06 / 12
                probability[1] = .94
            if variant == 'competing_middle':
                axis = depth[8] - depth[5]
                for joint, fraction in ((10, 1/3), (11, 2/3), (12, 1.)):
                    depth[joint] = depth[9] + axis * fraction
            if variant == 'no_world':
                depth = None
            row, details = resolver.resolve(probability, points, world_landmarks=depth,
                handedness=side, handedness_confidence=.99, image_shape=(480, 640, 3), require_fist_depth=True)
            expected = []
            for mass in (1., .69, .20):
                gate = TemporalGate(config)
                for tick in range(9):
                    decision = gate.update(row, now=1 + tick * .1,
                        **gate_arguments(details, {'known_gesture_mass': mass}))
                if variant == 'right' and mass == 1.:
                    assert decision.predicted_gesture == 'right' and decision.execute
                elif variant == 'competing_middle' or variant in ('right', 'no_world'):
                    assert decision.predicted_gesture != 'right'
                expected.append(dict(mass=mass, label=decision.predicted_gesture,
                    accepted=decision.predicted_gesture != 'no_gesture'))
            cases.append(dict(id=len(cases), variant=variant, landmarks=points.tolist(),
                world_landmarks=None if depth is None else depth.tolist(), physical_hand=side.lower(),
                probabilities=probability.tolist(), shape=[480, 640, 3],
                right_retracted_index=details['directional']['right_retracted_index'], expected=expected))
        if len(cases) >= 28:
            break
    assert len(cases) == 28
    full = json.loads((ROOT / 'artifacts/right_stability_20261111/first_full_feedback.json').read_text())
    for record in full['image_replay']['rows']:
        if record['actual'] != 'right' or not record.get('world_landmarks'):
            continue
        points, depth = np.asarray(record['landmarks']), np.asarray(record['world_landmarks'])
        side = record['quality']['detector']['selected_handedness']
        probs = np.full(13, .04 / 12)
        probs[1] = .96
        row, details = resolver.resolve(probs, points, world_landmarks=depth,
            handedness=side, handedness_confidence=.99, image_shape=(480, 640, 3))
        if not details['pose_validation']['valid'] or not details['pose_validation']['geometry_supported']:
            continue
        if not record['resolvers']['directional'].get('right_retracted_index'):
            continue
        assert not details['directional']['right_retracted_index']
        expected = []
        for mass in (1., .69, .20):
            gate = TemporalGate(config)
            for tick in range(9):
                decision = gate.update(row, now=1 + tick*.1, **gate_arguments(details, {'known_gesture_mass': mass}))
            assert decision.predicted_gesture == 'right'
            expected.append(dict(mass=mass, label='right', accepted=True))
        cases.append(dict(id=len(cases), variant='existing_path', landmarks=points.tolist(),
            world_landmarks=depth.tolist(), physical_hand=side.lower(), probabilities=probs.tolist(),
            shape=[480, 640, 3], right_retracted_index=False, expected=expected))
        if len(cases) == 32:
            break
    assert len(cases) == 32
    destination = ROOT / 'static_gesture_runtime/validation/right_pose_reference.json'
    destination.write_text(json.dumps(dict(schema_version=1,
        scope='Anonymous saved Right landmarks and synthetic negative variants; training regressions, not independent live accuracy',
        cases=cases), indent=2) + '\n')
    print('Exported', len(cases), 'cases;', len(cases) * 3, 'mass checks')


if __name__ == '__main__':
    main()
