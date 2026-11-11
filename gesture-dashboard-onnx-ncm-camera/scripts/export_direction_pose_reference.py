"""Export anonymous landmark regressions for the directional surface conflict."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.config import load_runtime_config
from backend.geometry import GeometryResolver
from backend.hand_selection import gate_arguments
from backend.runtime import TemporalGate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, default=ROOT / 'artifacts/direction_stability_20261111/candidate_feedback.json')
    args = parser.parse_args()
    config = load_runtime_config()
    resolver = GeometryResolver(config)
    report = json.loads(args.replay.read_text(encoding='utf-8'))
    cases = []
    for record in report['image_replay']['rows']:
        if not record.get('landmarks') or not record.get('world_landmarks'):
            continue
        with Image.open(ROOT / 'feedback/images' / (record['id'] + '.jpg')) as image:
            shape = (image.height, image.width, 3)
        points, world = np.asarray(record['landmarks']), np.asarray(record['world_landmarks'])
        side = record['quality']['detector']['selected_handedness']
        label = record['actual']
        probability = np.full(13, .04 / 12)
        probability[config.class_to_idx[label]] = .96
        _, details = resolver.resolve(probability, points, world_landmarks=world,
            handedness=side, handedness_confidence=.99, image_shape=shape, require_fist_depth=True)
        if not details['directional'].get('model_backed_single_index'):
            continue
        for mirror in (False, True):
            p, w = points.copy(), world.copy()
            target, physical = label, side
            if mirror:
                p[:, 0] = 1 - p[:, 0]
                w[:, 0] *= -1
                target = 'left' if label == 'right' else 'right'
                physical = 'Left' if side == 'Right' else 'Right'
            probs = np.full(13, .04 / 12)
            probs[config.class_to_idx[target]] = .96
            for contradict in (False, True):
                depth = w.copy()
                if contradict:
                    axis = w[8] - w[5]
                    for i, fraction in ((10, 1/3), (11, 2/3), (12, 1)):
                        depth[i] = w[9] + axis * fraction
                row, info = resolver.resolve(probs, p, world_landmarks=depth,
                    handedness=physical, handedness_confidence=.99, image_shape=shape, require_fist_depth=True)
                expected = []
                for mass in (1., .69, .2):
                    gate = TemporalGate(config)
                    for tick in range(9):
                        decision = gate.update(row, now=1 + tick*.1,
                            **gate_arguments(info, {'known_gesture_mass': mass}))
                    if not contradict and mass == 1.:
                        assert decision.execute and decision.predicted_gesture == target
                    else:
                        assert not (decision.execute and decision.predicted_gesture in {'left', 'right'})
                    expected.append(dict(mass=mass, label=decision.predicted_gesture,
                        accepted=decision.predicted_gesture != 'no_gesture'))
                cases.append(dict(id=len(cases), landmarks=p.tolist(), world_landmarks=depth.tolist(),
                    physical_hand=physical.lower(), probabilities=probs.tolist(), shape=list(shape),
                    contradicting_middle=contradict, expected=expected))
        if len(cases) >= 32:
            break
    if not cases:
        raise ValueError('No directional conflict regressions were exported')
    destination = ROOT / 'static_gesture_runtime/validation/direction_pose_reference.json'
    destination.write_text(json.dumps(dict(schema_version=1,
        scope='Anonymous reviewed landmark regressions and synthetic mirror/world contradictions; not independent camera validation',
        cases=cases), indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(cases=len(cases), mass_checks=len(cases)*3, passed=True)))


if __name__ == '__main__':
    main()
