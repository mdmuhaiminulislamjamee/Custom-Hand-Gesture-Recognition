"""Right projection recovery retains mass and competing-finger safeguards."""
import json
from pathlib import Path
import numpy as np
import pytest
from backend.config import load_runtime_config
from backend.geometry import GeometryResolver
from backend.hand_selection import gate_arguments
from backend.runtime import TemporalGate
from backend.direction_stability import DirectionStability

FIXTURE = Path(__file__).resolve().parents[2] / 'static_gesture_runtime/validation/right_pose_reference.json'
CASES = json.loads(FIXTURE.read_text())['cases']


@pytest.mark.parametrize('case', CASES, ids=lambda c: str(c['id']) + '-' + c['variant'])
def test_right_projection_matches_qualified_mass_and_shape(case):
    config = load_runtime_config()
    world = case['world_landmarks']
    row, details = GeometryResolver(config).resolve(np.asarray(case['probabilities']),
        np.asarray(case['landmarks']), world_landmarks=None if world is None else np.asarray(world),
        handedness=case['physical_hand'], handedness_confidence=.99, image_shape=case['shape'], require_fist_depth=True)
    assert details['directional']['right_retracted_index'] == case['right_retracted_index']
    for expected in case['expected']:
        gate = TemporalGate(config)
        for tick in range(9):
            decision = gate.update(row, now=1 + tick * .1,
                **gate_arguments(details, {'known_gesture_mass': expected['mass']}))
        assert decision.predicted_gesture == expected['label']
        assert (decision.predicted_gesture != 'no_gesture') == expected['accepted']


def test_right_side_view_bridges_one_confidence_dip_without_new_entry_or_action():
    case = next(c for c in CASES if c['variant'] == 'weak_model')
    config = load_runtime_config()
    _, details = GeometryResolver(config).resolve(np.asarray(case['probabilities']),
        np.asarray(case['landmarks']), world_landmarks=np.asarray(case['world_landmarks']),
        handedness=case['physical_hand'], handedness_confidence=.99, image_shape=case['shape'])
    assert not details['pose_validation']['valid']
    assert not details['directional']['right_retracted_index']
    assert details['directional']['continuation_safe']
    state = DirectionStability()
    args = dict(directional=details['directional'], hard_rejection=False, confidence=.94,
        margin=.90, known_mass=1., raw_label='right')
    assert state.update(label='no_gesture', qualified=False, confirmed=False, now=0, **args) is None
    state.update(label='right', qualified=True, confirmed=True, now=1., **args)
    assert state.update(label='no_gesture', qualified=False, confirmed=False, now=1.1, **args) == 'right'
    assert state.update(label='no_gesture', qualified=False, confirmed=False, now=1.2, **args) is None
    assert state.label is None
