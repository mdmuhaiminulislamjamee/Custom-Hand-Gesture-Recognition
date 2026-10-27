"""Approximate the visible skeleton in the reported 0.01-mass Peace image.

The screenshot does not retain its detector/world vectors. These are overlay
traces and synthetic depth checks, not a replay of the original camera pixels.
"""
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.config import RuntimeConfig, load_runtime_config
from backend.geometry import GeometryResolver, landmarks_to_feature, peace_finger_geometry
from backend.hand_selection import gate_arguments
from backend.model_runtime import InferenceEngine, ModelManager, RuntimeSession
from backend.runtime import TemporalGate
from backend.tests.test_hand_detection import simulated_detector


SHAPE = (575, 960, 3)


def reported_pose(mirror=False):
    pixels = np.asarray([
        [676, 678], [736, 610], [758, 610], [783, 581], [801, 565],
        [734, 545], [769, 492], [791, 459], [810, 432],
        [736, 571], [781, 497], [816, 470], [845, 449],
        [738, 609], [800, 575], [797, 608], [787, 623],
        [728, 650], [797, 635], [797, 639], [793, 647],
    ], np.float32)
    points = (pixels - [0, 114]) / [SHAPE[1], SHAPE[0]]
    if mirror:
        points[:, 0] = 1. - points[:, 0]
    return points.astype(np.float32)


def resolve(points=None, raw='up', world=None):
    config = load_runtime_config()
    points = reported_pose() if points is None else points
    row = np.eye(len(config.class_names))[config.class_to_idx[raw]]
    row, details = GeometryResolver(config).resolve(row, points, world_landmarks=world, image_shape=SHAPE)
    return config, row, details


@pytest.mark.parametrize('mirror', [False, True])
@pytest.mark.parametrize('mass', [.005, .01, .0149, .10])
@pytest.mark.parametrize('raw', ['up', 'peace', 'like', 'rock'])
def test_reported_peace_executes_after_half_second_at_displayed_mass(mirror, mass, raw):
    points = reported_pose(mirror)
    assert peace_finger_geometry(points)['strong_geometry']
    config, row, details = resolve(points, raw)
    assert details['pose_validation']['gesture'] == 'peace'
    gate = TemporalGate(config)
    arguments = gate_arguments(details, {'known_gesture_mass': mass})
    for tick in range(5):
        assert not gate.update(row, now=1. + tick * .1, **arguments).execute
    decision = gate.update(row, now=1.51, **arguments)
    assert decision.execute and decision.predicted_gesture == 'peace'


@pytest.mark.parametrize('mass', [0., .001, .0049])
def test_even_a_strong_pair_needs_the_minimum_vocabulary_mass(mass):
    config, row, details = resolve()
    gate = TemporalGate(config)
    for tick in range(9):
        decision = gate.update(row, now=1. + tick * .1,
                               **gate_arguments(details, {'known_gesture_mass': mass}))
    assert not decision.execute and decision.predicted_gesture == 'no_gesture'


def synthetic_world(points):
    # Synthetic metric depth corroborates the two-finger shape; the original
    # screenshot's world joints cannot be recovered from its three HUD angles.
    pixels = (points - points[0]) * [SHAPE[1], SHAPE[0]] / 2000.
    return np.column_stack((pixels, -.3 * pixels[:, 1]))


@pytest.mark.parametrize('fault', ['folded_index', 'folded_middle', 'raised_ring', 'raised_pinky', 'sideways_ring', 'sideways_pinky', 'duplicate_fingers'])
@pytest.mark.parametrize('with_world', [False, True])
def test_low_mass_recovery_excludes_other_finger_shapes(fault, with_world):
    points = reported_pose()
    if fault.startswith('folded_'):
        base = 5 if fault == 'folded_index' else 9
        points[base+1:base+4] = points[base] + .1 * (points[base+1:base+4] - points[base])
    elif fault == 'duplicate_fingers':
        points[9:13] = points[5:9]
    elif fault.startswith('sideways_'):
        base = 13 if fault == 'sideways_ring' else 17
        pixels = points * [SHAPE[1], SHAPE[0]]
        pair = pixels[12] - pixels[9]
        sideways = np.asarray([-pair[1], pair[0]])
        pixels[base+1:base+4] = [pixels[base] + fraction * sideways for fraction in (.4, .7, 1.)]
        points = pixels / [SHAPE[1], SHAPE[0]]
    else:
        base = 13 if fault == 'raised_ring' else 17
        points[base+1:base+4] = points[base] + (points[6:9] - points[5])
    config, row, details = resolve(points, raw='peace', world=synthetic_world(points) if with_world else None)
    assert not details['pose_validation']['peace_pair_recovery']
    gate = TemporalGate(config)
    for tick in range(9):
        decision = gate.update(row, now=1. + tick * .1,
                               **gate_arguments(details, {'known_gesture_mass': .01}))
    assert not decision.execute and decision.predicted_gesture != 'peace'


@pytest.mark.parametrize('fault', ['folded_index', 'folded_middle', 'raised_ring', 'raised_pinky'])
def test_world_joint_contradictions_veto_a_clear_projected_pair(fault):
    points = reported_pose()
    world = synthetic_world(points)
    base = {'folded_index': 5, 'folded_middle': 9, 'raised_ring': 13, 'raised_pinky': 17}[fault]
    if fault.startswith('folded_'):
        world[base+1:base+4] = world[base] + .1 * (world[base+1:base+4] - world[base])
    else:
        world[base+1:base+4] = world[base] + (world[6:9] - world[5])
    _, _, details = resolve(points, raw='peace', world=world)
    assert not details['pose_validation']['peace_pair_recovery']
    assert not details['pose_validation']['valid']


def test_reviewed_negative_feedback_disables_the_low_mass_pair_recovery():
    config, row, details = resolve()
    args = gate_arguments(details, {'known_gesture_mass': .01, 'online_adapter': {'negative_support': .9}})
    assert not args['peace_pair_recovery']
    gate = TemporalGate(config)
    for tick in range(9):
        decision = gate.update(row, now=1. + tick * .1, **args)
    assert not decision.execute and decision.predicted_gesture == 'no_gesture'


def test_ordinary_peace_does_not_gain_the_strict_pair_exception():
    points = reported_pose()
    points[6:9] = points[5] + .65 * (points[6:9] - points[5])
    config, row, details = resolve(points, raw='peace')
    assert not details['pose_validation']['peace_pair_recovery']
    gate = TemporalGate(config)
    for tick in range(9):
        decision = gate.update(row, now=1. + tick * .1,
                               **gate_arguments(details, {'known_gesture_mass': .01}))
    assert not decision.execute


@pytest.mark.parametrize('mirror', [False, True])
def test_full_command_path_selects_the_reported_peace_and_executes(mirror):
    config = load_runtime_config()
    points = reported_pose(mirror)
    manager = ModelManager(config)
    assert manager.ready, manager.errors
    original = manager.predict_detailed
    def screenshot_mass(feature, model_name=None):
        probabilities, name, elapsed, quality = original(feature, model_name)
        return probabilities, name, elapsed, {**quality, 'known_gesture_mass': .01}
    manager.predict_detailed = screenshot_mass
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine.model_manager = config, manager
    engine.resolver, engine._lock = GeometryResolver(config), threading.Lock()
    engine.detector = simulated_detector()
    engine.detector._run_candidates = lambda *args, **kwargs: [
        (.1, points.copy(), SimpleNamespace(category_name='Left' if mirror else 'Right', score=.99), None)]
    jpeg = cv2.imencode('.jpg', np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    results = [engine.process_frame(jpeg, session, now=1. + tick * .1) for tick in range(15)]
    assert results[-1]['runtime_prediction'] == 'peace'
    assert results[-1]['hands'][0]['eligible']
    assert any(r['runtime_action'] == config.gesture_to_action['peace'] for r in results)
