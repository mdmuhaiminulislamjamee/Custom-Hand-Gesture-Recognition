"""Open Palm accepts XY -160 to -20; Stop inverted keeps -145 to -35."""
import numpy as np
import pytest

from backend.config import CLASS_NAMES, RuntimeConfig
from backend.geometry import GeometryResolver, stop_inverted_geometry
from backend.hand_selection import gate_arguments
from backend.runtime import TemporalGate
from backend.tests.test_stop_inverted import stop_pose

ANGLE_BANDS = {'stop_inverted': (-145., -35.), 'open_palm': (-160., -20.)}


@pytest.mark.parametrize('gesture,xy', [
    (gesture, xy) for gesture, (minimum, maximum) in ANGLE_BANDS.items()
    for xy in (minimum, minimum+.01, (minimum-90.)/2, -90., (maximum-90.)/2, maximum-.01, maximum)
])
@pytest.mark.parametrize('mirror', [False, True])
@pytest.mark.parametrize('shape', [(600, 600, 3), (540, 960, 3), (960, 540, 3)])
@pytest.mark.parametrize('spread', [0., 22.])
def test_both_hands_accept_the_inclusive_symmetric_band(xy, mirror, shape, spread, gesture):
    points, side = stop_pose(xy+90., mirror, spread, shape)
    if gesture == 'open_palm':
        side = 'Left' if side == 'Right' else 'Right'
    config = RuntimeConfig()
    row = np.eye(len(config.class_names))[config.class_to_idx[gesture]]
    row, details = GeometryResolver(config).resolve(row, points, handedness=side,
        handedness_confidence=.99, image_shape=shape)
    assert stop_inverted_geometry(points, shape)['xy_degrees'] == pytest.approx(xy, abs=1e-4)
    assert details['pose_validation']['valid']
    gate = TemporalGate(config)
    args = gate_arguments(details, {'known_gesture_mass': .99})
    for tick in range(4):
        decision = gate.update(row, now=1. + tick*.1, **args)
    assert decision.execute and decision.predicted_gesture == gesture


@pytest.mark.parametrize('gesture,xy', [
    (gesture, xy) for gesture, (minimum, maximum) in ANGLE_BANDS.items()
    for xy in (-180., -170., minimum-.01, maximum+.01, -10., 0., 35., 90., 179.)
])
@pytest.mark.parametrize('mirror', [False, True])
@pytest.mark.parametrize('raw', CLASS_NAMES)
def test_outside_angles_are_no_gesture_even_with_confident_other_outputs(xy, mirror, raw, gesture):
    shape = (540, 960, 3)
    points, side = stop_pose(xy+90., mirror, 22., shape)
    if gesture == 'open_palm':
        side = 'Left' if side == 'Right' else 'Right'
    config = RuntimeConfig()
    row = np.eye(len(config.class_names))[config.class_to_idx[raw]]
    row, details = GeometryResolver(config).resolve(row, points, handedness=side,
        handedness_confidence=.99, image_shape=shape)
    assert not details['pose_validation']['valid']
    assert not details['pose_validation']['stop_inverted_recovery']
    assert not details['pose_validation']['sideways_palm_recovery']
    gate = TemporalGate(config)
    for tick in range(8):
        decision = gate.update(row, now=1. + tick*.1,
            **gate_arguments(details, {'known_gesture_mass': .99}))
        assert not decision.execute and decision.predicted_gesture == 'no_gesture'


@pytest.mark.parametrize('gesture,xy,valid_xy', [
    case for gesture, (minimum, maximum) in ANGLE_BANDS.items()
    for case in ((gesture, minimum-.01, minimum+.1), (gesture, maximum+.01, maximum-.1))
])
@pytest.mark.parametrize('mirror', [False, True])
def test_crossing_either_limit_clears_the_hold_immediately(xy, gesture, mirror, valid_xy):
    config = RuntimeConfig()
    resolver = GeometryResolver(config)
    shape = (540, 960, 3)
    points, side = stop_pose(valid_xy+90., mirror, image_shape=shape)
    if gesture == 'open_palm':
        side = 'Left' if side == 'Right' else 'Right'
    probability = np.eye(len(config.class_names))[config.class_to_idx[gesture]]
    row, details = resolver.resolve(probability, points, handedness=side, handedness_confidence=.99, image_shape=shape)
    gate = TemporalGate(config)
    args = gate_arguments(details, {'known_gesture_mass': .99})
    for tick in range(4):
        decision = gate.update(row, now=1. + tick*.1, **args)
    assert decision.execute
    outside, side = stop_pose(xy+90., mirror, image_shape=shape)
    if gesture == 'open_palm':
        side = 'Left' if side == 'Right' else 'Right'
    bad_row, bad_details = resolver.resolve(probability, outside, handedness=side, handedness_confidence=.99, image_shape=shape)
    decision = gate.update(bad_row, now=1.4, **gate_arguments(bad_details, {'known_gesture_mass': .99}))
    assert not decision.execute and decision.predicted_gesture == 'no_gesture'
    assert not gate.update(row, now=1.5, **args).execute
    assert not gate.update(row, now=1.6, **args).execute
    assert gate.update(row, now=1.71, **args).execute


@pytest.mark.parametrize('xy,allowed', [
    (-180., False), (-160.01, False), (-160., True), (-150., True),
    (-30., True), (-20., True), (-19.99, False), (0., False),
])
@pytest.mark.parametrize('raw', ['open_palm', 'stop_inverted'])
def test_palm_facing_camera_obeys_the_band_despite_surface_label_swaps(xy, raw, allowed):
    shape = (540, 960, 3)
    points, side = stop_pose(xy+90., image_shape=shape)
    config = RuntimeConfig()
    row = np.eye(len(config.class_names))[config.class_to_idx[raw]]
    row, details = GeometryResolver(config).resolve(row, points, handedness='Left' if side == 'Right' else 'Right',
        handedness_confidence=.99, image_shape=shape)
    assert details['pose_validation']['valid'] == allowed
    gate = TemporalGate(config)
    for tick in range(8):
        decision = gate.update(row, now=1. + tick*.1,
            **gate_arguments(details, {'known_gesture_mass': .99}))
        if not allowed:
            assert not decision.execute and decision.predicted_gesture == 'no_gesture'
    assert decision.execute == allowed
    assert decision.predicted_gesture == ('open_palm' if allowed else 'no_gesture')


@pytest.mark.parametrize('xy', [-160.01, -19.99])
@pytest.mark.parametrize('raw', CLASS_NAMES)
def test_missing_handedness_does_not_bypass_open_palm_angle_limit(xy, raw):
    shape = (540, 960, 3)
    points, _ = stop_pose(xy+90., image_shape=shape)
    config = RuntimeConfig()
    row = np.eye(len(config.class_names))[config.class_to_idx[raw]]
    row, details = GeometryResolver(config).resolve(row, points, image_shape=shape)
    assert not details['pose_validation']['valid']
    decision = TemporalGate(config).update(row, now=1.,
        **gate_arguments(details, {'known_gesture_mass': .99}))
    assert not decision.execute and decision.predicted_gesture == 'no_gesture'


@pytest.fixture(scope='module')
def shipped_palm_manager():
    from backend.config import load_runtime_config
    from backend.model_runtime import ModelManager

    manager = ModelManager(load_runtime_config())
    assert manager.ready, manager.errors
    return manager


@pytest.mark.parametrize('xy', [-160., -90., -20.])
@pytest.mark.parametrize('mirror', [False, True])
def test_live_open_palm_pipeline_obeys_both_endpoints(shipped_palm_manager, xy, mirror):
    import threading
    from types import SimpleNamespace

    import cv2
    from backend.model_runtime import InferenceEngine, RuntimeSession
    from backend.tests.test_hand_detection import simulated_detector
    from backend.tests.test_reported_two_hand_palm import world_for_scale

    shape = (540, 960, 3)
    points, dorsal_side = stop_pose(xy+90., mirror, 20., shape)
    side = 'Left' if dorsal_side == 'Right' else 'Right'
    points += [.5, .5] - points.mean(0)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine.model_manager = shipped_palm_manager.config, shipped_palm_manager
    engine.resolver, engine._lock = GeometryResolver(engine.config), threading.Lock()
    engine.detector = simulated_detector()

    def candidates(rgb, view, *, video):
        return [(.1, points.copy(), SimpleNamespace(category_name=side, score=.99),
                 world_for_scale(points, 900.))]

    engine.detector._run_candidates = candidates
    frame = cv2.imencode('.jpg', np.full(shape, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(engine.config)
    results = [engine.process_frame(frame, session, now=1 + tick*.1) for tick in range(15)]
    assert results[-1]['runtime_prediction'] == 'open_palm'
    assert any(result['runtime_action'] == 'Enable / Disable Object Tracking' for result in results)

    outside_xy = -160.01 if xy <= -90. else -19.99
    points, _ = stop_pose(outside_xy+90., mirror, 20., shape)
    points += [.5, .5] - points.mean(0)
    # Fresh acquisition tests the full detector/model/selection path without
    # inheriting filtered joints from the previous, valid camera pose.
    session = RuntimeSession(engine.config)
    results = [engine.process_frame(frame, session, now=3 + tick*.1) for tick in range(15)]
    assert results[-1]['runtime_prediction'] == 'no_gesture'
    assert all(result['runtime_action'] == 'Wait / No Action' for result in results)
