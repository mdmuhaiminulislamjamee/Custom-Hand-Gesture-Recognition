"""World-axis regressions from the webcam HUD, not raw screenshot replays."""

import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.config import load_runtime_config
from backend.geometry import GeometryResolver
from backend.model_runtime import InferenceEngine, RuntimeSession, hand_plane_angles
from backend.hand_selection import gate_arguments
from backend.runtime import TemporalGate
import backend.model_runtime as runtime_module
from backend.tests.test_geometry import fist_at_xy_angle


@pytest.mark.parametrize('xy', [-170, -120, -86, -45, 0, 45, 90, 135])
@pytest.mark.parametrize('mirror', [False, True])
def test_camera_facing_closed_fist_recovers_from_thumb_down_score(xy, mirror):
    config = load_runtime_config()
    points, world = fist_at_xy_angle(xy, .17, mirror=mirror)
    row = np.zeros(len(config.class_names))
    row[config.class_to_idx['thumb_down']] = .82
    row[config.class_to_idx['fist']] = .18

    resolved, details = GeometryResolver(config).resolve(
        row, points, world_landmarks=world, require_fist_depth=True,
    )

    if xy <= 0:
        assert config.class_names[int(resolved.argmax())] == 'fist'
        assert details['pose_validation']['valid'] is True
        assert details['pose_validation']['gesture'] == 'fist'
    else:
        assert details['pose_validation']['valid'] is False
        decision = TemporalGate(config).update(resolved, now=1., **gate_arguments(
            details, dict(known_gesture_mass=1.)))
        assert decision.predicted_gesture == 'no_gesture'


@pytest.mark.parametrize('mirror', [False, True])
@pytest.mark.parametrize('model_label', ['fist', 'thumb_down', 'down'])
def test_opposite_facing_closed_fist_rejects_other_model_labels(mirror, model_label):
    config = load_runtime_config()
    depth_ratio = np.cos(np.deg2rad(-83)) / np.tan(np.deg2rad(170))
    points, world = fist_at_xy_angle(-83, depth_ratio, mirror=mirror)
    row = np.zeros(len(config.class_names))
    row[config.class_to_idx[model_label]] = .96
    row[config.class_to_idx['fist']] = max(row[config.class_to_idx['fist']], .04)

    _, details = GeometryResolver(config).resolve(
        row, points, handedness='Right' if mirror else 'Left',
        handedness_confidence=.99, world_landmarks=world,
        require_fist_depth=True,
    )

    assert details['hand_shape']['reverse_closed_fist'] is True
    assert details['pose_validation']['valid'] is False
    assert details['pose_validation']['reason'] == 'fist fingers face away from the camera'


@pytest.mark.parametrize('mirror', [False, True])
def test_reported_dorsal_fist_recovers_beyond_30_degree_forward_pitch(mirror):
    config = load_runtime_config()
    xy, zx = -75, 159
    depth_ratio = np.cos(np.deg2rad(xy)) / np.tan(np.deg2rad(zx))
    points, world = fist_at_xy_angle(xy, depth_ratio, mirror=mirror)
    if not mirror:
        angles = hand_plane_angles(world)
        assert (angles['xy'], angles['yz'], angles['zx']) == pytest.approx(
            (-75, -146, 159), abs=2,
        )
    row = np.zeros(len(config.class_names))
    row[config.class_to_idx['fist']] = .995
    dorsal_hand = 'Left' if mirror else 'Right'
    palmar_hand = 'Right' if mirror else 'Left'

    for hand, expected in ((dorsal_hand, True), (palmar_hand, False)):
        resolved, details = GeometryResolver(config).resolve(
            row, points, handedness=hand, handedness_confidence=.99,
            world_landmarks=world, require_fist_depth=True,
        )
        shape = details['hand_shape']
        assert shape['fist_depth_alignment'] < -.17
        assert shape['fist_toward_camera_pitch_degrees'] > 30
        assert shape['fist_dorsal_depth_recovery'] is expected
        assert details['pose_validation']['valid'] is expected
        if expected:
            assert config.class_names[int(resolved.argmax())] == 'fist'
        else:
            assert details['pose_validation']['reason'] == 'fist fingers face away from the camera'


@pytest.mark.parametrize('confidence', [.96, .995, 1.0])
def test_webcam_releases_fist_for_opposite_facing_view(monkeypatch, confidence):
    config = load_runtime_config()
    front_ratio = np.cos(np.deg2rad(-86)) / np.tan(np.deg2rad(51))
    reverse_ratio = np.cos(np.deg2rad(-83)) / np.tan(np.deg2rad(170))
    points, accepted_world = fist_at_xy_angle(-86, front_ratio)
    _, rotated_world = fist_at_xy_angle(-83, reverse_ratio)
    front_angles = hand_plane_angles(accepted_world)
    reverse_angles = hand_plane_angles(rotated_world)
    assert (front_angles['xy'], front_angles['yz'], front_angles['zx']) == pytest.approx((-86, 177, 51), abs=2)
    assert (reverse_angles['xy'], reverse_angles['yz'], reverse_angles['zx']) == pytest.approx((-83, -148, 170), abs=3)
    row = np.zeros(len(config.class_names))
    row[config.class_to_idx['fist']] = confidence
    row[config.class_to_idx['left']] = 1 - confidence

    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config = config
    engine._lock = threading.Lock()
    engine.resolver = GeometryResolver(config)
    detector = SimpleNamespace(ready=True, current_points=points, current_world=accepted_world)
    # Hold the same closed 2-D silhouette while the world axis flips toward
    # the second screenshot's opposite-facing view.
    detector.detect = lambda image: detector.current_points
    detector.world_landmarks = lambda: detector.current_world
    detector.diagnostics = lambda: {}
    engine.detector = detector
    engine.model_manager = SimpleNamespace(
        ready=True, models={'ONNX': None}, resolve_name=lambda name: 'ONNX',
        predict_many=lambda feature, names: {'ONNX': (row, 0., {'known_gesture_mass': 1.0})},
    )
    success, encoded = cv2.imencode('.jpg', np.full((640, 640, 3), 145, dtype=np.uint8))
    assert success
    frame = encoded.tobytes()
    session = RuntimeSession(config)

    results = []
    for now in (1., 1.1, 1.2, 1.3, 1.4, 1.5):
        monkeypatch.setattr(runtime_module.time, 'monotonic', lambda: now)
        results.append(engine.process_frame(frame, session))
    assert results[-1]['runtime_prediction'] == 'fist'
    assert any(result['runtime_action'] == config.gesture_to_action['fist'] for result in results)

    detector.current_world = rotated_world
    for now in (1.6, 1.7, 1.8, 1.9, 2.0):
        monkeypatch.setattr(runtime_module.time, 'monotonic', lambda: now)
        result = engine.process_frame(frame, session)
        assert result['raw_prediction'] == 'fist'
        assert result['runtime_prediction'] == 'no_gesture'
        assert result['runtime_action'] == 'Wait / No Action'
        assert result['diagnostics']['ONNX']['prediction'] == 'no_gesture'

    detector.current_world = accepted_world
    monkeypatch.setattr(runtime_module.time, 'monotonic', lambda: 2.1)
    result = engine.process_frame(frame, session)
    assert result['runtime_prediction'] == 'fist'

    # The newly reported view has negative palm-axis depth too, but its
    # opposite surface winding identifies a dorsal, fully curled fist.
    xy, zx = -75, 159
    depth_ratio = np.cos(np.deg2rad(xy)) / np.tan(np.deg2rad(zx))
    detector.current_points, detector.current_world = fist_at_xy_angle(xy, depth_ratio)
    detector.diagnostics = lambda: {
        'selected_handedness': 'Right',
        'selected_handedness_confidence': .99,
    }
    monkeypatch.setattr(runtime_module.time, 'monotonic', lambda: 2.2)
    result = engine.process_frame(frame, session)
    assert result['runtime_prediction'] == 'fist'
