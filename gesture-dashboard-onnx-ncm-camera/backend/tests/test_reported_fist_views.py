"""World-axis regressions from the webcam HUD, not raw screenshot replays."""

import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.config import load_runtime_config
from backend.geometry import GeometryResolver
from backend.model_runtime import InferenceEngine, RuntimeSession
import backend.model_runtime as runtime_module
from backend.tests.test_geometry import fist_at_xy_angle


@pytest.mark.parametrize('xy,zx', [(-61, 142), (-64, 40), (-64, 140)])
@pytest.mark.parametrize('confidence', [.96, .995, 1.0])
def test_webcam_keeps_a_held_fist_when_only_its_world_view_changes(monkeypatch, xy, zx, confidence):
    config = load_runtime_config()
    points, accepted_world = fist_at_xy_angle(-130, -.575)
    depth_ratio = np.cos(np.deg2rad(xy)) / np.tan(np.deg2rad(zx))
    _, rotated_world = fist_at_xy_angle(xy, depth_ratio)
    row = np.zeros(len(config.class_names))
    row[config.class_to_idx['fist']] = confidence
    row[config.class_to_idx['left']] = 1 - confidence

    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config = config
    engine._lock = threading.Lock()
    engine.resolver = GeometryResolver(config)
    detector = SimpleNamespace(ready=True, current_world=accepted_world)
    # Hold the closed image track fixed to exercise a lagging smoother. The
    # latest policy accepts a verified closed fist at every world orientation.
    detector.detect = lambda image: points
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
        assert result['runtime_prediction'] == 'fist'
        assert result['runtime_action'] == 'Wait / No Action'
        assert len(result['display_landmarks']) == 21
        assert result['action_reason'] == 'duplicate command suppressed'
        assert result['diagnostics']['ONNX']['prediction'] == 'fist'

    # Changing back preserves the held Fist and its command suppression. A
    # camera angle change does not count as releasing and reissuing a command.
    detector.current_world = accepted_world
    monkeypatch.setattr(runtime_module.time, 'monotonic', lambda: 2.1)
    result = engine.process_frame(frame, session)
    assert result['runtime_prediction'] == 'fist'
    assert result['runtime_action'] == 'Wait / No Action'
    assert result['action_reason'] == 'duplicate command suppressed'
