"""Approximate the NCM screenshot skeleton; no raw landmark replay is available."""

import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.config import load_runtime_config
from backend.geometry import (
    GeometryResolver, arched_thumb_pointing_geometry, landmarks_to_feature,
)
from backend.hand_selection import gate_arguments
from backend.model_runtime import InferenceEngine, ModelManager, RuntimeSession, hand_plane_angles
from backend.runtime import TemporalGate
import backend.model_runtime as runtime_module


PIXELS = np.asarray([
    [344, 300], [400, 253], [472, 236], [529, 258], [544, 292],
    [472, 302], [574, 311], [632, 313], [677, 317],
    [460, 359], [511, 318], [541, 327], [553, 351],
    [427, 420], [470, 389], [521, 364], [530, 386],
    [430, 418], [470, 389], [494, 390], [499, 416],
], dtype=np.float32)
FRAME_SIZE = np.asarray([956, 748], dtype=np.float32)


def screenshot_world():
    # Preserve the projected skeleton and reconstruct the HUD palm-axis angles.
    xy = PIXELS.copy()
    xy[:, 1] *= .64
    axis = xy[[5, 9, 13, 17]].mean(axis=0) - xy[0]
    unit = axis / np.linalg.norm(axis)
    depth_ratio = (axis[0] / np.tan(np.deg2rad(151))) / np.linalg.norm(axis)
    return np.column_stack((xy, (xy - xy[0]) @ unit * depth_ratio))


def test_reported_view_recovers_left_from_shipped_onnx_low_mass():
    config = load_runtime_config()
    model = ModelManager(config)
    assert model.ready, model.errors
    points = PIXELS / FRAME_SIZE
    probabilities, _, _, quality = model.predict_detailed(landmarks_to_feature(points))
    world = screenshot_world()
    angles = hand_plane_angles(world)
    assert angles["xy"] == pytest.approx(25, abs=1)
    assert angles["yz"] == pytest.approx(-75, abs=1)
    assert angles["zx"] == pytest.approx(151, abs=1)
    assert quality["known_gesture_mass"] < .02
    resolved, details = GeometryResolver(config).resolve(
        probabilities, points, world_landmarks=world, require_fist_depth=True)
    assert details["directional"]["arched_thumb_pointing"]
    assert details["pose_validation"]["expanded_thumb_direction_recovery"]
    assert config.class_names[int(resolved.argmax())] == "left"
    gate = TemporalGate(config)
    for frame in range(6):
        decision = gate.update(resolved, now=1 + frame * .1,
                               **gate_arguments(details, quality))
    assert decision.predicted_gesture == "left" and decision.execute


@pytest.mark.parametrize("mirror", [False, True])
def test_arched_thumb_view_recovers_horizontal_direction_at_rounded_001_mass(mirror):
    config = load_runtime_config()
    points = PIXELS.copy()
    if mirror:
        points[:, 0] = FRAME_SIZE[0] - points[:, 0]
    points /= FRAME_SIZE
    row = np.eye(len(config.class_names))[config.class_to_idx["like"]]
    resolved, details = GeometryResolver(config).resolve(row, points)
    expected = "right" if mirror else "left"
    assert details["directional"]["arched_thumb_pointing"]
    assert config.class_names[int(resolved.argmax())] == expected
    gate = TemporalGate(config)
    for frame in range(6):
        decision = gate.update(resolved, now=1 + frame * .1,
                               **gate_arguments(details, dict(known_gesture_mass=.006)))
    assert decision.predicted_gesture == expected and decision.execute


@pytest.mark.parametrize("fault", [
    "fold_index", "raise_middle", "tuck_thumb", "same_side_thumb", "diagonal",
])
def test_arched_thumb_recovery_keeps_nearby_non_pointing_poses_rejected(fault):
    points = PIXELS.copy()
    if fault == "fold_index":
        points[8] = points[5] + [35, 20]
    elif fault == "raise_middle":
        points[9:13] = points[5:9] + [0, 35]
    elif fault == "tuck_thumb":
        points[2:5] = [[430, 275], [450, 290], [465, 310]]
    elif fault == "same_side_thumb":
        points[1:5, 1] += 155
    else:
        rotation = np.deg2rad(40)
        transform = np.asarray([[np.cos(rotation), -np.sin(rotation)],
                                [np.sin(rotation), np.cos(rotation)]])
        points = (points - points[0]) @ transform.T + points[0]
    assert not arched_thumb_pointing_geometry(points / FRAME_SIZE)


def test_live_pipeline_shows_left_and_executes_after_low_mass_hold(monkeypatch):
    config = load_runtime_config()
    points = PIXELS / FRAME_SIZE
    world = screenshot_world()
    row = np.eye(len(config.class_names))[config.class_to_idx["like"]]
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config = config
    engine._lock = threading.Lock()
    engine.resolver = GeometryResolver(config)
    engine.detector = SimpleNamespace(
        ready=True, detect=lambda image: points, world_landmarks=lambda: world,
        diagnostics=lambda: {})
    engine.model_manager = SimpleNamespace(
        ready=True, models={"ONNX": None}, resolve_name=lambda name: "ONNX",
        predict_many=lambda feature, names: {
            "ONNX": (row, 0., {"known_gesture_mass": .006})})
    _, encoded = cv2.imencode(".jpg", np.full((748, 956, 3), 145, dtype=np.uint8))
    session = RuntimeSession(config)
    outputs = []
    for frame in range(7):
        monkeypatch.setattr(runtime_module.time, "monotonic", lambda frame=frame: 1 + frame * .1)
        outputs.append(engine.process_frame(encoded.tobytes(), session))
    assert outputs[-1]["raw_prediction"] == "like"
    assert outputs[-1]["runtime_prediction"] == "left"
    assert any(output["runtime_action"] == config.gesture_to_action["left"]
               for output in outputs)
