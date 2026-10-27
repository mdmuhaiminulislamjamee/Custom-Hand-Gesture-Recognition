"""Upper Fist command arc and real 3-D pitch, including live release behavior."""
from types import SimpleNamespace
import threading

import cv2
import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.geometry import GeometryResolver
from backend.hand_selection import gate_arguments
from backend.model_runtime import InferenceEngine, RuntimeSession
from backend.runtime import TemporalGate
from backend.tests.test_geometry import compact_side_fist_pose, fist_at_xy_angle
from backend.tests.test_hand_detection import simulated_detector


def pitched_fist(angle, pitch=0, *, mirror=False, shape="front"):
    """Pitch the actual closed joints toward the camera, then rotate on screen.

    Image Y points down. Starting upright and rotating about X by +30 degrees
    makes the knuckles closer than the wrist (negative Z); mirroring keeps Z.
    """
    _, world = (fist_at_xy_angle(-90, 0) if shape == "front"
                else compact_side_fist_pose(-90, depth_ratio=0))
    world = world - world[0]
    radians = np.deg2rad(pitch)
    c, s = np.cos(radians), np.sin(radians)
    world = world @ np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]]).T
    radians = np.deg2rad(angle + 90)
    c, s = np.cos(radians), np.sin(radians)
    world = world @ np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
    if mirror:
        world[:, 0] *= -1
    return (world[:, :2] + [.5, .6]).astype(np.float32), world.astype(np.float32)


def resolve_fist(points, world, mirror, label):
    config = RuntimeConfig()
    row = np.eye(len(config.class_names))[config.class_to_idx[label]]
    resolved, details = GeometryResolver(config).resolve(
        row, points, world_landmarks=world, require_fist_depth=True,
        handedness="Left" if mirror else "Right", handedness_confidence=.99)
    return config, resolved, details


@pytest.mark.parametrize("angle", range(-180, 1, 10))
@pytest.mark.parametrize("pitch", [0, 15, 29.9, 30])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("shape,label", [("front", "fist"), ("compact", "thumb_down")])
def test_upper_arc_accepts_closed_fist_with_forward_pitch(angle, pitch, mirror, shape, label):
    points, world = pitched_fist(angle, pitch, mirror=mirror, shape=shape)
    config, probabilities, details = resolve_fist(points, world, mirror, label)
    pose, hand = details["pose_validation"], details["hand_shape"]
    assert hand["fist_toward_camera_pitch_degrees"] == pytest.approx(pitch, abs=1e-4)
    assert hand["fist_screen_direction_allowed"]
    assert pose["valid"] and pose["geometry_supported"]
    assert config.class_names[int(probabilities.argmax())] == "fist"
    gate = TemporalGate(config)
    for frame in range(8):
        decision = gate.update(probabilities, now=1 + frame * .1,
                               **gate_arguments(details, dict(known_gesture_mass=1.)))
    assert decision.predicted_gesture == "fist"


@pytest.mark.parametrize("angle", [.1, 1, 10, 30, 45, 60, 90, 120, 150, 170, 179, 179.9])
@pytest.mark.parametrize("pitch", [0, 30])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("label", ["fist", "thumb_down", "down"])
def test_lower_arc_is_no_gesture_even_with_forward_pitch_or_another_model_label(angle, pitch, mirror, label):
    points, world = pitched_fist(angle, pitch, mirror=mirror)
    config, probabilities, details = resolve_fist(points, world, mirror, label)
    pose = details["pose_validation"]
    assert not details["hand_shape"]["fist_screen_direction_allowed"]
    assert not pose["valid"]
    gate = TemporalGate(config)
    for frame in range(8):
        decision = gate.update(probabilities, now=1 + frame * .1,
                               **gate_arguments(details, dict(known_gesture_mass=1.)))
        assert decision.predicted_gesture == "no_gesture" and not decision.execute


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("shape", ["front", "compact"])
def test_forward_pitch_does_not_require_reliable_handedness(mirror, shape):
    points, world = pitched_fist(-135, 30, mirror=mirror, shape=shape)
    config = RuntimeConfig()
    row = np.eye(len(config.class_names))[config.class_to_idx["fist"]]
    _, details = GeometryResolver(config).resolve(row, points, world_landmarks=world,
                                                 require_fist_depth=True)
    assert details["pose_validation"]["valid"]


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("pitch,accepted", [(29.9, True), (30, True), (30.1, False), (45, False)])
def test_forward_pitch_boundary_without_surface_recovery(mirror, pitch, accepted):
    points, world = pitched_fist(-90, pitch, mirror=mirror)
    config = RuntimeConfig()
    row = np.eye(len(config.class_names))[config.class_to_idx["fist"]]
    _, details = GeometryResolver(config).resolve(row, points, world_landmarks=world,
                                                 require_fist_depth=True)
    assert details["hand_shape"]["fist_depth_allowed"] is accepted
    assert details["pose_validation"]["valid"] is accepted


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("lower_angle", [.1, 45, 135, 179.9])
def test_live_upper_fist_releases_immediately_below_horizontal_and_reholds_after_return(mirror, lower_angle):
    config = RuntimeConfig()
    row = np.eye(len(config.class_names))[config.class_to_idx["fist"]]
    quality = dict(known_gesture_mass=1.)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine._lock = config, threading.Lock()
    engine.resolver = GeometryResolver(config)
    engine.detector = simulated_detector()
    engine.model_manager = SimpleNamespace(
        ready=True, models={"ONNX": object()}, resolve_name=lambda name: "ONNX",
        predict_detailed=lambda *args: (row.copy(), "ONNX", .01, quality),
        predict_many=lambda *args: {"ONNX": (row.copy(), .01, quality)})
    points, world = pitched_fist(-90, 30, mirror=mirror)
    category = SimpleNamespace(category_name="Left" if mirror else "Right", score=.99)
    engine.detector._run_candidates = lambda *args, **kwargs: [(.1, points, category, world)]
    image = cv2.imencode(".jpg", np.full((640, 640, 3), 145, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    outputs = [engine.process_frame(image, session, now=1 + frame * .1) for frame in range(10)]
    assert any(result["runtime_action"] == "Mute" for result in outputs)
    points, world = pitched_fist(lower_angle, 30, mirror=mirror)
    for frame in range(4):
        result = engine.process_frame(image, session, now=2 + frame * .1)
        assert result["runtime_prediction"] == "no_gesture"
        assert result["runtime_action"] == "Wait / No Action"
        assert not result["resolvers"]["pose_validation"]["valid"]
    points, world = pitched_fist(-90, 30, mirror=mirror)
    result = engine.process_frame(image, session, now=2.4)
    assert result["runtime_action"] == "Wait / No Action"
    assert result["stable_frames"] < config.stable_frames_required
