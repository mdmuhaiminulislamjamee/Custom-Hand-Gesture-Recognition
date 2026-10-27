"""Approximate overlay traces from reverse palm.mp4, not raw camera/depth data."""
import json
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.config import CLASS_NAMES, PROJECT_ROOT, load_runtime_config
from backend.geometry import GeometryResolver, stop_inverted_geometry
from backend.hand_selection import gate_arguments
from backend.model_runtime import InferenceEngine, ModelManager, RuntimeSession
from backend.runtime import TemporalGate
from backend.tests.test_hand_detection import simulated_detector


REVIEW = json.loads((PROJECT_ROOT / "artifacts/stop_inverted/reverse_palm_review/overlay_traces.json").read_text())
FRAMES = REVIEW["frames"]
SHAPE = (REVIEW["viewport"]["height"], REVIEW["viewport"]["width"], 3)


def traced_pose(frame, mirror=False):
    viewport = REVIEW["viewport"]
    points = (np.asarray(frame["points"], np.float32) - [viewport["x"], viewport["y"]]) / [SHAPE[1], SHAPE[0]]
    if mirror:
        points[:, 0] = 1. - points[:, 0]
    return points.astype(np.float32), "Left" if mirror else "Right"


def resolve(frame, mirror=False, label="dorsal", confidence=.99, world=None):
    points, side = traced_pose(frame, mirror)
    config = load_runtime_config()
    row = np.eye(len(CLASS_NAMES))[config.class_to_idx[label]]
    row, details = GeometryResolver(config).resolve(row, points, handedness=side,
        handedness_confidence=confidence, world_landmarks=world, image_shape=SHAPE)
    return config, row, details


@pytest.mark.parametrize("frame", FRAMES, ids=lambda f: f"{f['seconds']}s")
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("label", ["dorsal", "stop_inverted", "open_palm", "like"])
def test_reported_open_back_of_hand_recovers_despite_classifier_miss(frame, mirror, label):
    config, row, details = resolve(frame, mirror, label)
    pose = details["pose_validation"]
    assert pose["valid"] and pose["geometry_supported"] and pose["stop_inverted_recovery"]
    assert pose["gesture"] == "stop_inverted"
    assert not details["hand_shape"]["dorsal_hard_negative"]
    arguments = gate_arguments(details, {"known_gesture_mass": frame["displayed_mass"]})
    gate = TemporalGate(config)
    assert not gate.update(row, now=1., **arguments).execute
    for now in (1.1, 1.2, 1.3, 1.4, 1.5):
        decision = gate.update(row, now=now, **arguments)
        if frame["displayed_mass"] < config.known_mass_floor and now < 1.45:
            assert not decision.execute
    assert decision.execute and decision.predicted_gesture == "stop_inverted"
    assert config.gesture_to_action[decision.predicted_gesture] == "Return to main position"


@pytest.mark.parametrize("frame", FRAMES, ids=lambda f: f"{f['seconds']}s")
@pytest.mark.parametrize("confidence", [.2, float("nan")])
def test_reported_pose_still_requires_confident_surface(frame, confidence):
    _, _, details = resolve(frame, confidence=confidence)
    assert not details["pose_validation"]["valid"]
    assert not details["pose_validation"]["stop_inverted_recovery"]


@pytest.mark.parametrize("frame", FRAMES, ids=lambda f: f"{f['seconds']}s")
def test_reviewed_negative_feedback_disables_low_mass_recovery(frame):
    config, row, details = resolve(frame)
    args = gate_arguments(details, {"known_gesture_mass": .05, "online_adapter": {"negative_support": .5}})
    gate = TemporalGate(config)
    for now in (1., 1.1, 1.2, 1.5, 1.8):
        decision = gate.update(row, now=now, **args)
        assert not decision.execute and decision.predicted_gesture == "no_gesture"


def test_world_curl_disproves_strong_projected_open_hand():
    points, _ = traced_pose(FRAMES[0])
    world = np.column_stack((points, np.zeros(21)))
    for m in (5, 9, 13, 17):
        world[m + 1, 2] = .15
        world[m + 2, 2] = -.15
    _, _, details = resolve(FRAMES[0], world=world)
    assert not details["pose_validation"]["valid"]
    assert not details["pose_validation"]["stop_inverted_recovery"]


@pytest.mark.parametrize("mirror", [False, True])
def test_lowering_a_recovered_stop_clears_hold_and_requires_new_hold(mirror):
    config, row, details = resolve(FRAMES[1], mirror)
    args = gate_arguments(details, {"known_gesture_mass": .05})
    gate = TemporalGate(config)
    for now in (1., 1.1, 1.2, 1.3, 1.4, 1.5):
        decision = gate.update(row, now=now, **args)
    assert decision.execute
    points, side = traced_pose(FRAMES[1], mirror)
    # Flip the same open hand vertically into the lower semicircle.
    points[:, 1] = 2 * points[0, 1] - points[:, 1]
    outside, rejected = GeometryResolver(config).resolve(row, points,
        handedness="Right" if side == "Left" else "Left", handedness_confidence=.99, image_shape=SHAPE)
    decision = gate.update(outside, now=1.6,
        **gate_arguments(rejected, {"known_gesture_mass": .05}))
    assert not decision.execute and decision.predicted_gesture == "no_gesture"
    assert not rejected["pose_validation"]["stop_inverted_recovery"]
    for now in (1.7, 1.8, 1.9, 2., 2.1):
        assert not gate.update(row, now=now, **args).execute
    assert gate.update(row, now=2.2, **args).execute


@pytest.mark.parametrize("frame", FRAMES, ids=lambda f: f"{f['seconds']}s")
def test_palm_facing_camera_never_claims_stop_recovery(frame):
    points, _ = traced_pose(frame)
    config = load_runtime_config()
    row = np.eye(len(CLASS_NAMES))[config.class_to_idx["stop_inverted"]]
    _, details = GeometryResolver(config).resolve(row, points,
        handedness="Left", handedness_confidence=.99, image_shape=SHAPE)
    assert not details["hand_shape"]["stop_inverted_valid"]
    assert not details["pose_validation"]["stop_inverted_recovery"]


@pytest.mark.parametrize("frame", FRAMES, ids=lambda f: f"{f['seconds']}s")
def test_source_image_aspect_does_not_change_open_finger_geometry(frame):
    points, _ = traced_pose(frame)
    pixels = points * [SHAPE[1], SHAPE[0]]
    results = [stop_inverted_geometry(pixels / [w, h], (h, w, 3))
               for h, w in ((600, 600), (540, 960), (1008, 1918))]
    assert all(result["valid"] and result["strong_geometry"] for result in results)
    assert results[0]["minimum_reach"] == pytest.approx(results[-1]["minimum_reach"], abs=1e-5)
    assert results[0]["minimum_extension"] == pytest.approx(results[-1]["minimum_extension"], abs=1e-5)


@pytest.fixture(scope="module")
def model_manager():
    manager = ModelManager(load_runtime_config())
    assert manager.ready, manager.errors
    return manager


@pytest.mark.parametrize("frame", FRAMES, ids=lambda f: f"{f['seconds']}s")
@pytest.mark.parametrize("mirror", [False, True])
def test_live_path_recognizes_video_traces_with_shipped_onnx(frame, mirror, model_manager):
    config = load_runtime_config()
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine.model_manager = config, model_manager
    engine.resolver, engine._lock = GeometryResolver(config), threading.Lock()
    engine.detector = simulated_detector()
    points, side = traced_pose(frame, mirror)
    # No depth/handedness confidence is available in the screen recording.
    # This supplies verified synthetic detector metadata to exercise the live
    # selection, real ONNX, temporal gate, and command mapping together.
    engine.detector._run_candidates = lambda *args, **kwargs: [
        (.1, points.copy(), SimpleNamespace(category_name=side, score=.99), None)]
    jpeg = cv2.imencode(".jpg", np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    session.follow_object.start(now=1.)
    results = [engine.process_frame(jpeg, session, now=1 + tick * .1) for tick in range(16)]
    assert results[-1]["runtime_prediction"] == "stop_inverted"
    assert sum(r["runtime_action"] == "Return to main position" for r in results) == 1
    assert not session.follow_object.active
