"""Reviewed screen-overlay traces from left.mp4, not a raw camera replay."""
import json
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.config import PROJECT_ROOT, load_runtime_config
from backend.geometry import GeometryResolver, horizontal_pointing_geometry
from backend.hand_selection import gate_arguments
from backend.model_runtime import InferenceEngine, ModelManager, RuntimeSession
from backend.runtime import TemporalGate
from backend.tests.test_hand_detection import simulated_detector


REVIEW = json.loads((PROJECT_ROOT / "artifacts/left_video_review/overlay_traces.json").read_text())
FRAMES = REVIEW["frames"]
SHAPE = (721, 954, 3)


def traced_pose(frame, mirror=False):
    points = (np.asarray(frame["points"], np.float32) - [0, 27]) / [SHAPE[1], SHAPE[0]]
    if mirror:
        points[:, 0] = 1. - points[:, 0]
    return points.astype(np.float32)


def model_row(config, label):
    return np.eye(len(config.class_names))[config.class_to_idx[label]]


@pytest.mark.parametrize("frame", FRAMES, ids=lambda f: f"{f['seconds']}s")
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("label", ["left", "like", "rock", "peace"])
def test_video_pointing_views_recover_at_recorded_mass_after_half_second(frame, mirror, label):
    config = load_runtime_config()
    row, details = GeometryResolver(config).resolve(model_row(config, label), traced_pose(frame, mirror), image_shape=SHAPE)
    expected = "right" if mirror else "left"
    assert details["pose_validation"]["gesture"] == expected
    assert details["pose_validation"]["horizontal_direction_recovery"]
    gate = TemporalGate(config)
    args = gate_arguments(details, {"known_gesture_mass": frame["displayed_mass"]})
    for tick in range(7):
        decision = gate.update(row, now=1 + tick * .1, **args)
        if tick < 5 and frame["displayed_mass"] < config.known_mass_floor:
            assert not decision.execute
    assert decision.execute and decision.predicted_gesture == expected


@pytest.mark.parametrize("frame", FRAMES, ids=lambda f: f"{f['seconds']}s")
@pytest.mark.parametrize("seed", range(3))
def test_pixel_geometry_is_stable_under_small_trace_error_and_image_aspect(frame, seed):
    rng = np.random.default_rng(seed)
    pixels = np.asarray(frame["points"], float) + rng.uniform(-.75, .75, (21, 2))
    for h, w in ((600, 600), (540, 960), (721, 954)):
        assert horizontal_pointing_geometry(pixels / [w, h], (h, w, 3))["valid"]


@pytest.mark.parametrize("frame", FRAMES, ids=lambda f: f"{f['seconds']}s")
@pytest.mark.parametrize("fault", ["folded_index", "middle", "ring", "pinky", "diagonal"])
def test_recovery_excludes_curled_index_other_extended_fingers_and_diagonals(frame, fault):
    points = traced_pose(frame)
    if fault == "folded_index":
        points[5:9] = points[9:13]
    elif fault == "diagonal":
        pixels = (points - points[0]) * [SHAPE[1], SHAPE[0]]
        angle = np.deg2rad(50)
        pixels = pixels @ np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).T
        points = pixels / [SHAPE[1], SHAPE[0]]
    else:
        base = {"middle": 9, "ring": 13, "pinky": 17}[fault]
        vector = points[8] - points[5]
        points[base:base + 4] = [points[base] + fraction * vector for fraction in (0., .4, .7, 1.)]
    assert not horizontal_pointing_geometry(points, SHAPE)["valid"]


@pytest.mark.parametrize("frame", FRAMES, ids=lambda f: f"{f['seconds']}s")
def test_reviewed_negative_feedback_blocks_recovery_even_at_zero_mass(frame):
    config = load_runtime_config()
    row, details = GeometryResolver(config).resolve(model_row(config, "left"), traced_pose(frame), image_shape=SHAPE)
    args = gate_arguments(details, {"known_gesture_mass": .0, "online_adapter": {"negative_support": .9}})
    gate = TemporalGate(config)
    for tick in range(8):
        decision = gate.update(row, now=1 + tick * .1, **args)
        assert not decision.execute and decision.predicted_gesture == "no_gesture"


def world_pointing_hand(extra_finger=None):
    """Synthetic metric pose to test depth agreement, not video-derived depth."""
    world = np.zeros((21, 3))
    world[5:9] = [[.04, -.03, 0], [.07, -.03, 0], [.09, -.03, 0], [.11, -.03, 0]]
    for m, y in ((9, -.01), (13, .01), (17, .03)):
        world[m:m + 4] = [[.05, y, 0], [.07, y, .005], [.06, y, .015], [.05, y, .01]]
        if m == extra_finger:
            world[m:m + 4] = [[.05 + step, y, 0] for step in (0., .03, .05, .07)]
    return world


@pytest.mark.parametrize("frame", FRAMES, ids=lambda f: f"{f['seconds']}s")
@pytest.mark.parametrize("extra_finger", [None, 9, 13, 17])
def test_real_depth_must_agree_with_the_projected_single_index(frame, extra_finger):
    config = load_runtime_config()
    row, details = GeometryResolver(config).resolve(model_row(config, "rock"), traced_pose(frame),
        image_shape=SHAPE, world_landmarks=world_pointing_hand(extra_finger))
    assert details["pose_validation"]["horizontal_direction_recovery"] == (extra_finger is None)
    args = gate_arguments(details, {"known_gesture_mass": 0.})
    gate = TemporalGate(config)
    for tick in range(8):
        decision = gate.update(row, now=1 + tick * .1, **args)
    assert decision.execute == (extra_finger is None)


@pytest.fixture(scope="module")
def shipped_model():
    cv2.setNumThreads(1)
    manager = ModelManager(load_runtime_config())
    assert manager.ready, manager.errors
    return manager


@pytest.mark.parametrize("frame", FRAMES, ids=lambda f: f"{f['seconds']}s")
@pytest.mark.parametrize("mirror", [False, True])
def test_live_path_selects_and_executes_left_from_the_shipped_onnx(frame, mirror, shipped_model):
    config = load_runtime_config()
    points = traced_pose(frame, mirror)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine.model_manager = config, shipped_model
    engine.resolver, engine._lock = GeometryResolver(config), threading.Lock()
    engine.detector = simulated_detector()
    # Handedness confidence is synthetic; the recording lacks raw metadata.
    engine.detector._run_candidates = lambda *args, **kwargs: [
        (.1, points.copy(), SimpleNamespace(category_name="Left" if mirror else "Right", score=.99), None)]
    # Detection is supplied by the trace: preserve the camera aspect with a
    # smaller neutral image to keep this command-path test inexpensive.
    live_shape = (SHAPE[0] // 2, SHAPE[1] // 2, 3)
    jpeg = cv2.imencode(".jpg", np.full(live_shape, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    results = [engine.process_frame(jpeg, session, now=1 + tick * .1) for tick in range(16)]
    expected = "right" if mirror else "left"
    assert results[-1]["runtime_prediction"] == expected
    assert results[-1]["hands"][0]["eligible"]
    assert any(r["runtime_action"] == config.gesture_to_action[expected] for r in results)
