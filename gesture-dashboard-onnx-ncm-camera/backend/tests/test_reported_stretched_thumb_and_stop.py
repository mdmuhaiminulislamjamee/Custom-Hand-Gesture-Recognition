"""Approximate joints from the reported Left screenshots and tilted Stop inv.

The screenshots contain no raw detector metadata. World joints below are
synthetic perturbations that exercise the live depth checks, not captured depth.
"""
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.config import load_runtime_config
from backend.geometry import GeometryResolver
from backend.hand_selection import gate_arguments
from backend.model_runtime import InferenceEngine, ModelManager, RuntimeSession
from backend.runtime import TemporalGate
from backend.tests.test_hand_detection import simulated_detector


# Width, client height, title height, reported mass, traced screen joints.
VIEWS = [
    (960, 718, 27, .00, [
        [406, 562], [419, 518], [453, 485], [489, 476], [513, 473],
        [467, 498], [517, 495], [546, 496], [568, 499],
        [472, 531], [511, 516], [504, 508], [493, 505],
        [475, 556], [505, 541], [499, 530], [485, 534],
        [474, 583], [496, 559], [488, 550], [478, 555],
    ]),
    (963, 722, 27, .54, [
        [444, 571], [458, 519], [486, 489], [516, 481], [532, 484],
        [516, 518], [563, 511], [588, 512], [605, 516],
        [520, 542], [549, 529], [539, 520], [526, 519],
        [516, 565], [540, 551], [530, 541], [519, 542],
        [516, 593], [526, 570], [520, 555], [511, 562],
    ]),
    (953, 711, 26, .43, [
        [497, 597], [487, 540], [493, 490], [508, 453], [522, 421],
        [537, 484], [572, 435], [593, 410], [609, 394],
        [558, 505], [595, 460], [618, 433], [636, 414],
        [570, 531], [605, 490], [627, 465], [641, 448],
        [578, 561], [607, 537], [624, 519], [634, 504],
    ]),
    (963, 713, 25, .09, [
        [364, 604], [373, 537], [403, 490], [438, 473], [458, 476],
        [451, 497], [503, 479], [534, 472], [558, 467],
        [457, 532], [481, 516], [467, 515], [452, 520],
        [462, 566], [474, 547], [461, 544], [448, 551],
        [461, 601], [465, 576], [454, 576], [443, 580],
    ]),
]
LEFT_VIEWS = [0, 1, 3]


def traced_pose(view, mirror=False, palm_error=False):
    width, height, title, _, pixels = VIEWS[view]
    pixels = np.asarray(pixels, dtype=np.float64)
    points = (pixels - [0, title]) / [width, height]
    xy = (pixels - pixels[0]) / 1000.
    world = np.column_stack((xy, .2 * xy[:, 0] + .5 * xy[:, 1]))
    if palm_error:
        # Keep every finger chain intact; perturb only the estimated wrist.
        world[0] = (-1.1 if view == 2 else .55) * world[9]
    if mirror:
        points[:, 0] = 1. - points[:, 0]
        world[:, 0] *= -1.
    return points.astype(np.float32), world, (height, width, 3)


def resolve(view, mirror=False, palm_error=False, raw="like", world_override=None):
    config = load_runtime_config()
    points, world, shape = traced_pose(view, mirror, palm_error)
    if world_override is not None:
        world = world_override
    row = np.eye(len(config.class_names))[config.class_to_idx[raw]]
    row, details = GeometryResolver(config).resolve(
        row, points, world_landmarks=world, image_shape=shape,
        handedness="Left" if mirror else "Right", handedness_confidence=.99)
    return config, row, details


@pytest.mark.parametrize("view", LEFT_VIEWS)
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("palm_error", [False, True])
@pytest.mark.parametrize("raw", ["left", "like", "peace", "rock", "dorsal"])
def test_stretched_thumb_left_at_reported_mass(view, mirror, palm_error, raw):
    config, row, details = resolve(view, mirror, palm_error, raw)
    assert details["pose_validation"]["horizontal_direction_recovery"]
    if palm_error:
        if view in (0, 1):
            assert not details["directional"]["world_index_only"]
        assert details["directional"]["world_folded_pointing"]
    args = gate_arguments(details, {"known_gesture_mass": VIEWS[view][3]})
    gate = TemporalGate(config)
    for tick in range(7):
        decision = gate.update(row, now=1. + tick * .1, **args)
        if tick < 5:
            assert not decision.execute
    assert decision.execute and decision.predicted_gesture == ("right" if mirror else "left")


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("palm_error", [False, True])
@pytest.mark.parametrize("raw", ["like", "dorsal", "open_palm", "stop_inverted"])
def test_tilted_stop_at_reported_mass(mirror, palm_error, raw):
    config, row, details = resolve(2, mirror, palm_error, raw)
    pose = details["pose_validation"]
    assert pose["valid"] and pose["stop_inverted_recovery"]
    gate = TemporalGate(config)
    for tick in range(7):
        decision = gate.update(row, now=1. + tick * .1,
                               **gate_arguments(details, {"known_gesture_mass": .43}))
        if tick < 5:
            assert not decision.execute
    assert decision.execute and decision.predicted_gesture == "stop_inverted"


@pytest.mark.parametrize("view", LEFT_VIEWS)
@pytest.mark.parametrize("finger", [5, 9, 13, 17])
def test_depth_still_rejects_folded_index_or_an_extra_raised_finger(view, finger):
    _, world, _ = traced_pose(view, palm_error=True)
    if finger == 5:
        world[8] = world[5]
    else:
        index = world[8] - world[5]
        world[finger + 1:finger + 4] = [world[finger] + part * index for part in (.4, .7, 1.)]
    config, row, details = resolve(view, palm_error=True, raw="rock", world_override=world)
    assert not details["pose_validation"]["horizontal_direction_recovery"]
    gate = TemporalGate(config)
    for tick in range(8):
        assert not gate.update(row, now=1. + tick * .1,
                               **gate_arguments(details, {"known_gesture_mass": 0.})).execute


@pytest.mark.parametrize("finger", [5, 9, 13, 17])
def test_stop_recovery_still_rejects_a_depth_curl(finger):
    _, world, _ = traced_pose(2, palm_error=True)
    world[finger + 1, 2] += .15
    world[finger + 2, 2] -= .15
    _, _, details = resolve(2, palm_error=True, world_override=world)
    assert not details["pose_validation"]["stop_inverted_recovery"]


@pytest.mark.parametrize("view", range(len(VIEWS)))
def test_reviewed_negative_feedback_still_blocks_recovery(view):
    config, row, details = resolve(view, palm_error=True)
    gate = TemporalGate(config)
    args = gate_arguments(details, {"known_gesture_mass": VIEWS[view][3],
                                   "online_adapter": {"negative_support": .9}})
    for tick in range(8):
        assert not gate.update(row, now=1. + tick * .1, **args).execute


def distorted_index_world(view, mirror, distortion):
    """Synthetic depth errors; no original world joints exist in the photos."""
    _, world, _ = traced_pose(view, mirror)
    if distortion == "head_on":
        # Depth says the index points at the lens although the image shows a
        # long horizontal finger. Keep the world chain straight and unfolded.
        span = world[8] - world[5]
        for joint, part in zip((6, 7, 8), (.45, .75, 1.)):
            world[joint] = world[5] + np.r_[part * .15 * span[:2], part * np.linalg.norm(span)]
    else:
        world[6:9] = world[5] + .30 * (world[6:9] - world[5])
    return world


@pytest.mark.parametrize("view", LEFT_VIEWS)
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("distortion", ["head_on", "shortened"])
@pytest.mark.parametrize("raw", ["left", "like", "peace", "rock"])
def test_clear_pointing_image_recovers_despite_a_distorted_world_index(view, mirror, distortion, raw):
    world = distorted_index_world(view, mirror, distortion)
    config, row, details = resolve(view, mirror, raw=raw, world_override=world)
    direction = details["directional"]
    assert direction["horizontal_pointing_geometry"]["clear_single_index"]
    assert not direction["world_pointing_contradiction"]
    assert details["pose_validation"]["valid"]
    if distortion == "head_on":
        assert direction["world_projection_fraction"] < .20
    else:
        assert not direction["world_index_only"] and not direction["world_folded_pointing"]
    gate = TemporalGate(config)
    args = gate_arguments(details, {"known_gesture_mass": VIEWS[view][3]})
    for tick in range(7):
        decision = gate.update(row, now=1. + tick * .1, **args)
        if tick < 5:
            assert not decision.execute
    assert decision.execute and decision.predicted_gesture == ("right" if mirror else "left")


@pytest.mark.parametrize("view", LEFT_VIEWS)
@pytest.mark.parametrize("distortion", ["zeros", "nan", "curled"])
def test_clear_image_recovery_still_requires_usable_non_curled_world_joints(view, distortion):
    _, world, _ = traced_pose(view)
    if distortion == "zeros":
        world[:] = 0.
    elif distortion == "nan":
        world[6, 2] = np.nan
    else:
        world[8] = world[5]
    config, row, details = resolve(view, raw="like", world_override=world)
    assert details["directional"]["world_pointing_contradiction"]
    assert not details["pose_validation"]["horizontal_direction_recovery"]
    gate = TemporalGate(config)
    for tick in range(8):
        assert not gate.update(row, now=1. + tick * .1,
                               **gate_arguments(details, {"known_gesture_mass": .09})).execute


def test_latest_screenshot_angle_uses_the_index_mcp_instead_of_the_palm():
    pixels = np.asarray(VIEWS[3][4], dtype=np.float64)
    vector = pixels[8] - pixels[5]
    assert np.degrees(np.arctan2(vector[1], vector[0])) == pytest.approx(-15.66, abs=.01)


@pytest.fixture(scope="module")
def shipped_model():
    manager = ModelManager(load_runtime_config())
    assert manager.ready, manager.errors
    return manager


@pytest.mark.parametrize("view", range(len(VIEWS)))
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("palm_error", [False, True])
def test_live_selection_onnx_and_action_for_reported_poses(view, mirror, palm_error, shipped_model):
    config = shipped_model.config
    points, world, shape = traced_pose(view, mirror, palm_error)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine.model_manager = config, shipped_model
    engine.resolver, engine._lock = GeometryResolver(config), threading.Lock()
    engine.detector = simulated_detector()
    engine.detector._run_candidates = lambda *args, **kwargs: [
        (.1, points.copy(), SimpleNamespace(category_name="Left" if mirror else "Right", score=.99), world)]
    jpeg = cv2.imencode(".jpg", np.full(shape, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    results = [engine.process_frame(jpeg, session, now=1. + tick * .1) for tick in range(16)]
    expected = "stop_inverted" if view == 2 else ("right" if mirror else "left")
    assert results[-1]["runtime_prediction"] == expected
    assert results[-1]["hands"][0]["eligible"]
    assert any(r["runtime_action"] == config.gesture_to_action[expected] for r in results)


@pytest.mark.parametrize("view", LEFT_VIEWS)
@pytest.mark.parametrize("distortion", ["head_on", "shortened"])
def test_live_path_with_shipped_onnx_accepts_distorted_depth(view, distortion, shipped_model):
    config = shipped_model.config
    points, _, shape = traced_pose(view)
    world = distorted_index_world(view, False, distortion)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine.model_manager = config, shipped_model
    engine.resolver, engine._lock = GeometryResolver(config), threading.Lock()
    engine.detector = simulated_detector()
    engine.detector._run_candidates = lambda *args, **kwargs: [
        (.1, points.copy(), SimpleNamespace(category_name="Right", score=.99), world)]
    jpeg = cv2.imencode(".jpg", np.full(shape, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    results = [engine.process_frame(jpeg, session, now=1. + tick * .1) for tick in range(16)]
    assert results[-1]["runtime_prediction"] == "left"
    assert results[-1]["hands"][0]["eligible"]
    assert any(r["runtime_action"] == config.gesture_to_action["left"] for r in results)
