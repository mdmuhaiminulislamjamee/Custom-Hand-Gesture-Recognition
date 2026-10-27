"""Approximate traces of the three reported Left screenshots.

These are manually read overlay joints, not original detector output or depth.
Keep reported vocabulary masses separate from the shipped model's predictions.
"""
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.config import load_runtime_config
from backend.geometry import GeometryResolver, arched_thumb_pointing_geometry, landmarks_to_feature
from backend.hand_selection import gate_arguments
from backend.model_runtime import InferenceEngine, ModelManager, RuntimeSession
from backend.runtime import TemporalGate
from backend.tests.test_hand_detection import simulated_detector


# (window width, image height excluding title bar, title-bar height, HUD mass)
VIEWS = [
    ((953, 726, 25, .14), [
        [290, 595], [300, 548], [327, 514], [351, 506], [365, 520],
        [358, 539], [392, 531], [411, 528], [428, 526],
        [356, 557], [372, 544], [379, 551], [363, 561],
        [354, 578], [368, 563], [363, 570], [350, 578],
        [352, 601], [363, 581], [359, 575], [354, 578],
    ]),
    ((954, 717, 22, .01), [
        [280, 601], [288, 556], [310, 526], [336, 520], [351, 534],
        [344, 551], [365, 546], [385, 544], [400, 541],
        [341, 567], [355, 554], [361, 559], [348, 568],
        [340, 587], [353, 573], [350, 577], [339, 581],
        [339, 612], [349, 591], [344, 578], [338, 573],
    ]),
    ((952, 741, 27, .01), [
        [268, 548], [286, 508], [312, 489], [334, 491], [348, 495],
        [335, 502], [370, 499], [389, 502], [405, 504],
        [334, 531], [350, 513], [359, 521], [345, 531],
        [332, 550], [347, 535], [341, 544], [326, 548],
        [330, 568], [341, 551], [335, 539], [329, 535],
    ]),
]


def side_view(view, mirror=False):
    (width, height, title, _), pixels = VIEWS[view]
    points = (np.asarray(pixels, np.float32) - [0, title]) / [width, height]
    if mirror:
        points[:, 0] = 1. - points[:, 0]
    return points.astype(np.float32)


@pytest.mark.parametrize("view", range(3))
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("raw_label", ["left", "like", "peace", "rock"])
def test_reported_side_views_recover_at_the_displayed_mass(view, mirror, raw_label):
    config = load_runtime_config()
    points = side_view(view, mirror)
    assert arched_thumb_pointing_geometry(points)
    raw = np.eye(len(config.class_names))[config.class_to_idx[raw_label]]
    resolved, details = GeometryResolver(config).resolve(raw, points)
    assert details["pose_validation"]["expanded_thumb_direction_recovery"]
    gate = TemporalGate(config)
    args = gate_arguments(details, {"known_gesture_mass": VIEWS[view][0][3]})
    for frame in range(7):
        decision = gate.update(resolved, now=1. + frame * .1, **args)
        if frame < 5:
            assert not decision.execute
    assert decision.execute
    assert decision.predicted_gesture == ("right" if mirror else "left")


@pytest.fixture(scope="module")
def shipped_model():
    model = ModelManager(load_runtime_config())
    assert model.ready, model.errors
    return model


@pytest.mark.parametrize("view", range(3))
def test_reported_left_side_views_with_shipped_classifier(shipped_model, view):
    config = shipped_model.config
    points = side_view(view)
    raw, _, _, quality = shipped_model.predict_detailed(landmarks_to_feature(points))
    resolved, details = GeometryResolver(config).resolve(raw, points)
    gate = TemporalGate(config)
    for frame in range(7):
        decision = gate.update(resolved, now=1. + frame * .1,
                               **gate_arguments(details, quality))
    assert decision.execute, (quality, details)
    assert decision.predicted_gesture == "left"


@pytest.mark.parametrize("view", range(3))
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("fault", [
    "middle", "ring", "pinky", "middle_forward", "ring_forward", "pinky_forward",
    "index", "thumb", "same_side", "diagonal",
])
def test_side_view_recovery_requires_a_single_pointing_index(view, mirror, fault):
    points = side_view(view, mirror)
    index = points[8] - points[5]
    if fault in {"middle", "ring", "pinky"}:
        base = {"middle": 9, "ring": 13, "pinky": 17}[fault]
        points[base + 1:base + 4] = [points[base] + index * part for part in (.4, .7, 1.)]
    elif fault.endswith("_forward"):
        base = {"middle_forward": 9, "ring_forward": 13, "pinky_forward": 17}[fault]
        points[base + 3] = points[base + 1] + .1 * index
    elif fault == "index":
        points[5:9] = points[9:13]
    elif fault == "thumb":
        points[4] = points[5]
    elif fault == "same_side":
        unit = index / np.linalg.norm(index)
        normal = np.asarray([-unit[1], unit[0]])
        offsets = (points[1:5] - points[5]) @ normal
        points[1:5] -= 2 * offsets[:, None] * normal
    else:
        angle = np.deg2rad(40)
        points = (points - points[0]) @ np.asarray([
            [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).T
    assert not arched_thumb_pointing_geometry(points)


@pytest.mark.parametrize("view", range(3))
@pytest.mark.parametrize("quality", [
    {"known_gesture_mass": .00499},
    {"known_gesture_mass": .14, "online_adapter": {"negative_support": .9}},
])
def test_side_view_recovery_keeps_unknown_and_feedback_rejection(view, quality):
    config = load_runtime_config()
    raw = np.eye(len(config.class_names))[config.class_to_idx["like"]]
    resolved, details = GeometryResolver(config).resolve(raw, side_view(view))
    gate = TemporalGate(config)
    for frame in range(8):
        decision = gate.update(resolved, now=1. + frame * .1,
                               **gate_arguments(details, quality))
        assert not decision.execute
        assert decision.predicted_gesture == "no_gesture"


@pytest.mark.parametrize("view", range(3))
def test_reported_left_reaches_live_candidate_selection_and_action(shipped_model, view):
    config = shipped_model.config
    points = side_view(view)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config = config
    engine.model_manager = shipped_model
    engine.resolver = GeometryResolver(config)
    engine._lock = threading.Lock()
    engine.detector = simulated_detector()
    # Supply the traced landmarks; camera image detection is not replayed.
    engine.detector._run_candidates = lambda rgb, crop, video: [
        (.1, points.copy(), SimpleNamespace(category_name="Right", score=.99), None)]
    width, height = VIEWS[view][0][:2]
    encoded = cv2.imencode(".jpg", np.full((height, width, 3), 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    results = [engine.process_frame(encoded, session, now=1. + tick * .1) for tick in range(14)]
    assert results[-1]["runtime_prediction"] == "left"
    assert results[-1]["hands"][0]["eligible"]
    assert results[-1]["hands"][0]["gesture"] == "left"
    assert results[-1]["resolvers"]["pose_validation"]["expanded_thumb_direction_recovery"]
    assert any(result["runtime_action"] == config.gesture_to_action["left"] for result in results)
