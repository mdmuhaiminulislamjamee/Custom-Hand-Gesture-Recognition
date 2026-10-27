"""Approximate Oct 19 screenshots; world errors are synthetic, not captured.

These regressions replay the visible joints through the live command path.
They cannot reproduce image detection or the screenshots' missing metadata.
"""
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.config import load_runtime_config
from backend.geometry import GeometryResolver, stop_inverted_geometry
from backend.hand_selection import gate_arguments
from backend.model_runtime import InferenceEngine, ModelManager, RuntimeSession
from backend.runtime import TemporalGate
from backend.tests.test_hand_detection import simulated_detector


# Width, client height, title height, displayed mass, detector side assumed
# from the demonstrated back of the hand, and approximately traced joints.
VIEWS = [
    (955, 723, 27, .20, "Right", [
        [584, 574], [565, 508], [568, 443], [576, 392], [579, 357],
        [626, 427], [658, 367], [675, 339], [688, 320],
        [653, 450], [693, 393], [711, 366], [724, 350],
        [669, 482], [708, 432], [722, 410], [734, 392],
        [679, 513], [710, 486], [724, 472], [734, 456],
    ]),
    (956, 718, 27, .43, "Left", [
        [832, 503], [842, 458], [840, 411], [836, 373], [835, 343],
        [814, 403], [789, 359], [773, 332], [758, 309],
        [801, 423], [772, 378], [746, 348], [730, 327],
        [790, 444], [757, 407], [735, 382], [718, 366],
        [780, 470], [750, 455], [732, 445], [717, 435],
    ]),
]


def traced_pose(view, mirror=False, depth="coherent"):
    width, height, title, _, side, pixels = VIEWS[view]
    pixels = np.asarray(pixels, dtype=np.float64)
    points = (pixels - [0, title]) / [width, height]
    world = np.column_stack(((pixels - pixels[0]) / 1000., np.zeros(21)))
    if depth == "noisy_joint":
        world[18, 2] += .02
    elif depth == "shortened_pinky":
        world[18:21] = world[17] + .25 * (world[18:21] - world[17])
    elif depth == "stretched_wrist":
        world[0] = -3. * world[9]
    if mirror:
        points[:, 0] = 1. - points[:, 0]
        world[:, 0] *= -1.
        side = "Left" if side == "Right" else "Right"
    if depth == "absent":
        world = None
    return points.astype(np.float32), world, side, (height, width, 3)


def resolve(view, mirror=False, depth="coherent", label="dorsal", confidence=.99,
            world_override=None, palmar=False):
    config = load_runtime_config()
    points, world, side, shape = traced_pose(view, mirror, depth)
    if world_override is not None:
        world = world_override
    if palmar:
        side = "Left" if side == "Right" else "Right"
    row = np.eye(len(config.class_names))[config.class_to_idx[label]]
    row, details = GeometryResolver(config).resolve(
        row, points, handedness=side, handedness_confidence=confidence,
        world_landmarks=world, image_shape=shape)
    return config, row, details


@pytest.mark.parametrize("view", range(2))
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("label", ["dorsal", "like", "open_palm", "stop_inverted"])
@pytest.mark.parametrize("depth", ["absent", "coherent", "noisy_joint", "shortened_pinky", "stretched_wrist"])
def test_open_back_of_hand_recovers_at_screenshot_mass(view, mirror, label, depth):
    config, row, details = resolve(view, mirror, depth, label)
    pose = details["pose_validation"]
    assert pose["valid"] and pose["stop_inverted_recovery"]
    assert pose["gesture"] == "stop_inverted"
    assert not pose["hard_negative"]
    assert details["hand_shape"]["stop_inverted_rejection_reason"] is None
    gate = TemporalGate(config)
    args = gate_arguments(details, {"known_gesture_mass": VIEWS[view][3]})
    for tick in range(7):
        decision = gate.update(row, now=1. + tick * .1, **args)
        if tick < 5:
            assert not decision.execute
    assert decision.execute and decision.predicted_gesture == "stop_inverted"


@pytest.mark.parametrize("view", range(2))
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("label", ["dorsal", "stop_inverted"])
@pytest.mark.parametrize("finger", [5, 9, 13, 17])
def test_an_actual_depth_curl_still_rejects_even_a_confident_model(view, mirror, label, finger):
    _, world, _, _ = traced_pose(view, mirror)
    world[finger + 1, 2] += .15
    world[finger + 2, 2] -= .15
    config, row, details = resolve(view, mirror, label=label, world_override=world)
    assert not details["hand_shape"]["stop_inverted_valid"]
    depth = details["hand_shape"]["stop_inverted_depth_geometry"]
    assert not depth["valid"] and depth["curled_fingers"]
    gate = TemporalGate(config)
    for tick in range(8):
        decision = gate.update(row, now=1. + tick * .1,
                               **gate_arguments(details, {"known_gesture_mass": .99}))
        assert not decision.execute and decision.predicted_gesture == "no_gesture"


@pytest.mark.parametrize("view", range(2))
@pytest.mark.parametrize("confidence", [.2, .74, float("nan")])
def test_weak_surface_confidence_is_explained_and_does_not_recover(view, confidence):
    _, _, details = resolve(view, confidence=confidence)
    assert not details["pose_validation"]["stop_inverted_recovery"]
    assert details["hand_shape"]["stop_inverted_rejection_reason"] == "Stop inv: hand surface is uncertain"


@pytest.mark.parametrize("view", range(2))
@pytest.mark.parametrize("mirror", [False, True])
def test_opposite_surface_stays_open_palm(view, mirror):
    _, _, details = resolve(view, mirror, palmar=True, label="stop_inverted")
    assert not details["hand_shape"]["stop_inverted_valid"]
    assert details["pose_validation"]["gesture"] == "open_palm"
    assert details["hand_shape"]["stop_inverted_rejection_reason"] == "Stop inv: show the back of the hand"


@pytest.mark.parametrize("view", range(2))
@pytest.mark.parametrize("invalid", ["zeros", "nan"])
def test_invalid_world_chains_cannot_be_recovered(view, invalid):
    _, world, _, _ = traced_pose(view)
    if invalid == "zeros":
        world[:] = 0.
    else:
        world[18, 2] = np.nan
    _, _, details = resolve(view, world_override=world)
    assert not details["pose_validation"]["stop_inverted_recovery"]


@pytest.mark.parametrize("view", range(2))
def test_negative_feedback_blocks_the_new_recovery(view):
    config, row, details = resolve(view, depth="noisy_joint")
    args = gate_arguments(details, {"known_gesture_mass": VIEWS[view][3],
                                   "online_adapter": {"negative_support": .9}})
    gate = TemporalGate(config)
    for tick in range(8):
        assert not gate.update(row, now=1. + tick * .1, **args).execute


@pytest.mark.parametrize("view", range(2))
def test_lowering_the_same_open_hand_immediately_releases_the_hold(view):
    config, row, details = resolve(view, depth="noisy_joint")
    gate = TemporalGate(config)
    args = gate_arguments(details, {"known_gesture_mass": VIEWS[view][3]})
    for tick in range(7):
        decision = gate.update(row, now=1. + tick * .1, **args)
    assert decision.execute
    points, world, side, shape = traced_pose(view, depth="noisy_joint")
    points[:, 1] = 2 * points[0, 1] - points[:, 1]
    world[:, 1] *= -1.
    side = "Left" if side == "Right" else "Right"
    outside, rejected = GeometryResolver(config).resolve(
        row, points, world_landmarks=world, handedness=side,
        handedness_confidence=.99, image_shape=shape)
    decision = gate.update(outside, now=1.7,
                           **gate_arguments(rejected, {"known_gesture_mass": VIEWS[view][3]}))
    assert not decision.execute and decision.predicted_gesture == "no_gesture"
    assert not rejected["pose_validation"]["stop_inverted_recovery"]
    assert not gate.update(row, now=1.8, **args).execute


@pytest.fixture(scope="module")
def shipped_model():
    cv2.setNumThreads(1)
    model = ModelManager(load_runtime_config())
    assert model.ready, model.errors
    return model


@pytest.mark.parametrize("view", range(2))
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("depth", ["coherent", "noisy_joint", "shortened_pinky"])
def test_live_selection_and_shipped_onnx_execute_one_return_action(view, mirror, depth, shipped_model):
    config = shipped_model.config
    points, world, side, shape = traced_pose(view, mirror, depth)
    assert stop_inverted_geometry(points, shape)["strong_geometry"]
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine.model_manager = config, shipped_model
    engine.resolver, engine._lock = GeometryResolver(config), threading.Lock()
    engine.detector = simulated_detector()
    engine.detector._run_candidates = lambda *args, **kwargs: [
        (.1, points.copy(), SimpleNamespace(category_name=side, score=.99), world)]
    jpeg = cv2.imencode(".jpg", np.full(shape, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    session.follow_object.start(now=1.)
    results = [engine.process_frame(jpeg, session, now=1. + tick * .1) for tick in range(16)]
    assert results[-1]["runtime_prediction"] == "stop_inverted"
    assert results[-1]["hands"][0]["eligible"]
    assert sum(r["runtime_action"] == "Return to main position" for r in results) == 1
    assert not session.follow_object.active
