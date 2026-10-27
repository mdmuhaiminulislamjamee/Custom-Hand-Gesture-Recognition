"""Open Palm accepts up +/-70 degrees; Stop inverted keeps up +/-55."""
import json

import numpy as np
import pytest

from backend.config import CLASS_NAMES, PROJECT_ROOT, RuntimeConfig, load_runtime_config
from backend.geometry import (GeometryResolver, hand_surface_orientation, landmarks_to_feature,
                              stop_inverted_geometry, open_hand_xy_orientation)
from backend.hand_selection import gate_arguments
from backend.runtime import TemporalGate
from scripts.train_stop_inverted import rotate_tilt


def stop_pose(tilt=0., mirror=False, spread=0., image_shape=(600, 600, 3)):
    with np.load(PROJECT_ROOT / "data/stop_inverted/hagrid_train.npz") as data:
        source = data["landmarks"][0].copy()
    pixels = rotate_tilt(source, tilt, mirror, spread) * 1800 + [300, 390]
    points = (pixels / [image_shape[1], image_shape[0]]).astype(np.float32)
    side = next(side for side in ("Left", "Right")
                if hand_surface_orientation(points, side, .99)["dorsal_visible"])
    return points, side


def resolve(points, side, image_shape=(600, 600, 3), raw="stop_inverted", confidence=.99):
    config = RuntimeConfig()
    row = np.eye(len(CLASS_NAMES))[config.class_to_idx[raw]]
    resolved, details = GeometryResolver(config).resolve(row, points, handedness=side,
                                                       handedness_confidence=confidence, image_shape=image_shape)
    return config, resolved, details


@pytest.mark.parametrize("tilt", [-90., -80., -65., 0., 65., 80., 90.])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("spread", [0., 22.])
@pytest.mark.parametrize("shape", [(600, 600, 3), (540, 960, 3)])
@pytest.mark.parametrize("gesture", ["stop_inverted", "open_palm"])
@pytest.mark.parametrize("raw_output", ["stop_inverted", "open_palm"])
def test_both_hands_together_and_spread_respect_each_gesture_range(tilt, mirror, spread, shape, gesture, raw_output):
    points, side = stop_pose(tilt, mirror, spread, shape)
    if gesture == "open_palm":
        side = "Right" if side == "Left" else "Left"
    config, row, details = resolve(points, side, shape, raw=raw_output)
    assert stop_inverted_geometry(points, shape)["tilt_from_up_degrees"] == pytest.approx(tilt, abs=1e-4)
    assert open_hand_xy_orientation(points, shape)["xy_degrees"] == pytest.approx(tilt - 90., abs=1e-4)
    gate = TemporalGate(config)
    arguments = gate_arguments(details, {"known_gesture_mass": .99})
    if abs(tilt) > (70. if gesture == "open_palm" else 55.):
        assert not details["pose_validation"]["valid"]
        for now in (1., 1.1, 1.3, 1.8):
            decision = gate.update(row, now=now, **arguments)
            assert not decision.execute and decision.predicted_gesture == "no_gesture"
        return
    assert details["pose_validation"]["valid"]
    assert details["pose_validation"]["gesture"] == gesture
    assert not gate.update(row, now=1., **arguments).execute
    assert not gate.update(row, now=1.1, **arguments).execute
    decision = gate.update(row, now=1.21, **arguments)
    assert decision.execute and decision.predicted_gesture == gesture
    assert config.gesture_to_action[decision.predicted_gesture] == (
        "Return to main position" if gesture == "stop_inverted" else "Enable / Disable Object Tracking")


@pytest.mark.parametrize("tilt", [-180., -150., -100., -90.01, 90.01, 100., 150., 180.])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("spread", [0., 22.])
@pytest.mark.parametrize("gesture", ["stop_inverted", "open_palm"])
def test_entire_lower_half_is_no_gesture(tilt, mirror, spread, gesture):
    points, side = stop_pose(tilt, mirror, spread)
    if gesture == "open_palm":
        side = "Right" if side == "Left" else "Left"
    config, row, details = resolve(points, side, raw=gesture)
    assert not details["pose_validation"]["valid"]
    gate = TemporalGate(config)
    for now in (1., 1.1, 1.21, 1.8):
        decision = gate.update(row, now=now, **gate_arguments(details, {"known_gesture_mass": 1.}))
        assert decision.predicted_gesture == "no_gesture" and not decision.execute


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("side_mode", ["opposite", "missing", "uncertain"])
def test_palmar_or_unverified_surface_cannot_execute_stop(mirror, side_mode):
    points, side = stop_pose(mirror=mirror)
    if side_mode == "opposite":
        side = "Right" if side == "Left" else "Left"
    elif side_mode == "missing":
        side = None
    _, _, details = resolve(points, side, confidence=.2 if side_mode == "uncertain" else .99)
    assert not details["hand_shape"]["stop_inverted_valid"]
    assert details["pose_validation"]["gesture"] != "stop_inverted" or not details["pose_validation"]["valid"]


@pytest.mark.parametrize("gesture", ["stop_inverted", "open_palm"])
def test_crossing_boundary_rejects_immediately_and_return_needs_new_hold(gesture):
    gate = TemporalGate(RuntimeConfig())
    boundary = 70. if gesture == "open_palm" else 55.
    points, side = stop_pose(boundary - 1.)
    if gesture == "open_palm":
        side = "Right" if side == "Left" else "Left"
    _, row, details = resolve(points, side, raw=gesture)
    arguments = gate_arguments(details, {"known_gesture_mass": .99})
    for now in (1., 1.1, 1.21):
        decision = gate.update(row, now=now, **arguments)
    assert decision.execute
    outside, side = stop_pose(boundary + .1)
    if gesture == "open_palm":
        side = "Right" if side == "Left" else "Left"
    _, bad_row, bad_details = resolve(outside, side, raw=gesture)
    bad = gate.update(bad_row, now=1.3, **gate_arguments(bad_details, {"known_gesture_mass": .99}))
    assert not bad.execute and bad.predicted_gesture == "no_gesture"
    returned = gate.update(row, now=1.4, **arguments)
    assert not returned.execute and returned.stable_frames == 1


@pytest.mark.parametrize("raw_label", CLASS_NAMES)
@pytest.mark.parametrize("surface", ["palmar", "dorsal"])
def test_lower_open_hand_cannot_fall_through_to_another_command(raw_label, surface):
    points, side = stop_pose(135.)
    if surface == "palmar":
        side = "Right" if side == "Left" else "Left"
    config, row, details = resolve(points, side, raw=raw_label)
    assert not details["pose_validation"]["valid"]
    decision = TemporalGate(config).update(row, now=1., **gate_arguments(details, {"known_gesture_mass": 1.}))
    assert not decision.execute and decision.predicted_gesture == "no_gesture"


def test_no_recovery_of_retired_dorsal_output_or_low_vocabulary_mass():
    points, side = stop_pose()
    for label, mass in (("dorsal", 1.), ("stop_inverted", .01)):
        config, row, details = resolve(points, side, raw=label)
        decision = TemporalGate(config).update(row, now=1., **gate_arguments(details, {"known_gesture_mass": mass}))
        assert decision.predicted_gesture == "no_gesture" and not decision.execute


def test_stop_surface_correction_requires_confident_open_hand_classifier_evidence():
    points, side = stop_pose()
    config = RuntimeConfig()
    row = np.zeros(len(CLASS_NAMES))
    row[config.class_to_idx["open_palm"]] = .69
    row[config.class_to_idx["like"]] = .31
    resolved, details = GeometryResolver(config).resolve(row, points, handedness=side,
                                                       handedness_confidence=.99)
    decision = TemporalGate(config).update(resolved, now=1., **gate_arguments(details, {"known_gesture_mass": 1.}))
    assert not details["hand_shape"]["stop_inverted_valid"]
    assert not decision.execute and decision.predicted_gesture == "no_gesture"


def test_saved_feedback_and_people_are_audited_without_evaluation_leakage():
    audit = json.loads((PROJECT_ROOT / "artifacts/stop_inverted/data_audit.json").read_text())
    assert audit["feedback"]["rows"] == 25
    assert audit["feedback"]["class_counts"]["peace"] == 12
    assert audit["feedback"]["class_counts"]["no_gesture"] == 13
    assert not any(audit["participant_overlap"].values())
    with np.load(PROJECT_ROOT / "artifacts/stop_inverted/train.npz") as train:
        assert sum(str(s).startswith("feedback:") for s in train["source_id"]) == 25 * 128
    for split in ("val", "test"):
        with np.load(PROJECT_ROOT / f"artifacts/stop_inverted/{split}.npz") as data:
            assert not any(str(s).startswith("feedback:") for s in data["source_id"])


def test_live_configuration_widens_only_the_open_palm_band():
    raw = load_runtime_config().raw
    stop = raw["stop_inverted_validation"]
    palm = raw["open_palm_rotation_validation"]
    assert stop["maximum_tilt_from_up_degrees"] == 55.
    assert stop["xy_minimum_degrees"] == -145.
    assert stop["xy_maximum_degrees"] == -35.
    assert stop["endpoints_inclusive"] and stop["symmetric_about_up"]
    assert palm["xy_minimum_degrees"] == -160.
    assert palm["xy_maximum_degrees"] == -20.
    assert palm["maximum_tilt_from_up_degrees"] == 70.
    assert palm["endpoints_inclusive"] and palm["symmetric_about_up"]
    assert palm["outside_range"] == "no_gesture"
    for policy in (stop, palm):
        assert policy["angle_space"] == "source image XY pixels"


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("spread", [0., 20.])
@pytest.mark.parametrize("tilt", [-90., -70., -55., -40., 0., 40., 55., 70., 90.])
@pytest.mark.parametrize("gesture", ["stop_inverted", "open_palm"])
@pytest.mark.parametrize("shape", [(600, 600, 3), (540, 960, 3)])
def test_trained_onnx_supports_both_hands_and_spread(mirror, spread, tilt, gesture, shape):
    from research.v18_20 import predict_onnx
    points, side = stop_pose(tilt, mirror, spread, shape)
    if gesture == "open_palm":
        side = "Right" if side == "Left" else "Left"
    probabilities, mass = predict_onnx(PROJECT_ROOT / "artifacts/stop_inverted/candidate.onnx",
                                      landmarks_to_feature(points)[None])
    config = RuntimeConfig()
    row, details = GeometryResolver(config).resolve(probabilities[0], points, handedness=side,
                                                    handedness_confidence=.99, image_shape=shape)
    gate = TemporalGate(config)
    for now in (1., 1.1, 1.21, 1.61):
        decision = gate.update(row, now=now, **gate_arguments(details, {"known_gesture_mass": float(mass[0])}))
    if abs(tilt) > (70. if gesture == "open_palm" else 55.):
        assert not decision.execute and decision.predicted_gesture == "no_gesture"
    else:
        assert decision.execute and decision.predicted_gesture == gesture


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("spread", [0., 20.])
@pytest.mark.parametrize("tracking_setup", [False, True])
@pytest.mark.parametrize("tilt", [-55., 0., 55.])
def test_live_pipeline_selects_stop_over_invalid_peer_and_rejects_lowering(mirror, spread, tracking_setup, tilt):
    import threading
    from types import SimpleNamespace
    import cv2
    from backend.model_runtime import InferenceEngine, ModelManager, RuntimeSession
    from backend.tests.test_hand_detection import simulated_detector
    from backend.tests.test_reported_two_hand_palm import reported_hands, world_for_scale, SHAPE

    config = load_runtime_config()
    manager = ModelManager(config)
    assert manager.ready, manager.errors
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine.model_manager = config, manager
    engine.resolver, engine._lock = GeometryResolver(config), threading.Lock()
    engine.detector = simulated_detector()
    points, stop_side = stop_pose(tilt, mirror, spread, SHAPE)
    scene = reported_hands(stop_side == "Right", scene=2)
    idle_side = "Left" if stop_side == "Right" else "Right"
    center = scene[stop_side].mean(0)
    scene[stop_side] = points + center - points.mean(0)

    def candidates(rgb, view, *, video):
        return [(.1, pose.copy(), SimpleNamespace(category_name=side, score=.99),
                 world_for_scale(pose, 1300. if side == idle_side else 900.))
                for side, pose in scene.items()]

    engine.detector._run_candidates = candidates
    frame = cv2.imencode(".jpg", np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    if tracking_setup:
        session.follow_object.start(now=1.)
    results = [engine.process_frame(frame, session, now=1 + tick * .1) for tick in range(15)]
    assert results[-1]["runtime_prediction"] == "stop_inverted"
    assert results[-1]["quality"]["detector"]["selected_handedness"] == stop_side
    assert any(result["runtime_action"] == "Return to main position" for result in results)
    assert sum(result["runtime_action"] == "Return to main position" for result in results) == 1
    assert not session.follow_object.active
    assert all(hand["gesture"] == "no_gesture" for hand in results[-1]["hands"] if not hand["selected"])
    outside, _ = stop_pose(91., mirror, spread, SHAPE)
    scene[stop_side] = outside + center - outside.mean(0)
    rejected = [engine.process_frame(frame, session, now=2.5 + tick * .1) for tick in range(10)]
    assert rejected[-1]["runtime_prediction"] == "no_gesture"
    assert all(result["runtime_action"] == "Wait / No Action" for result in rejected)
