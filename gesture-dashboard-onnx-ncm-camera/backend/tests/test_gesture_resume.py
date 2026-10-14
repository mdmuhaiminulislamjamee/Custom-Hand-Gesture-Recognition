"""Valid hand observations must resume after movement or context rejection."""
import numpy as np
import pytest
from types import SimpleNamespace
import threading

from backend.config import RuntimeConfig
from backend.hand_selection import AutomaticHandSelector
from backend.face_contact import (FACE_OVERLAP_REASON, contact_evidence,
                                  face_rejection_reason, update_face_references)
from backend.tests.test_hand_selection import observation, feed, SHAPE
from backend.tests.test_face_contact import reported_contact_scene


@pytest.fixture(scope="module")
def shipped_manager():
    from backend.config import load_runtime_config
    from backend.model_runtime import ModelManager
    manager = ModelManager(load_runtime_config())
    assert manager.ready, manager.errors
    return manager


@pytest.mark.parametrize("label,index,side", [
    ("left", 0, "Right"), ("right", 1, "Right"), ("up", 2, "Right"),
    ("down", 3, "Right"), ("open_palm", 1600, "Right"), ("like", 2000, "Right"),
    ("dorsal", 4000, "Left"), ("ok", 2400, "Right"), ("fist", 2900, "Right"),
    ("thumb_down", 3600, "Right"), ("peace", 24000, "Right"), ("rock", 20000, "Right"),
])
def test_shipped_classifier_commands_survive_live_selection_and_reacquisition(
        label, index, side, shipped_manager, monkeypatch):
    """Fixed recorded 2D samples through ONNX and the actual live pipeline.

    The world joints are a planar fixture for the depth availability contract;
    this smoke test does not claim measured depth or image detection accuracy.
    """
    import cv2
    from backend.config import PROJECT_ROOT
    from backend.geometry import GeometryResolver
    from backend.model_runtime import InferenceEngine, RuntimeSession
    from backend.tests.test_hand_detection import simulated_detector
    import backend.model_runtime as module
    source = ("data/ten_gesture/hagrid_train.npz" if label in {"peace", "rock"}
              else "artifacts/ten_gesture/public_test.npz")
    with np.load(PROJECT_ROOT / source, allow_pickle=False) as data:
        points = data["landmarks"][index].copy()
    points -= points[0]
    points *= 160 / max(np.ptp(points, axis=0))
    points += [200, 300]
    points /= [640, 480]
    world = np.column_stack((points - points[0], np.zeros(21)))
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config = shipped_manager.config
    engine.model_manager = shipped_manager
    engine.resolver = GeometryResolver(engine.config)
    engine._lock = threading.Lock()
    engine.detector = simulated_detector()
    engine.detector._run_candidates = lambda rgb, view, *, video: [
        (.1, points.copy(), SimpleNamespace(category_name=side, score=.99), world)]
    clock = [1.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    frame = cv2.imencode(".jpg", np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(engine.config)
    outputs = []
    for i in range(10):
        clock[0] = 1 + i * .1
        outputs.append(engine.process_frame(frame, session))
    assert outputs[-1]["runtime_prediction"] == label
    assert any(r["runtime_action"] != "Wait / No Action" for r in outputs)
    owner = outputs[-1]["hand_selection"]["selected_track_id"]
    points[:, 0] += .4
    clock[0] = 2.
    reacquired = engine.process_frame(frame, session)
    assert reacquired["hand_selection"]["selected_track_id"] == owner
    assert reacquired["quality"]["detector"]["reacquired"]
    assert reacquired["stable_frames"] == 1
    assert reacquired["runtime_action"] == "Wait / No Action"
    for i in range(1, 10):
        clock[0] = 2 + i * .1
        output = engine.process_frame(frame, session)
    assert output["runtime_prediction"] == label
    assert output["hand_selection"]["selected_track_id"] == owner
    assert output["stable_frames"] >= 3


@pytest.mark.parametrize("label", RuntimeConfig().class_names)
def test_all_commands_reacquire_same_hand_after_fast_screen_movement(label):
    selector = AutomaticHandSelector(RuntimeConfig())
    owner = feed(selector, [observation(label=label)])[-1][1]["selected_track_id"]
    moved = observation(x=.8, label=label)
    first, selection, _ = selector.update([moved], SHAPE, 1.)
    assert first is moved and moved["identity_reacquired"]
    assert selection["selected_track_id"] == owner
    assert len(selector.tracks[owner].gate._history) == 1
    results = feed(selector, [moved], start=11, count=10)
    assert results[-1][0] is moved
    assert results[-1][1]["selected_track_id"] == owner
    assert results[-1][2][0]["gesture"] == label
    assert results[-1][2][0]["eligible"]


@pytest.mark.parametrize("label", RuntimeConfig().class_names)
def test_paused_owner_keeps_moving_and_resumes_all_commands(label):
    selector = AutomaticHandSelector(RuntimeConfig())
    owner = feed(selector, [observation(label=label)])[-1][1]["selected_track_id"]
    for frame, x in enumerate(np.linspace(.3, .8, 14)):
        paused = observation(x=x, label=label)
        paused["rejection_reason"] = FACE_OVERLAP_REASON
        chosen, selection, previews = selector.update([paused], SHAPE, 1 + frame * .1)
        assert chosen is None
        assert selection["active_track_id"] == owner
        assert previews[0]["track_id"] == owner
        np.testing.assert_array_equal(selector.tracks[owner].points, paused["points"])
    resumed = observation(x=.8, label=label)
    results = feed(selector, [resumed], start=24, count=10)
    assert results[-1][0] is resumed
    assert results[-1][1]["selected_track_id"] == owner
    assert results[-1][2][0]["eligible"]


@pytest.mark.parametrize("label", RuntimeConfig().class_names)
def test_slower_valid_frames_can_qualify_closer_hand(label):
    selector = AutomaticHandSelector(RuntimeConfig())
    near = observation(scale=.6, label=label)
    far = observation(x=.75, scale=.3, hand="Left", label=label)
    for frame in range(12):
        chosen, selection, previews = selector.update([near, far], SHAPE, 1 + frame * .4)
    assert chosen is near
    assert selection["active_track_id"] is not None
    assert previews[0]["gesture"] == label and previews[0]["eligible"]


def test_same_side_ambiguity_cannot_reassign_owner_after_large_movement():
    selector = AutomaticHandSelector(RuntimeConfig())
    owner = feed(selector, [observation()])[-1][1]["selected_track_id"]
    a, b = observation(x=.65, scale=.6), observation(x=.85, scale=.3)
    chosen, selection, _ = selector.update([a, b], SHAPE, 1.)
    assert chosen is None
    assert selection["active_track_id"] == owner


def test_new_face_fit_on_hand_does_not_block_a_valid_gesture_without_head_evidence():
    state = {}
    for frame in range(12):
        now = 1 + frame * .1
        hand, faces, shape = reported_contact_scene(now)
        update_face_references(faces, [hand], shape, state=state, now=now)
        evidence, _ = contact_evidence(hand, shape, faces, [], now=now)
        assert faces[0]["clear_reference"] is False
        assert evidence["unverified_face_overlap_count"] == 1
        assert face_rejection_reason(evidence) is None


def test_face_seen_before_hand_still_suppresses_contact_without_pose_depth():
    hand, faces, shape = reported_contact_scene(1.)
    state = {}
    update_face_references(faces, [], shape, state=state, now=1.)
    for frame in range(1, 12):
        now = 1 + frame * .1
        hand, faces, shape = reported_contact_scene(now)
        update_face_references(faces, [hand], shape, state=state, now=now)
        evidence, _ = contact_evidence(hand, shape, faces, [], now=now)
        assert faces[0]["clear_reference"] is True
        assert face_rejection_reason(evidence) == FACE_OVERLAP_REASON


def test_old_clear_face_cannot_validate_new_face_fit_after_it_disappears():
    hand, faces, shape = reported_contact_scene(1.)
    state = {}
    update_face_references(faces, [], shape, state=state, now=1.)
    hand, faces, shape = reported_contact_scene(2.)
    update_face_references(faces, [hand], shape, state=state, now=2.)
    evidence, _ = contact_evidence(hand, shape, faces, [], now=2.)
    assert faces[0]["clear_reference"] is False
    assert face_rejection_reason(evidence) is None
