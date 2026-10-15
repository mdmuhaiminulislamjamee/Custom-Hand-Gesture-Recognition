"""Two-hand ownership follows distance and identity, independent of labels."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.hand_selection import AutomaticHandSelector
from backend.model_runtime import InferenceEngine, RuntimeSession
from backend.tests.test_hand_detection import simulated_detector
from backend.tests.test_hand_selection import SHAPE, feed, observation


@pytest.mark.parametrize("label", RuntimeConfig().class_names)
@pytest.mark.parametrize("same_gesture", [False, True])
def test_equal_distance_blocks_identical_and_different_gestures(label, same_gesture):
    selector = AutomaticHandSelector(RuntimeConfig())
    other = label if same_gesture else ("left" if label != "left" else "like")
    results = feed(selector, [observation(label=label),
                             observation(x=.75, hand="Left", label=other)])
    assert all(chosen is None for chosen, _, _ in results)
    assert selector.active_id is None
    assert all(not preview["selected"] and not preview["eligible"]
               for _, _, previews in results for preview in previews)


@pytest.mark.parametrize("mass,reason", [(.01, None), (1., "face_contact_pending")])
def test_equal_distance_also_blocks_when_other_visible_hand_has_no_usable_gesture(mass, reason):
    selector = AutomaticHandSelector(RuntimeConfig())
    peer = observation(x=.75, hand="Left", mass=mass)
    peer["rejection_reason"] = reason
    assert all(chosen is None for chosen, _, _ in feed(selector, [observation(), peer]))
    assert selector.active_id is None


def test_farther_gesture_does_not_win_against_nearer_idle_hand():
    selector = AutomaticHandSelector(RuntimeConfig())
    far = observation(scale=.3)
    near_idle = observation(x=.75, scale=.6, hand="Left", mass=.01)
    results = feed(selector, [far, near_idle])
    assert results[-1][0] is near_idle
    assert results[-1][2][0]["gesture"] == "no_gesture"
    assert results[-1][2][1]["gesture"] == "no_gesture"
    assert not results[-1][2][1]["eligible"]


@pytest.mark.parametrize("owner_hand", ["Left", "Right"])
def test_only_qualified_nearer_hand_acquires_initial_ownership(owner_hand):
    selector = AutomaticHandSelector(RuntimeConfig())
    other_hand = "Right" if owner_hand == "Left" else "Left"
    near = observation(scale=.6, hand=owner_hand, label="left")
    far = observation(x=.75, scale=.3, hand=other_hand, label="like")
    results = feed(selector, [far, near])
    assert results[-1][0] is near
    owner = results[-1][1]["selected_track_id"]
    assert owner is not None and selector.active_id == owner
    assert not results[-1][2][0]["selected"]


@pytest.mark.parametrize("challenger_label", ["like", "left", "fist"])
def test_late_closer_challenger_takes_selection_after_confirmation(challenger_label):
    selector = AutomaticHandSelector(RuntimeConfig())
    active = observation(scale=.35, label="like")
    first = feed(selector, [active])[-1]
    owner = first[1]["selected_track_id"]
    challenger = observation(x=.75, scale=.65, hand="Left", label=challenger_label)
    results = feed(selector, [challenger, active], start=10, count=25)
    assert results[0][0] is active
    assert results[0][1]["switch_pending"]
    assert not results[0][2][1]["eligible"]
    assert results[-1][0] is challenger
    assert results[-1][1]["selected_track_id"] != owner
    assert selector.active_id == results[-1][1]["selected_track_id"]


def test_previously_farther_peer_moving_forward_replaces_owner():
    selector = AutomaticHandSelector(RuntimeConfig())
    active = observation(scale=.6, label="left")
    peer = observation(x=.75, scale=.3, hand="Left", label="like")
    owner = feed(selector, [active, peer])[-1][1]["selected_track_id"]
    advanced = observation(x=.75, scale=.9, hand="Left", label="fist")
    results = feed(selector, [advanced, active], start=10, count=15)
    assert results[0][0] is active
    assert results[0][1]["switch_pending"]
    assert results[-1][0] is advanced
    assert results[-1][1]["selected_track_id"] != owner


@pytest.mark.parametrize("returning_hand", ["owner", "peer"])
def test_equal_distance_keeps_tracked_owner_despite_filtered_history(returning_hand):
    selector = AutomaticHandSelector(RuntimeConfig())
    active = observation(scale=.6, label="left")
    peer = observation(x=.75, scale=.3, hand="Left", label="fist")
    first = feed(selector, [active, peer])[-1]
    assert first[0] is active
    scale = .3 if returning_hand == "owner" else .6
    equal_active = observation(scale=scale, label="left")
    equal_peer = observation(x=.75, scale=scale, hand="Left", label="fist")
    chosen, selection, previews = selector.update([equal_active, equal_peer], SHAPE, 1.)
    assert chosen is equal_active
    assert selection["active_track_id"] == first[1]["selected_track_id"]
    assert selector.active_id == first[1]["selected_track_id"]
    assert not selection["ownership_reset"]
    assert previews[0]["selected"] and not previews[1]["eligible"]
    assert previews[1]["gesture"] == "no_gesture"


def test_same_track_stays_selected_after_equal_distance():
    selector = AutomaticHandSelector(RuntimeConfig())
    active, peer = observation(scale=.6), observation(x=.75, scale=.3, hand="Left")
    owner = feed(selector, [active, peer])[-1][1]["selected_track_id"]
    equal = observation(scale=.3)
    chosen, selection, _ = selector.update([equal, peer], SHAPE, 1.)
    assert chosen is equal
    assert selection["selected_track_id"] == owner
    assert not selection["ownership_reset"]
    returning = feed(selector, [active, peer], start=11, count=10)
    assert all(chosen is active for chosen, _, _ in returning)
    assert returning[-1][0] is active
    assert returning[-1][1]["selected_track_id"] == owner


def test_uncertain_orientation_comparison_cannot_keep_active_gestures_running():
    selector = AutomaticHandSelector(RuntimeConfig())
    active = observation(scale=.6)
    peer = observation(x=.75, scale=.3, hand="Left")
    first = feed(selector, [active, peer])[-1]
    assert first[0] is active
    owner = first[1]["selected_track_id"]
    turned = observation(scale=.6)
    unclear_peer = observation(x=.75, scale=.3, hand="Left")
    turned["metrics"]["palm_segments_px"] = [70., 35., 75., 20.]
    unclear_peer["metrics"]["palm_segments_px"] = [35., 35., 35., 35.]
    chosen, selection, previews = selector.update([turned, unclear_peer], SHAPE, 1.)
    assert chosen is None
    assert selection["active_track_id"] == owner
    assert all(not item["selected"] and not item["eligible"] for item in previews)
    restored = feed(selector, [peer, active], start=11, count=10)
    assert all(chosen is None or chosen is active for chosen, _, _ in restored)
    assert all(selection["active_track_id"] == owner for _, selection, _ in restored)
    assert restored[-1][0] is active
    assert restored[-1][1]["selected_track_id"] == owner


def test_owner_gesture_change_keeps_identity_when_peer_performs_same_new_gesture():
    selector = AutomaticHandSelector(RuntimeConfig())
    active, peer = observation(scale=.6), observation(x=.75, scale=.3, hand="Left", label="left")
    owner = feed(selector, [active, peer])[-1][1]["selected_track_id"]
    changed = observation(scale=.6, label="left")
    results = feed(selector, [changed, peer], start=10, count=12)
    assert all(selection["active_track_id"] == owner for _, selection, _ in results)
    assert all(chosen is None or chosen is changed for chosen, _, _ in results)
    assert results[-1][0] is changed


def test_invalid_owner_remains_owned_and_other_hand_cannot_take_over():
    selector = AutomaticHandSelector(RuntimeConfig())
    active, peer = observation(scale=.6), observation(x=.75, scale=.3, hand="Left", label="left")
    owner = feed(selector, [active, peer])[-1][1]["selected_track_id"]
    invalid = observation(scale=.6, mass=.01)
    results = feed(selector, [invalid, peer], start=10, count=25)
    assert all(chosen is invalid for chosen, _, _ in results)
    assert selector.active_id == owner
    assert all(selection["active_track_id"] == owner for _, selection, _ in results)
    assert all(not previews[0]["eligible"] and not previews[1]["eligible"]
               for _, _, previews in results)


def test_missing_owner_expires_and_visible_hand_requalifies():
    selector = AutomaticHandSelector(RuntimeConfig())
    active, peer = observation(scale=.6), observation(x=.75, scale=.3, hand="Left")
    owner = feed(selector, [active, peer])[-1][1]["selected_track_id"]
    results = feed(selector, [peer], start=10, count=25)
    assert all(chosen is None for chosen, _, _ in results[:7])
    assert any(selection["reason"] == "active_hand_expired" and selection["ownership_reset"]
               for _, selection, _ in results)
    assert results[-1][0] is peer
    assert results[-1][1]["selected_track_id"] != owner


@pytest.mark.parametrize("owner_hand", ["Left", "Right"])
def test_confident_opposite_hand_gets_new_identity_after_missing_owner_expires(owner_hand):
    selector = AutomaticHandSelector(RuntimeConfig())
    other_hand = "Right" if owner_hand == "Left" else "Left"
    owner = feed(selector, [observation(scale=.4, hand=owner_hand)])[-1][1]["selected_track_id"]
    other = observation(x=.3, scale=.5, hand=other_hand, label="left")
    results = feed(selector, [other], start=10, count=20)
    assert all(chosen is None for chosen, _, _ in results[:7])
    assert results[-1][0] is other
    assert results[-1][1]["selected_track_id"] != owner


def test_owner_returns_before_expiry_without_transferring_selection():
    selector = AutomaticHandSelector(RuntimeConfig())
    active, peer = observation(scale=.6), observation(x=.75, scale=.3, hand="Left")
    owner = feed(selector, [active, peer])[-1][1]["selected_track_id"]
    assert all(chosen is None for chosen, _, _ in feed(selector, [peer], start=10, count=3))
    returned = feed(selector, [active, peer], start=13, count=8)
    assert returned[-1][0] is active
    assert returned[-1][1]["selected_track_id"] == owner


def test_live_runtime_reselects_visible_hand_after_owner_disappears(monkeypatch):
    import cv2
    import backend.model_runtime as runtime_module

    config = RuntimeConfig()
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine._lock = config, threading.Lock()
    label = ["like"]

    def prediction():
        row = np.eye(len(config.class_names))[config.class_to_idx[label[0]]]
        return row, "ONNX", .01, {"known_gesture_mass": 1.0}

    engine.model_manager = SimpleNamespace(
        ready=True, models={"ONNX": object()}, resolve_name=lambda name: "ONNX",
        predict_detailed=lambda *args: prediction(),
        predict_many=lambda *args: {"ONNX": (prediction()[0], .01, prediction()[3])},
    )
    engine.resolver = SimpleNamespace(resolve=lambda probabilities, points, **kwargs: (
        probabilities, {"pose_validation": {"valid": True}}))
    engine.detector = simulated_detector()
    observed = [observation(scale=.6, label="like")]

    def detect(rgb, *, state, candidate_score):
        for row in observed:
            candidate_score(row)
        engine.detector._last_diagnostics = {"candidate_count": len(observed)}
        return observed

    monkeypatch.setattr(engine.detector, "detect_candidates", detect)
    clock = [0.]
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: clock[0])
    frame = cv2.imencode(".jpg", np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    for index in range(10):
        clock[0] = index * .1
        initial = engine.process_frame(frame, session)
    owner = initial["hand_selection"]["selected_track_id"]
    assert initial["runtime_prediction"] == "like"

    label[0] = "left"
    observed[:] = [observation(x=.75, scale=.3, hand="Left", label="left")]
    results = []
    for index in range(20):
        clock[0] = 1 + index * .1
        results.append(engine.process_frame(frame, session))
    assert any(row["hand_selection"]["reason"] == "active_hand_expired" for row in results)
    assert results[-1]["hand_selection"]["selected_track_id"] != owner
    assert results[-1]["runtime_prediction"] == "left"
    assert any(row["runtime_action"] == "Move Left" for row in results)


def test_uncertain_handedness_flip_does_not_change_established_owners_side():
    selector = AutomaticHandSelector(RuntimeConfig())
    owner = feed(selector, [observation(hand="Right")])[-1][1]["selected_track_id"]
    noisy = observation(hand="Left")
    noisy["handedness_confidence"] = .55
    results = feed(selector, [noisy], start=10, count=10)
    assert all(chosen is noisy for chosen, _, _ in results)
    assert all(selection["selected_track_id"] == owner for _, selection, _ in results)
    assert selector.tracks[owner].handedness == "Right"


def test_moved_owner_reacquires_with_moderate_handedness_confidence():
    selector = AutomaticHandSelector(RuntimeConfig())
    owner = feed(selector, [observation(hand="Right")])[-1][1]["selected_track_id"]
    moved = observation(x=.75, hand="Right")
    moved["handedness_confidence"] = .72
    results = feed(selector, [moved], start=10, count=10)
    assert results[-1][0] is moved
    assert results[-1][1]["selected_track_id"] == owner


def test_completely_empty_scene_eventually_releases_ownership():
    selector = AutomaticHandSelector(RuntimeConfig())
    feed(selector, [observation()])
    assert selector.active_id is not None
    assert all(chosen is None for chosen, _, _ in feed(selector, [], start=10, count=15))
    assert selector.active_id is None
    peer = observation(x=.75, hand="Left")
    assert feed(selector, [peer], start=25)[-1][0] is peer


@pytest.mark.parametrize("reason", ["face_detected_as_hand", "incomplete_hand", "insufficient_hand_pixels"])
def test_invalid_cloud_does_not_count_as_second_same_distance_hand(reason):
    selector = AutomaticHandSelector(RuntimeConfig())
    active = observation()
    cloud = observation(x=.75, hand="Left")
    cloud["rejection_reason"] = reason
    assert feed(selector, [active, cloud])[-1][0] is active


def test_runtime_keeps_same_identity_on_equal_distance(monkeypatch):
    import cv2
    import backend.model_runtime as runtime_module

    config = RuntimeConfig()
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine._lock = config, threading.Lock()
    probabilities = observation()["probabilities"]
    quality = {"known_gesture_mass": 1.0}
    manager = SimpleNamespace(ready=True, models={"ONNX": object()}, resolve_name=lambda name: "ONNX")
    manager.predict_detailed = lambda *args: (probabilities.copy(), "ONNX", .01, quality)
    manager.predict_many = lambda *args: {"ONNX": (probabilities.copy(), .01, quality)}
    engine.model_manager = manager
    engine.resolver = SimpleNamespace(resolve=lambda p, points, **kwargs: (p, {"pose_validation": {"valid": True}}))
    engine.detector = simulated_detector()
    engine.detector._last_diagnostics = {}
    active, peer = observation(scale=.6), observation(x=.75, scale=.3, hand="Left")
    rows = [active, peer]

    def detect(rgb, *, state, candidate_score):
        for row in rows:
            candidate_score(row)
        engine.detector._last_diagnostics = {"candidate_count": len(rows)}
        return rows

    monkeypatch.setattr(engine.detector, "detect_candidates", detect)
    clock = [0.0]
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: clock[0])
    encoded = cv2.imencode(".jpg", np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    initial = []
    for frame in range(15):
        clock[0] = 1 + frame * .1
        initial.append(engine.process_frame(encoded, session))
    assert any(output["runtime_action"] != "Wait / No Action" for output in initial)
    owner = initial[-1]["hand_selection"]["selected_track_id"]
    assert owner is not None

    rows[:] = [observation(scale=.3), peer]
    clock[0] = 2.5
    continued = engine.process_frame(encoded, session)
    assert continued["runtime_prediction"] == "like"
    assert continued["hand_selection"]["active_track_id"] == owner
    assert continued["hand_selection"]["selected_track_id"] == owner
    assert not continued["hand_selection"]["ownership_reset"]
    assert session.selected_track_id == owner

    rows[:] = [active, peer]
    clock[0] = 2.6
    output = engine.process_frame(encoded, session)
    assert output["hand_selection"]["selected_track_id"] == owner
    assert output["quality"]["detector"]["reacquired"] is False

    rows[:] = [observation(scale=.3), observation(x=.75, scale=.37, hand="Left")]
    clock[0] = 2.7
    spike = engine.process_frame(encoded, session)
    assert spike["runtime_prediction"] == "like"
    assert spike["hand_selection"]["selected_track_id"] == owner
    assert spike["hand_selection"]["switch_pending"]
    assert spike["runtime_action"] == "Wait / No Action"

    rows[:] = [observation(scale=.3), peer]
    clock[0] = 2.8
    resumed = engine.process_frame(encoded, session)
    assert resumed["runtime_prediction"] == "like"
    assert resumed["hand_selection"]["selected_track_id"] == owner
    assert not resumed["hand_selection"]["switch_pending"]
    assert not resumed["hand_selection"]["ownership_reset"]
    assert resumed["quality"]["detector"]["reacquired"] is False
