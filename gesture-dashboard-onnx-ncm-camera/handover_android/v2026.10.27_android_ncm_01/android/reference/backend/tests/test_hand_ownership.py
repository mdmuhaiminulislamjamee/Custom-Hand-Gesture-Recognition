"""Two-hand ownership follows distance, with valid gestures breaking equal ties."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from backend.config import ACTIVE_CLASS_NAMES, RuntimeConfig
from backend.hand_selection import AutomaticHandSelector, SelectionPolicy
from backend.model_runtime import InferenceEngine, RuntimeSession
from backend.tests.test_hand_detection import simulated_detector
from backend.tests.test_hand_selection import SHAPE, feed, observation, projected_world


@pytest.mark.parametrize("label", ACTIVE_CLASS_NAMES)
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
def test_equal_distance_selects_gesture_over_other_visible_unusable_pose(mass, reason):
    selector = AutomaticHandSelector(RuntimeConfig())
    peer = observation(x=.75, hand="Left", mass=mass)
    peer["rejection_reason"] = reason
    gesture = observation()
    chosen, selection, previews = feed(selector, [gesture, peer])[-1]
    assert chosen is gesture
    assert selector.active_id == selection["selected_track_id"]
    assert previews[0]["eligible"]
    assert previews[1]["gesture"] == "no_gesture"
    assert not previews[1]["selected"] and not previews[1]["eligible"]


@pytest.mark.parametrize("label", ACTIVE_CLASS_NAMES)
@pytest.mark.parametrize("gesture_hand", ["Left", "Right"])
@pytest.mark.parametrize("world_scale", [False, True])
def test_equal_distance_selects_only_valid_gesture_for_either_hand(label, gesture_hand, world_scale):
    selector = AutomaticHandSelector(RuntimeConfig())
    gesture = observation(label=label, hand=gesture_hand)
    idle = observation(x=.75, hand="Right" if gesture_hand == "Left" else "Left", mass=.01)
    if world_scale:
        projected_world(gesture, 1200.)
        projected_world(idle, 1200.)
    results = []
    for tick in range(14):
        rows = [gesture, idle] if tick % 2 else [idle, gesture]
        chosen, selection, previews = selector.update(rows, SHAPE, tick * .1)
        results.append((chosen, selection, previews))
        idle_preview = previews[0 if rows[0] is idle else 1]
        assert idle_preview["gesture"] == "no_gesture"
        assert not idle_preview["selected"] and not idle_preview["eligible"]
    assert all(chosen is None for chosen, _, _ in results[:3])
    assert results[-1][0] is gesture
    assert results[-1][1]["distance_method"] == (
        "projected world landmarks" if world_scale else "apparent palm size")
    assert len({selection["selected_track_id"] for chosen, selection, _ in results if chosen}) == 1
    assert results[-1][2][0]["gesture"] == label
    assert results[-1][2][0]["eligible"]


@pytest.mark.parametrize("idle_kind", ["dorsal", "missing_probabilities", "invalid_pose", "low_confidence"])
def test_equal_distance_rejected_gesture_cannot_compete_with_active_class(idle_kind):
    selector = AutomaticHandSelector(RuntimeConfig())
    gesture = observation(label="open_palm")
    idle = observation(x=.75, hand="Left", label="dorsal" if idle_kind == "dorsal" else "like")
    if idle_kind == "missing_probabilities":
        idle.pop("probabilities")
    elif idle_kind == "invalid_pose":
        idle["gate_args"]["pose_valid"] = False
    elif idle_kind == "low_confidence":
        idle["probabilities"][:] = 1 / len(idle["probabilities"])
    chosen, _, previews = feed(selector, [idle, gesture])[-1]
    assert chosen is gesture
    assert previews[1]["gesture"] == "open_palm" and previews[1]["eligible"]
    assert previews[0]["gesture"] == "no_gesture" and not previews[0]["eligible"]


@pytest.mark.parametrize("label,mass", [("like", .01), ("dorsal", 1.)])
def test_equal_distance_with_no_valid_gesture_never_selects(label, mass):
    selector = AutomaticHandSelector(RuntimeConfig())
    results = feed(selector, [observation(label=label, mass=mass),
                             observation(x=.75, hand="Left", label=label, mass=mass)])
    assert all(chosen is None for chosen, _, _ in results)
    assert selector.active_id is None
    assert all(not preview["selected"] and not preview["eligible"]
               for _, _, previews in results for preview in previews)


@pytest.mark.parametrize("gesture_hand", ["Left", "Right"])
def test_equal_distance_transfers_from_idle_owner_after_confirmation(gesture_hand):
    selector = AutomaticHandSelector(RuntimeConfig())
    owner = observation(label="fist", hand="Right" if gesture_hand == "Left" else "Left")
    peer = observation(x=.75, hand=gesture_hand, label="open_palm", mass=.01)
    owner_id = feed(selector, [owner, peer])[-1][1]["selected_track_id"]
    owner["gate_args"]["known_gesture_mass"] = .01
    peer["gate_args"]["known_gesture_mass"] = 1.
    results = feed(selector, [owner, peer], start=10, count=10)
    assert results[0][0] is owner
    assert any(selection["reason"] == "confirming_gesture_hand_switch"
               and selection["switch_pending"] for _, selection, _ in results)
    handoff = next(result for result in results if result[1]["ownership_reset"])
    assert handoff[0] is peer
    assert not handoff[2][1]["eligible"]
    assert len(selector.tracks[owner_id].gate._history) == 0
    assert results[-1][0] is peer
    assert results[-1][1]["selected_track_id"] != owner_id
    assert results[-1][2][1]["gesture"] == "open_palm"
    assert results[-1][2][1]["eligible"]


def test_brief_idle_owner_cannot_transfer_equal_distance_ownership():
    selector = AutomaticHandSelector(RuntimeConfig())
    owner = observation(label="like")
    peer = observation(x=.75, hand="Left", label="peace")
    owner_id = feed(selector, [owner])[-1][1]["selected_track_id"]
    feed(selector, [owner, peer], start=10)
    owner["gate_args"]["known_gesture_mass"] = .01
    results = feed(selector, [owner, peer], start=20, count=3)
    owner["gate_args"]["known_gesture_mass"] = 1.
    results += feed(selector, [owner, peer], start=23)
    assert all(chosen is owner for chosen, _, _ in results)
    assert all(selection["selected_track_id"] == owner_id
               and not selection["ownership_reset"] for _, selection, _ in results)


def test_equal_distance_valid_peer_still_building_hold_prevents_initial_choice():
    selector = AutomaticHandSelector(RuntimeConfig())
    gesture = observation(label="open_palm")
    peer = observation(x=.75, hand="Left", mass=.01)
    # Let the unique gesture finish qualification and start its selection hold.
    feed(selector, [gesture, peer], count=4)
    peer["gate_args"]["known_gesture_mass"] = 1.
    results = feed(selector, [gesture, peer], start=4)
    assert all(chosen is None for chosen, _, _ in results)
    assert selector.active_id is None


def test_uncertain_world_distance_with_unequal_palms_selects_the_only_valid_gesture():
    selector = AutomaticHandSelector(RuntimeConfig())
    gesture = projected_world(observation(label="open_palm", scale=.5), 1280.)
    idle = projected_world(observation(x=.75, hand="Left", mass=.01), 1200.)
    results = feed(selector, [gesture, idle])
    assert results[-1][0] is gesture
    assert results[-1][1]["gesture_priority"]
    assert results[-1][2][0]["eligible"]


@pytest.mark.parametrize("label", ACTIVE_CLASS_NAMES)
@pytest.mark.parametrize("gesture_hand", ["Left", "Right"])
def test_unrecognized_owner_releases_to_valid_peer_while_distance_is_uncertain(label, gesture_hand):
    selector = AutomaticHandSelector(RuntimeConfig())
    idle_hand = "Right" if gesture_hand == "Left" else "Left"
    owner = observation(scale=.6, hand=idle_hand)
    peer = observation(x=.75, scale=.3, hand=gesture_hand, label=label)
    owner_id = feed(selector, [owner, peer])[-1][1]["selected_track_id"]
    invalid = observation(scale=.6, hand=idle_hand, mass=0.)
    # Bent / foreshortened fingers can make palm segments disagree. Distance
    # uncertainty must not stop the other hand from being qualified at all.
    invalid["metrics"]["palm_segments_px"] = [70., 35., 75., 20.]
    peer["metrics"]["palm_segments_px"] = [35., 35., 35., 35.]
    results = feed(selector, [invalid, peer], start=10, count=14)
    handoff = next(result for result in results if result[1]["ownership_reset"])
    assert handoff[0] is peer
    assert handoff[1]["selected_track_id"] != owner_id
    assert not handoff[2][1]["eligible"]
    assert results[-1][0] is peer
    assert results[-1][2][1]["gesture"] == label
    assert results[-1][2][1]["eligible"]
    assert results[-1][2][0]["gesture"] == "no_gesture"
    assert not results[-1][1]["switch_pending"]


@pytest.mark.parametrize("projected_ratio", [1.04, 1.07, 1.12])
@pytest.mark.parametrize("gesture_hand", ["Left", "Right"])
def test_similar_palms_transfer_from_rejected_owner_despite_small_world_scale_difference(projected_ratio, gesture_hand):
    selector = AutomaticHandSelector(RuntimeConfig())
    idle_hand = "Right" if gesture_hand == "Left" else "Left"
    owner = observation(label="like", hand=idle_hand)
    owner_id = feed(selector, [owner])[-1][1]["selected_track_id"]
    # The selected hand turns dorsal. The other hand has a stable valid palm.
    # A small world reconstruction mismatch must not keep the negative selected
    # when their four image palm segments agree that they are similarly sized.
    idle = projected_world(observation(scale=.412, label="dorsal", hand=idle_hand), 1200. * projected_ratio)
    palm = projected_world(observation(x=.75, label="open_palm", hand=gesture_hand), 1200.)
    results = feed(selector, [idle, palm], start=10, count=18)
    handoff = next(result for result in results if result[1]["ownership_reset"])
    assert handoff[0] is palm
    assert handoff[1]["selected_track_id"] != owner_id
    assert not handoff[2][1]["eligible"]
    assert results[-1][0] is palm
    assert results[-1][2][1]["gesture"] == "open_palm"
    assert results[-1][2][1]["eligible"]
    assert results[-1][2][0]["gesture"] == "no_gesture"
    # Subsequent frames must retain the valid palm instead of switching back
    # to the rejected hand on its small projected-scale lead.
    assert all(chosen is palm and not selection["switch_pending"]
               for chosen, selection, _ in results if selection["selected_track_id"] == palm["track_id"])


@pytest.mark.parametrize("projected_ratio", [1.07, 1.12])
@pytest.mark.parametrize("label", ACTIVE_CLASS_NAMES)
def test_similar_distance_unique_gesture_can_acquire_without_previous_owner(projected_ratio, label):
    selector = AutomaticHandSelector(RuntimeConfig())
    idle = projected_world(observation(label="dorsal"), 1200. * projected_ratio)
    gesture = projected_world(observation(x=.75, hand="Left", label=label), 1200.)
    results = feed(selector, [idle, gesture], count=16)
    assert all(chosen is None for chosen, _, _ in results[:3])
    assert results[-1][0] is gesture
    assert results[-1][2][1]["gesture"] == label
    assert results[-1][2][1]["eligible"]


@pytest.mark.parametrize("world_ratio,palm_ratio", [(1.17, 1.), (1.30, 1.), (1.12, 1.20)])
def test_valid_gesture_takes_priority_over_idle_hand_at_different_distances(world_ratio, palm_ratio):
    selector = AutomaticHandSelector(RuntimeConfig())
    idle = projected_world(observation(label="dorsal", scale=.4 * palm_ratio), 1200. * world_ratio)
    palm = projected_world(observation(x=.75, hand="Left", label="open_palm"), 1200.)
    chosen, selection, previews = feed(selector, [idle, palm])[-1]
    assert chosen is palm
    assert selection["gesture_priority"]
    assert previews[1]["gesture"] == "open_palm" and previews[1]["eligible"]
    assert previews[0]["gesture"] == "no_gesture" and not previews[0]["eligible"]


@pytest.mark.parametrize("arrangement", ["overlap", "inconsistent_palm_segments"])
def test_unique_gesture_ignores_palm_size_consistency_but_respects_overlap(arrangement):
    selector = AutomaticHandSelector(RuntimeConfig())
    idle = projected_world(observation(label="dorsal"), 1344.)
    palm = projected_world(observation(x=.31 if arrangement == "overlap" else .75,
                                       hand="Left", label="open_palm"), 1200.)
    if arrangement == "inconsistent_palm_segments":
        idle["metrics"]["palm_segments_px"] = [70., 35., 75., 20.]
        palm["metrics"]["palm_segments_px"] = [35., 35., 35., 35.]
    chosen, selection, previews = feed(selector, [idle, palm])[-1]
    if arrangement == "overlap":
        assert chosen is idle
        assert not selection["gesture_priority"]
        assert all(not preview["eligible"] for preview in previews)
    else:
        assert chosen is palm
        assert selection["gesture_priority"]
        assert previews[1]["eligible"]


def test_small_world_scale_jitter_cannot_starve_the_only_valid_palm():
    selector = AutomaticHandSelector(RuntimeConfig())
    idle = projected_world(observation(label="dorsal"), 1200.)
    palm = projected_world(observation(x=.75, hand="Left", label="open_palm"), 1200.)
    results = []
    for tick in range(24):
        projected_world(idle, 1200. * (1.04, 1.08, 1.12)[tick % 3])
        results.append(selector.update([idle, palm], SHAPE, tick * .1))
    assert results[-1][0] is palm
    assert results[-1][2][1]["eligible"]
    assert all(chosen is palm and not selection["switch_pending"]
               for chosen, selection, _ in results if selection["selected_track_id"] == palm["track_id"])


def test_small_world_scale_lead_still_selects_nearer_hand_when_both_gestures_are_valid():
    selector = AutomaticHandSelector(RuntimeConfig())
    near = projected_world(observation(label="like"), 1344.)
    far = projected_world(observation(x=.75, hand="Left", label="open_palm"), 1200.)
    assert feed(selector, [near, far])[-1][0] is near


def test_equal_distance_unique_gesture_does_not_bypass_overlap_checks():
    selector = AutomaticHandSelector(RuntimeConfig())
    results = feed(selector, [observation(x=.4), observation(x=.41, hand="Left", mass=.01)])
    assert all(chosen is None for chosen, _, _ in results)
    assert results[-1][1]["reason"] == "ambiguous_tracks"


def test_unique_gesture_does_not_wait_for_filtered_distance_history():
    selector = AutomaticHandSelector(RuntimeConfig(), SelectionPolicy(distance_smoothing_alpha=.1))
    gesture = projected_world(observation(label="open_palm"), 1600.)
    peer = projected_world(observation(x=.75, hand="Left"), 1200.)
    assert all(chosen is None for chosen, _, _ in feed(selector, [gesture, peer], count=2))
    projected_world(gesture, 1200.)
    peer["gate_args"]["known_gesture_mass"] = .01
    results = feed(selector, [gesture, peer], start=2, count=10)
    assert all(chosen is None for chosen, _, _ in results[:3])
    assert results[-1][0] is gesture
    assert results[-1][1]["gesture_priority"]


def test_high_frame_rate_cannot_skip_equal_distance_selection_hold():
    selector = AutomaticHandSelector(RuntimeConfig())
    gesture = observation(label="open_palm")
    peer = observation(x=.75, hand="Left", mass=.01)
    for frame in range(40):
        chosen, _, _ = selector.update([gesture, peer], SHAPE, frame / 100)
        assert chosen is None
    results = feed(selector, [gesture, peer], start=4, count=10)
    assert results[-1][0] is gesture


@pytest.mark.parametrize("gesture_hand", ["Left", "Right"])
@pytest.mark.parametrize("label", ["open_palm", "left", "fist"])
def test_runtime_equal_distance_selects_valid_gesture_and_resets_hold_on_transfer(monkeypatch, gesture_hand, label):
    import cv2

    config = RuntimeConfig()
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine._lock = config, threading.Lock()
    probabilities = observation(label=label)["probabilities"]
    quality = {"known_gesture_mass": 1.0}
    engine.model_manager = SimpleNamespace(
        ready=True, models={"ONNX": object()}, resolve_name=lambda name: "ONNX",
        predict_detailed=lambda *args: (probabilities.copy(), "ONNX", .01, quality),
        predict_many=lambda *args: {"ONNX": (probabilities.copy(), .01, quality)},
    )
    valid_hand = [gesture_hand]

    def resolve(probabilities, points, **kwargs):
        return probabilities, {"pose_validation": {
            "valid": kwargs.get("handedness") == valid_hand[0],
            "reason": "unknown hand shape",
        }}

    engine.resolver = SimpleNamespace(resolve=resolve)
    engine.detector = simulated_detector()
    other_hand = "Right" if gesture_hand == "Left" else "Left"
    rows = [observation(hand=gesture_hand), observation(x=.75, hand=other_hand)]

    def detect(rgb, *, state, candidate_score):
        for row in rows:
            candidate_score(row)
        engine.detector._last_diagnostics = {"candidate_count": 2}
        return rows

    monkeypatch.setattr(engine.detector, "detect_candidates", detect)
    frame = cv2.imencode(".jpg", np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    outputs = [engine.process_frame(frame, session, now=1 + tick * .1) for tick in range(14)]
    owner_id = outputs[-1]["hand_selection"]["selected_track_id"]
    assert owner_id is not None
    assert outputs[-1]["runtime_prediction"] == label
    assert outputs[-1]["hands"][0]["selected"]
    assert outputs[-1]["hands"][1]["gesture"] == "no_gesture"
    assert any(output["runtime_action"] != "Wait / No Action" for output in outputs)

    valid_hand[0] = other_hand
    probabilities[:] = observation(label="peace")["probabilities"]
    outputs = [engine.process_frame(frame, session, now=1 + tick * .1) for tick in range(14, 28)]
    pending = [output for output in outputs if output["hand_selection"]["switch_pending"]]
    assert pending
    assert all(output["runtime_action"] == "Wait / No Action" for output in pending)
    assert all(output["action_reason"] == "confirming gesture hand" for output in pending)
    handoff = next(output for output in outputs if output["hand_selection"]["ownership_reset"])
    assert handoff["stable_frames"] == 1
    assert handoff["runtime_action"] == "Wait / No Action"
    assert outputs[-1]["hand_selection"]["selected_track_id"] != owner_id
    assert outputs[-1]["runtime_prediction"] == "peace"
    assert outputs[-1]["hands"][0]["gesture"] == "no_gesture"
    assert outputs[-1]["hands"][1]["selected"]
    assert any(output["runtime_action"] == config.gesture_to_action["peace"] for output in outputs)


def test_farther_valid_gesture_wins_against_nearer_idle_hand():
    selector = AutomaticHandSelector(RuntimeConfig())
    far = observation(scale=.3)
    near_idle = observation(x=.75, scale=.6, hand="Left", mass=.01)
    results = feed(selector, [far, near_idle])
    assert results[-1][0] is far
    assert results[-1][2][0]["gesture"] == "like"
    assert results[-1][2][0]["eligible"]
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
    assert chosen is turned
    assert selection["active_track_id"] == owner
    assert selection["switch_pending"]
    assert previews[0]["selected"]
    assert all(not item["eligible"] for item in previews)
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


def test_invalid_nearer_owner_releases_to_other_hands_valid_gesture():
    selector = AutomaticHandSelector(RuntimeConfig())
    active, peer = observation(scale=.6), observation(x=.75, scale=.3, hand="Left", label="left")
    owner = feed(selector, [active, peer])[-1][1]["selected_track_id"]
    invalid = observation(scale=.6, mass=.01)
    results = feed(selector, [invalid, peer], start=10, count=25)
    assert results[0][0] is invalid
    assert results[0][1]["switch_pending"]
    handoff = next(result for result in results if result[1]["ownership_reset"])
    assert handoff[0] is peer
    assert not handoff[2][1]["eligible"]
    assert selector.active_id != owner
    assert results[-1][0] is peer
    assert results[-1][2][1]["gesture"] == "left" and results[-1][2][1]["eligible"]


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
