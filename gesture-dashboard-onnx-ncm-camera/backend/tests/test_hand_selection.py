from types import SimpleNamespace

import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.hand_detection import hand_pixel_metrics
from backend.hand_selection import AutomaticHandSelector
from backend.model_runtime import InferenceEngine, RuntimeSession
from backend.tests.test_camera_stability import representative_hand
from backend.tests.test_hand_detection import simulated_detector


SHAPE = (480, 640, 3)


def observation(x=.3, scale=.4, label="like", hand="Right", mass=1.0):
    points = (representative_hand() - [.5, .6]) * scale + [x, .45]
    points = points.astype(np.float32)
    config = RuntimeConfig()
    probabilities = np.full(len(config.class_names), .001)
    probabilities[config.class_to_idx[label]] = 1 - .001 * (len(probabilities) - 1)
    return dict(points=points, world_landmarks=None, metrics=hand_pixel_metrics(points, SHAPE),
                area=float(np.prod(np.ptp(points, axis=0))), handedness=hand, handedness_confidence=.99,
                view=(0, 0, 640, 480, 0, 0), rejection_reason=None,
                probabilities=probabilities, gate_args=dict(known_gesture_mass=mass))


def feed(selector, rows, start=0, count=10):
    results = []
    for frame in range(count):
        results.append(selector.update(rows, SHAPE, (start + frame) * .1))
    return results


def projected_world(row, pixels_per_metre):
    pixels = row["points"] * [SHAPE[1], SHAPE[0]]
    world = np.zeros((21, 3), dtype=np.float32)
    world[:, :2] = (pixels - pixels[0]) / pixels_per_metre
    row["world_landmarks"] = world
    return row


def turned_foreground(label, hand="Right"):
    # Foreground palm is foreshortened, so its raw palm is only about 1.2x
    # the background palm. Projected world/image scale still shows it as 1.33x.
    near = observation(x=.27, scale=.4, label=label, hand=hand)
    near["points"][:, 0] = .27 + (near["points"][:, 0] - .27) * .55
    near["metrics"] = hand_pixel_metrics(near["points"], SHAPE)
    far = observation(x=.75, scale=.3, hand="Left" if hand == "Right" else "Right", mass=.01)
    projected_world(near, 1600.)
    projected_world(far, 1200.)
    return near, far


@pytest.mark.parametrize("label", RuntimeConfig().class_names)
def test_rotated_foreground_hand_can_be_selected_for_every_gesture(label):
    selector = AutomaticHandSelector(RuntimeConfig())
    near, far = turned_foreground(label)
    assert near["metrics"]["palm_scale_px"] < far["metrics"]["palm_scale_px"] * selector.policy.closer_palm_ratio
    result = feed(selector, [far, near])[-1]
    assert result[0] is near
    assert result[1]["distance_method"] == "projected world landmarks"
    assert result[1]["distance_scale_ratio"] > 1.3


def test_projected_world_scale_respects_distance_even_when_raw_palm_order_reverses():
    selector = AutomaticHandSelector(RuntimeConfig())
    near, far = turned_foreground("up")
    far["points"][:, 0] = .75 + (far["points"][:, 0] - .75) * 1.3
    far["metrics"] = hand_pixel_metrics(far["points"], SHAPE)
    projected_world(far, 1200.)
    assert far["metrics"]["palm_scale_px"] > near["metrics"]["palm_scale_px"]
    assert feed(selector, [far, near])[-1][0] is near


def test_same_distance_world_scale_blocks_pose_based_palm_size_difference():
    selector = AutomaticHandSelector(RuntimeConfig())
    near, far = turned_foreground("left")
    projected_world(near, 1200.)
    result = feed(selector, [near, far])[-1]
    assert result[0] is None
    assert result[1]["reason"] == "hands_same_distance"


def test_incoherent_world_landmarks_fall_back_to_conservative_palm_comparison():
    selector = AutomaticHandSelector(RuntimeConfig())
    near, far = turned_foreground("left")
    near["world_landmarks"][5] = np.nan
    result = feed(selector, [near, far])[-1]
    assert result[0] is None
    assert result[1]["distance_method"] == "apparent palm size"


@pytest.mark.parametrize("label", ["left", "up"])
def test_runtime_reports_foreground_gesture_with_two_rotated_hands(monkeypatch, label):
    import threading
    import cv2
    import backend.model_runtime as runtime_module

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
    engine.resolver = SimpleNamespace(resolve=lambda p, points, **kwargs: (
        p, {"pose_validation": {"valid": True}}))
    engine.detector = simulated_detector()
    near, far = turned_foreground(label)

    def detect(rgb, *, state, candidate_score):
        for row in (far, near):
            candidate_score(row)
        engine.detector._last_diagnostics = {"candidate_count": 2}
        return [far, near]

    monkeypatch.setattr(engine.detector, "detect_candidates", detect)
    clock = [0.]
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: clock[0])
    frame = cv2.imencode(".jpg", np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    for index in range(16):
        clock[0] = index * .1
        output = engine.process_frame(frame, session)
    assert output["runtime_prediction"] == label
    assert output["hand_selection"]["distance_method"] == "projected world landmarks"
    assert output["hand_selection"]["selected_track_id"] is not None


def test_two_hands_choose_significantly_larger_palm_only_after_confirmation():
    selector = AutomaticHandSelector(RuntimeConfig())
    near, far = observation(scale=.6), observation(x=.75, scale=.3, hand="Left")
    results = feed(selector, [far, near])
    assert all(result[0] is None for result in results[:3])
    assert results[-1][0] is near
    assert results[-1][1]["selected_track_id"] == 2


def test_nearer_idle_hand_prevents_farther_gesture_acquisition():
    selector = AutomaticHandSelector(RuntimeConfig())
    idle = observation(x=.75, scale=.6, hand="Left", mass=.01)
    gesture = observation(scale=.35)
    chosen, _, previews = feed(selector, [idle, gesture])[-1]
    assert chosen is idle
    assert all(not preview["eligible"] for preview in previews)


def test_equal_depth_valid_gestures_remain_ambiguous():
    selector = AutomaticHandSelector(RuntimeConfig())
    results = feed(selector, [observation(), observation(x=.75, hand="Left")])
    assert all(result[0] is None for result in results)
    assert results[-1][1]["reason"] == "hands_same_distance"


def test_moderate_forward_motion_selects_right_then_left_and_equal_keeps_left():
    selector = AutomaticHandSelector(RuntimeConfig())
    right = observation(x=.27, scale=.48, label="like")
    left = observation(x=.75, scale=.40, hand="Left", label="peace")
    assert feed(selector, [right, left])[-1][0] is right

    # The same hand stays selected while it remains in front, regardless of
    # detection order and a small movement of the other hand.
    left_behind = observation(x=.75, scale=.41, hand="Left", label="peace")
    assert feed(selector, [left_behind, right], start=10, count=4)[-1][0] is right

    left_front = observation(x=.75, scale=.56, hand="Left", label="peace")
    switched = feed(selector, [left_front, right], start=14, count=8)
    assert switched[0][0] is right
    assert switched[0][1]["switch_pending"]
    assert not switched[0][2][1]["eligible"]
    assert switched[-1][0] is left_front
    assert switched[-1][2][0]["gesture"] == "peace"
    assert switched[-1][2][1]["gesture"] == "no_gesture"

    equal = observation(x=.75, scale=.48, hand="Left", label="peace")
    chosen, selection, previews = selector.update([right, equal], SHAPE, 2.2)
    assert chosen is equal
    assert selection["reason"] == "active_hand_locked"
    assert selection["selected_track_id"] == switched[-1][1]["selected_track_id"]
    assert previews[0]["gesture"] == "no_gesture"
    assert previews[1]["selected"]

    right_front = observation(x=.27, scale=.65, label="like")
    switched_back = feed(selector, [right_front, equal], start=23, count=8)
    assert switched_back[0][0] is equal
    assert switched_back[0][1]["switch_pending"]
    assert switched_back[-1][0] is right_front
    assert switched_back[-1][1]["selected_track_id"] != selection["selected_track_id"]


def test_equal_distance_owner_survives_single_frame_closeness_spikes():
    selector = AutomaticHandSelector(RuntimeConfig())
    owner = observation(scale=.6)
    peer = observation(x=.75, scale=.3, hand="Left")
    owner_id = feed(selector, [owner, peer])[-1][1]["selected_track_id"]
    level_owner = observation(scale=.5)
    level_peer = observation(x=.75, scale=.5, hand="Left")
    assert feed(selector, [level_owner, level_peer], start=10, count=3)[-1][0] is level_owner

    apparent_challenger = observation(x=.75, scale=.59, hand="Left")
    for frame in range(13, 29):
        peer_row = apparent_challenger if frame % 2 else level_peer
        chosen, selection, previews = selector.update(
            [level_owner, peer_row], SHAPE, frame * .1)
        assert chosen is level_owner
        assert selection["selected_track_id"] == owner_id
        assert not selection["ownership_reset"]
        assert previews[1]["gesture"] == "no_gesture"
        if frame % 2:
            assert selection["switch_pending"]
            assert not previews[0]["eligible"]
        else:
            assert not selection["switch_pending"]


def test_borderline_world_scale_keeps_clear_owner():
    selector = AutomaticHandSelector(RuntimeConfig())
    owner = observation(scale=.6)
    peer = observation(x=.75, scale=.3, hand="Left")
    owner_id = feed(selector, [owner, peer])[-1][1]["selected_track_id"]
    level_owner = projected_world(observation(scale=.5), 1200.)
    slightly_larger_peer = projected_world(
        observation(x=.75, scale=.5, hand="Left"), 1280.)
    chosen, selection, _ = selector.update(
        [slightly_larger_peer, level_owner], SHAPE, 1.)
    assert chosen is level_owner
    assert selection["selected_track_id"] == owner_id
    assert selection["distance_method"] == "projected world landmarks"
    assert selection["distance_scale_ratio"] < selector.policy.closer_projected_world_ratio
    assert not selection["ownership_reset"]


def test_close_together_hands_still_select_the_foreground_when_tracks_are_distinct():
    selector = AutomaticHandSelector(RuntimeConfig())
    right = observation(x=.42, scale=.60, label="like")
    left = observation(x=.46, scale=.42, hand="Left", label="peace")
    chosen, selection, _ = feed(selector, [right, left])[-1]
    assert chosen is right
    assert selection["reason"] in {"closer_hand_selected", "active_hand_locked"}


def test_reordering_detections_preserves_selected_identity():
    selector = AutomaticHandSelector(RuntimeConfig())
    a, b = observation(scale=.6), observation(x=.75, scale=.3, hand="Left")
    first = feed(selector, [a, b])[-1]
    later = feed(selector, [b, a], start=10, count=2)[-1]
    assert later[0] is a
    assert later[1]["selected_track_id"] == first[1]["selected_track_id"]


def test_overlap_suppresses_actions_without_transferring_identity():
    selector = AutomaticHandSelector(RuntimeConfig())
    first = feed(selector, [observation()])[-1]
    rows = [observation(x=.4), observation(x=.41, hand="Left")]
    overlapped = selector.update(rows, SHAPE, 1.0)
    assert overlapped[0] is None
    assert overlapped[1]["reason"] == "ambiguous_tracks"
    assert overlapped[1]["active_track_id"] == first[1]["selected_track_id"]
    assert all(not hand["selected"] for hand in overlapped[2])
    returned = selector.update([observation()], SHAPE, 1.1)
    assert returned[1]["selected_track_id"] == first[1]["selected_track_id"]
    assert returned[2][0]["eligible"] is False


def test_missing_active_hand_does_not_immediately_select_other_hand():
    selector = AutomaticHandSelector(RuntimeConfig())
    near, far = observation(scale=.6), observation(x=.75, scale=.3, hand="Left")
    owner = feed(selector, [near, far])[-1][1]["selected_track_id"]
    results = feed(selector, [far], start=10, count=3)
    assert all(result[0] is None for result in results)
    assert results[-1][1]["reason"] == "active_hand_missing"
    later = feed(selector, [far], start=13, count=10)
    assert any(selection["reason"] == "active_hand_expired" and selection["ownership_reset"]
               for _, selection, _ in later)
    assert later[-1][0] is far
    assert later[-1][1]["selected_track_id"] != owner


def test_closer_challenger_switches_the_first_successful_owner():
    selector = AutomaticHandSelector(RuntimeConfig())
    active = observation(scale=.35)
    original = feed(selector, [active])[-1]
    challenger = observation(x=.75, scale=.65, hand="Left")
    results = feed(selector, [active, challenger], start=10, count=9)
    assert results[0][0] is active
    assert results[0][1]["switch_pending"]
    assert not results[0][1]["ownership_reset"]
    assert results[-1][0] is challenger
    assert results[-1][1]["selected_track_id"] != original[1]["selected_track_id"]


@pytest.mark.parametrize("label,scale", [("like", .10), ("dorsal", .16)])
def test_tiny_or_distant_dorsal_hands_never_qualify(label, scale):
    results = feed(AutomaticHandSelector(RuntimeConfig()), [observation(label=label, scale=scale)])
    assert all(result[0] is None for result in results)
    assert results[-1][1]["reason"] == "hand_too_small"


def test_foreshortened_palm_ratios_are_not_treated_as_reliable_depth():
    selector = AutomaticHandSelector(RuntimeConfig())
    near, far = observation(scale=.6), observation(x=.75, scale=.3, hand="Left")
    near["metrics"]["palm_segments_px"] = [70., 35., 75., 20.]
    far["metrics"]["palm_segments_px"] = [35., 35., 35., 35.]
    result = feed(selector, [near, far])[-1]
    assert result[0] is None
    assert result[1]["reason"] == "hand_distance_uncertain"


def test_sessions_and_reset_do_not_share_selection_history():
    a, b = RuntimeSession(RuntimeConfig()), RuntimeSession(RuntimeConfig())
    feed(a.hand_selector, [observation()])
    assert a.hand_selector.active_id is not None
    assert b.hand_selector.active_id is None
    a.reset()
    assert not a.hand_selector.tracks
    assert a.selected_track_id is None


def test_resolution_change_gets_new_identity_and_fresh_confirmation():
    selector = AutomaticHandSelector(RuntimeConfig())
    row = observation()
    old = feed(selector, [row])[-1]
    changed = selector.update([row], (720, 960, 3), 1.0)
    assert changed[1]["selected_track_id"] != old[1]["selected_track_id"]
    assert changed[2][0]["eligible"] is False


def test_high_frame_rate_cannot_skip_the_hand_choice_hold():
    selector = AutomaticHandSelector(RuntimeConfig())
    rows = [observation(scale=.6), observation(x=.75, scale=.3, hand="Left")]
    for frame in range(20):
        chosen, _, _ = selector.update(rows, SHAPE, frame / 100)
        assert chosen is None


def test_closer_size_must_persist_after_a_single_frame_spike():
    selector = AutomaticHandSelector(RuntimeConfig())
    rows = [observation(), observation(x=.75, hand="Left")]
    feed(selector, rows)
    spike = observation(scale=.68)
    chosen, _, _ = selector.update([spike, rows[1]], SHAPE, 1.)
    assert chosen is None
    assert all(item[0] is None for item in feed(selector, rows, start=11, count=8))


def test_detector_keeps_full_frame_pass_and_deduplicates_recovery(monkeypatch):
    detector = simulated_detector()
    near, far = observation(scale=.6), observation(x=.75, scale=.3)
    full = (0, 0, 640, 480, 0, 0)
    seen = []

    def candidates(rgb, view, *, video):
        seen.append((view, video))
        rows = [near] if view == full else [near, far]
        return [(r["area"], r["points"], None, None) for r in rows]

    monkeypatch.setattr(detector, "_run_candidates", candidates)
    state = {"shape": (480, 640), "frames": 2}
    rows = detector.detect_candidates(np.zeros(SHAPE, np.uint8), state=state, candidate_score=lambda row: 1)
    assert len(rows) == 2
    detector.detect_candidates(np.zeros(SHAPE, np.uint8), state=state, candidate_score=lambda row: 1)
    assert seen[0][0] == full and seen[2][0] == full


def test_detector_recovers_recently_tracked_second_hand_on_first_dropout(monkeypatch):
    detector = simulated_detector()
    owner = observation(scale=.5)
    peer = observation(x=.75, scale=.5, hand="Left")
    full = (0, 0, 640, 480, 0, 0)
    seen = []

    def candidates(rgb, view, *, video):
        seen.append(view)
        rows = [peer] if view == full else [owner]
        return [(r["area"], r["points"], None, None) for r in rows]

    monkeypatch.setattr(detector, "_run_candidates", candidates)
    state = {"shape": (480, 640), "frames": 0, "expected_hands": 2}
    rows = detector.detect_candidates(
        np.zeros(SHAPE, np.uint8), state=state, candidate_score=lambda row: 1)
    assert len(seen) == 2
    assert seen[0] == full
    assert len(rows) == 2


def test_runtime_switches_to_closer_challenger(monkeypatch):
    import cv2
    import backend.model_runtime as runtime_module

    config = RuntimeConfig()
    engine = InferenceEngine.__new__(InferenceEngine)
    import threading
    engine.config, engine._lock = config, threading.Lock()
    manager = SimpleNamespace(ready=True, models={"ONNX": object()}, resolve_name=lambda name: "ONNX")
    probabilities = observation()["probabilities"]
    quality = {"known_gesture_mass": 1.0}
    manager.predict_detailed = lambda *args: (probabilities.copy(), "ONNX", .01, quality)
    manager.predict_many = lambda *args: {"ONNX": (probabilities.copy(), .01, quality)}
    engine.model_manager = manager
    def resolve(p, points, **kwargs):
        adjusted = observation(label="left" if points[0, 0] > .5 else "like")["probabilities"]
        return adjusted, {"pose_validation": {"valid": True}}
    engine.resolver = SimpleNamespace(resolve=resolve)
    engine.detector = simulated_detector()
    engine.detector._last_diagnostics = {}
    rows = [observation()]
    expected_counts = []

    def detect(rgb, *, state, candidate_score):
        expected_counts.append(state.get("expected_hands"))
        for row in rows:
            candidate_score(row)
        engine.detector._last_diagnostics = {"candidate_count": len(rows)}
        return rows

    monkeypatch.setattr(engine.detector, "detect_candidates", detect)
    clock = [0.0]
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: clock[0])
    encoded = cv2.imencode(".jpg", np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    outputs = []
    for i in range(6):
        clock[0] = 1 + i * .1
        outputs.append(engine.process_frame(encoded, session))
    assert outputs[-1]["hand_selection"]["selected_track_id"] == 1
    assert any(r["runtime_action"] != "Wait / No Action" for r in outputs)
    session.last_action_gesture = None
    session.last_action_at = -10.
    rows.append(observation(x=.75, scale=.8, hand="Left"))
    outputs = []
    for i in range(6, 20):
        clock[0] = 1 + i * .1
        output = engine.process_frame(encoded, session)
        outputs.append(output)
        assert len(output["hands"]) == 2
    assert outputs[0]["runtime_prediction"] == "like"
    assert outputs[0]["hand_selection"]["switch_pending"]
    assert outputs[0]["diagnostics"]["ONNX"]["execute"]
    assert outputs[0]["runtime_action"] == "Wait / No Action"
    assert outputs[-1]["hand_selection"]["selected_track_id"] != 1
    assert outputs[-1]["runtime_prediction"] == "left"
    assert outputs[-1]["hands"][0]["gesture"] == "no_gesture"
    assert expected_counts[-1] == 2


def test_runtime_classifies_current_finger_pose_after_landmark_filter_accepts_it(monkeypatch):
    import cv2
    import threading
    import backend.model_runtime as runtime_module

    config = RuntimeConfig()
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine._lock = config, threading.Lock()
    like = observation(label="like")["probabilities"]
    engine.model_manager = SimpleNamespace(
        ready=True, models={"ONNX": object()}, resolve_name=lambda name: "ONNX",
        predict_detailed=lambda *args: (like.copy(), "ONNX", .01, {"known_gesture_mass": 1.}),
        predict_many=lambda *args: {"ONNX": (like.copy(), .01, {"known_gesture_mass": 1.})},
    )

    def resolve(probabilities, points, **kwargs):
        label = "up" if points[8, 0] - points[5, 0] > .07 else "like"
        return observation(label=label)["probabilities"], {"pose_validation": {"valid": True}}

    engine.resolver = SimpleNamespace(resolve=resolve)
    engine.detector = simulated_detector()
    row = observation(scale=.4)
    rows = [row]

    def detect(rgb, *, state, candidate_score):
        candidate_score(rows[0])
        engine.detector._last_diagnostics = {"candidate_count": 1}
        return rows

    monkeypatch.setattr(engine.detector, "detect_candidates", detect)
    clock = [0.]
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: clock[0])
    frame = cv2.imencode(".jpg", np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    for index in range(10):
        clock[0] = 1 + index * .1
        output = engine.process_frame(frame, session)
    assert output["runtime_prediction"] == "like"

    changed = observation(scale=.4)
    changed["points"][8, 0] += .08
    changed["metrics"] = hand_pixel_metrics(changed["points"], SHAPE)
    rows[0] = changed
    clock[0] = 2.
    output = engine.process_frame(frame, session)
    assert output["runtime_prediction"] == "up"
    assert output["landmarks"][8][0] == pytest.approx(changed["points"][8, 0], abs=1e-6)
