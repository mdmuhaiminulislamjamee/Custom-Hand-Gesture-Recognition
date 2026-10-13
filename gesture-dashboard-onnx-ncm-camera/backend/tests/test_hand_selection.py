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


def test_two_hands_choose_significantly_larger_palm_only_after_confirmation():
    selector = AutomaticHandSelector(RuntimeConfig())
    near, far = observation(scale=.6), observation(x=.75, scale=.3, hand="Left")
    results = feed(selector, [far, near])
    assert all(result[0] is None for result in results[:5])
    assert results[-1][0] is near
    assert results[-1][1]["selected_track_id"] == 2


def test_larger_idle_hand_does_not_beat_smaller_gesture_hand():
    selector = AutomaticHandSelector(RuntimeConfig())
    idle = observation(x=.75, scale=.6, hand="Left", mass=.01)
    gesture = observation(scale=.35)
    assert feed(selector, [idle, gesture])[-1][0] is gesture


def test_equal_depth_valid_gestures_remain_ambiguous():
    selector = AutomaticHandSelector(RuntimeConfig())
    results = feed(selector, [observation(), observation(x=.75, hand="Left")])
    assert all(result[0] is None for result in results)
    assert results[-1][1]["reason"] == "ambiguous_hands"


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
    assert overlapped[1]["reason"] == "hands_overlap"
    assert all(not hand["selected"] for hand in overlapped[2])
    returned = selector.update([observation()], SHAPE, 1.1)
    assert returned[1]["selected_track_id"] == first[1]["selected_track_id"]
    assert returned[2][0]["eligible"] is False


def test_missing_active_hand_does_not_immediately_select_other_hand():
    selector = AutomaticHandSelector(RuntimeConfig())
    near, far = observation(scale=.6), observation(x=.75, scale=.3, hand="Left")
    feed(selector, [near, far])
    results = feed(selector, [far], start=10, count=3)
    assert all(result[0] is None for result in results)
    assert results[-1][1]["reason"] == "active_hand_missing"
    assert feed(selector, [far], start=13, count=7)[-1][0] is far


def test_closer_challenger_requires_sustained_margin_before_switch():
    selector = AutomaticHandSelector(RuntimeConfig())
    active = observation(scale=.35)
    original = feed(selector, [active])[-1]
    challenger = observation(x=.75, scale=.65, hand="Left")
    results = feed(selector, [active, challenger], start=10, count=9)
    assert results[0][1]["selected_track_id"] == original[1]["selected_track_id"]
    assert any(result[1]["reason"] == "confirming_closer_hand" for result in results)
    assert results[-1][0] is challenger


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
    assert result[1]["reason"] == "ambiguous_hands"


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


def test_runtime_resets_confirmation_when_selection_changes_and_reports_both_hands(monkeypatch):
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
    engine.resolver = SimpleNamespace(resolve=lambda p, points, **kwargs: (p, {"pose_validation": {"valid": True}}))
    engine.detector = simulated_detector()
    engine.detector._last_diagnostics = {}
    rows = [observation()]

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
    outputs = []
    for i in range(6):
        clock[0] = 1 + i * .1
        outputs.append(engine.process_frame(encoded, session))
    assert outputs[-1]["hand_selection"]["selected_track_id"] == 1
    assert any(r["runtime_action"] != "Wait / No Action" for r in outputs)
    rows.append(observation(x=.75, scale=.8, hand="Left"))
    for i in range(6, 20):
        clock[0] = 1 + i * .1
        output = engine.process_frame(encoded, session)
        assert len(output["hands"]) == 2
        if output["hand_selection"]["selected_track_id"] == 2:
            assert output["stable_frames"] == 1
            assert output["runtime_action"] == "Wait / No Action"
            assert output["quality"]["detector"]["reacquired"] is True
            break
    else:
        pytest.fail("The sustained closer hand was never selected")
