"""Apply strict Dorsal context policy at candidate and final runtime stages."""
from types import SimpleNamespace
import threading

import cv2
import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.body_guard import (RESTING_HAND_REASON, DORSAL_BODY_REQUIRED_REASON,
                                body_hand_context, dorsal_body_rejection)
from backend.hand_detection import hand_pixel_metrics
from backend.model_runtime import InferenceEngine, RuntimeSession
from backend.tests.test_camera_stability import representative_hand
from backend.tests.test_dorsal_posture import arm_case
from backend.tests.test_hand_detection import simulated_detector


SHAPE = (750, 960, 3)


def standing_case(*, hand_wrist_y=500, pose_wrist_y=None, elbow_visible=True):
    pose_wrist_y = hand_wrist_y if pose_wrist_y is None else pose_wrist_y
    hand = representative_hand().astype(float)
    hand[:, 0] = (hand[:, 0] - .5) * 200 + 470
    hand[:, 1] = (.9 - hand[:, 1]) * 220 + hand_wrist_y
    pose = np.zeros((33, 4), dtype=float)
    pose[11] = [440, 180, 1, 1]
    # Keep a measurable forearm in raised cases too; a wrist 30 pixels from
    # the elbow is insufficient evidence for the production posture check.
    pose[13] = [400 if pose_wrist_y <= 330 else 450, 330,
                int(elbow_visible), int(elbow_visible)]
    pose[15] = [470, pose_wrist_y, 1, 1]
    pose[23] = [470, 450, 1, 1]
    pose[25] = [475, 710, 1, 1]
    pose[:, :2] /= [SHAPE[1], SHAPE[0]]
    return hand / [SHAPE[1], SHAPE[0]], pose


def test_standing_dorsal_below_waist_is_negative_but_other_class_is_eligible():
    hand, pose = standing_case()
    context = body_hand_context(hand, SHAPE, [pose])
    assert context["below_waist"] and context["relaxed"]
    assert dorsal_body_rejection(context, "dorsal") == RESTING_HAND_REASON
    assert dorsal_body_rejection(context, "open_palm") is None
    assert dorsal_body_rejection(context, "down") is None


def test_dorsal_below_waist_with_unclear_elbow_is_not_assumed_relaxed():
    hand, pose = standing_case(elbow_visible=False)
    context = body_hand_context(hand, SHAPE, [pose])
    assert context["below_waist"] and not context["relaxed"]
    assert dorsal_body_rejection(context, "dorsal") == DORSAL_BODY_REQUIRED_REASON


def test_raised_dorsal_and_unmatched_body_are_not_blocked():
    # Fingertips may approach waist height while the wrist/forearm is raised.
    hand, pose = standing_case(hand_wrist_y=300)
    context = body_hand_context(hand, SHAPE, [pose])
    assert context and not context["below_waist"]
    assert dorsal_body_rejection(context, "dorsal") is None
    assert body_hand_context(hand, SHAPE, []) is None
    other_hand, other_pose = standing_case(pose_wrist_y=300)
    assert body_hand_context(other_hand, SHAPE, [other_pose]) is None


def test_detector_scores_hand_and_vetoes_only_dorsal_below_waist(monkeypatch):
    import backend.hand_detection as module
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.)
    hand, pose = standing_case()
    detector = simulated_detector()
    detector._body_guard = SimpleNamespace(
        detector=True, detect=lambda rgb, *, state: ([pose], dict(ready=True, body_count=1)))
    detector._run_candidates = lambda rgb, view, *, video: [(.1, hand, None, None)]
    state = {}
    frame = np.full(SHAPE, 140, np.uint8)
    scores = []

    def candidate(label):
        def score(row):
            scores.append(label)
            row["resolved_label"] = label
            return 1.
        return score

    dorsal = detector.detect_candidates(frame, state=state, candidate_score=candidate("dorsal"))
    assert dorsal[0]["rejection_reason"] == RESTING_HAND_REASON
    assert dorsal[0]["recovery_score"] == -1.
    palm = detector.detect_candidates(frame, state=state, candidate_score=candidate("open_palm"))
    assert palm[0]["rejection_reason"] is None
    assert scores == ["dorsal", "open_palm"]


@pytest.mark.parametrize("final_label", ["dorsal", "fist"])
@pytest.mark.parametrize("posture", ["standing", "seated", "unknown", "presented"])
def test_final_smoothed_body_rule_applies_only_to_dorsal(monkeypatch, final_label, posture):
    import backend.model_runtime as runtime_module
    if posture == "seated":
        hand, pose = arm_case(shoulder=(815, 230), elbow=(865, 315),
                              wrist=(750, 405), hip=(820, 650))
        pose[24, 2:] = 0
    elif posture == "presented":
        hand, pose = standing_case(hand_wrist_y=300)
    else:
        hand, pose = standing_case()
    context = None if posture == "unknown" else body_hand_context(hand, SHAPE, [pose])
    config = RuntimeConfig()
    palm = np.eye(len(config.class_names))[config.class_to_idx["open_palm"]]
    restricted = np.eye(len(config.class_names))[config.class_to_idx[final_label]]
    row = dict(points=hand, world_landmarks=None, metrics=hand_pixel_metrics(hand, SHAPE),
               area=.1, view=(0, 0, 960, 750, 0, 0), handedness="Left",
               handedness_confidence=.99, rejection_reason=None, body_context=context,
               probabilities=palm, gate_args=dict(known_gesture_mass=1.))
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine._lock = config, threading.Lock()
    engine.detector = simulated_detector()
    engine.detector._last_diagnostics = {}

    def detect(rgb, *, state, candidate_score):
        engine.detector._last_diagnostics = dict(candidate_count=1)
        return [row]

    monkeypatch.setattr(engine.detector, "detect_candidates", detect)
    quality = dict(known_gesture_mass=1.)
    engine.model_manager = SimpleNamespace(
        ready=True, models={"ONNX": object()}, resolve_name=lambda name: "ONNX",
        predict_many=lambda *args: {"ONNX": (palm.copy(), .01, quality)})
    engine.resolver = SimpleNamespace(resolve=lambda p, points, **kwargs: (
        restricted.copy(), {"pose_validation": {"valid": True}}))
    encoded = cv2.imencode(".jpg", np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    for frame in range(6):
        output = engine.process_frame(encoded, session, now=1 + frame * .1)
        if final_label == "dorsal" and posture != "presented":
            assert output["runtime_prediction"] == "no_gesture"
            assert output["runtime_action"] == "Wait / No Action"
            expected = DORSAL_BODY_REQUIRED_REASON if posture == "unknown" else RESTING_HAND_REASON
            assert output["resolvers"]["pose_validation"]["reason"] == expected
        else:
            assert output["runtime_prediction"] == final_label
            assert output["resolvers"]["pose_validation"]["valid"]
    assert session.hand_selector.detector_state["frame_time"] == 1.5


@pytest.mark.parametrize("label", RuntimeConfig().class_names)
def test_missing_body_estimate_pauses_only_dorsal(monkeypatch, label):
    import backend.hand_detection as module
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.)
    hand, _ = standing_case()
    detector = simulated_detector()
    detector._body_guard = SimpleNamespace(detector=True, detect=lambda rgb, *, state: (
        [], dict(ready=False, error="posture unavailable", body_count=0)))
    detector._run_candidates = lambda rgb, view, *, video: [(.1, hand, None, None)]
    def score(row):
        row["resolved_label"] = label
        return 1.
    rows = detector.detect_candidates(np.full(SHAPE, 145, np.uint8), state={}, candidate_score=score)
    assert rows[0]["rejection_reason"] == (DORSAL_BODY_REQUIRED_REASON if label == "dorsal" else None)
    assert not detector.diagnostics()["body_guard"]["ready"]


def test_recovery_crop_cannot_replace_dorsal_missing_body_veto(monkeypatch):
    import backend.hand_detection as module
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.)
    hand, _ = standing_case()
    detector = simulated_detector()
    detector._run_candidates = lambda rgb, view, *, video: [(.1, hand, None, None)]
    def score(row):
        row["resolved_label"] = "fist" if row["recovered"] else "dorsal"
        return 1.
    rows = detector.detect_candidates(np.full(SHAPE, 145, np.uint8),
                                     state={"shape": SHAPE[:2], "pose_recovery_requested": True},
                                     candidate_score=score)
    assert detector.diagnostics()["recovery_attempted"]
    assert len(rows) == 1 and rows[0]["resolved_label"] == "dorsal"
    assert rows[0]["rejection_reason"] == DORSAL_BODY_REQUIRED_REASON


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -1.])
def test_replay_clock_rejects_invalid_timestamp(timestamp):
    engine = InferenceEngine.__new__(InferenceEngine)
    with pytest.raises(ValueError, match="finite nonnegative"):
        engine.process_frame(b"", RuntimeSession(RuntimeConfig()), now=timestamp)
