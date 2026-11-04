"""Candidate-policy regressions; synthetic joints do not replay image detection."""
from types import SimpleNamespace
import threading

import cv2
import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.face_guard import FaceGuard, overlaps_face_as_small_hand
from backend.hand_validation import invalid_hand_geometry
from backend.model_runtime import InferenceEngine, RuntimeSession
from backend.tests.test_camera_stability import representative_hand
from backend.tests.test_hand_detection import simulated_detector


SHAPE = (480, 640, 3)
FACE = (350, 80, 140, 210)


def ear_points():
    return (representative_hand() - .5) * .20 + [.56, .31]


def collapsed_points():
    points = (representative_hand() - .5) * .5 + [.3, .5]
    for start in (5, 9, 13, 17):
        points[start:start + 4] = points[start]
    return points


def guarded_detector(monkeypatch, points, boxes=(FACE,)):
    import backend.hand_detection as module
    # Permit just the full-frame pass for deterministic synthetic observations.
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.)
    detector = simulated_detector()
    detector._face_guard = SimpleNamespace(
        detector=True, error=None, detect=lambda rgb, **kwargs: list(boxes))
    detector._run_candidates = lambda rgb, view, *, video: [(.1, point, None, None) for point in points]
    return detector


@pytest.mark.parametrize("points,reason", [
    (ear_points(), "face_detected_as_hand"),
    (collapsed_points(), "invalid_hand_geometry"),
    ((representative_hand() - .5) * .04 + [.1, .5], "insufficient_hand_pixels"),
])
def test_false_candidates_never_reach_classifier_or_recovery_history(monkeypatch, points, reason):
    detector = guarded_detector(monkeypatch, [points])
    state = {}
    def score(row):
        pytest.fail("Rejected objects must not reach gesture classification")
    for index in range(6):
        state["frame_time"] = index * .1
        rows = detector.detect_candidates(np.full(SHAPE, 140, np.uint8), state=state, candidate_score=score)
        assert rows == []
        assert state["recent_hands"] == []
        assert detector.diagnostics()["rejection_reason"] == reason


def test_real_hand_and_ear_produce_only_one_hand_overlay_and_valid_command(monkeypatch):
    import backend.model_runtime as module
    config = RuntimeConfig()
    real = (representative_hand() - .5) * .55 + [.23, .5]
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine._lock = config, threading.Lock()
    probabilities = np.eye(len(config.class_names))[config.class_to_idx["like"]]
    quality = {"known_gesture_mass": 1.}
    engine.model_manager = SimpleNamespace(
        ready=True, models={"ONNX": object()}, resolve_name=lambda name: "ONNX",
        predict_detailed=lambda *args: (probabilities.copy(), "ONNX", .01, quality),
        predict_many=lambda *args: {"ONNX": (probabilities.copy(), .01, quality)})
    engine.resolver = SimpleNamespace(resolve=lambda p, points, **kwargs: (
        p, {"pose_validation": {"valid": True}}))
    engine.detector = guarded_detector(monkeypatch, [ear_points(), real])
    clock = [1.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    jpeg = cv2.imencode(".jpg", np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    for index in range(16):
        clock[0] = 1 + index * .1
        output = engine.process_frame(jpeg, session)
        assert len(output["hands"]) == 1
        assert output["quality"]["detector"]["rejected_candidates"][0]["reason"] == "face_detected_as_hand"
    assert output["runtime_prediction"] == "like"
    assert len(session.hand_selector.tracks) == 1
    # Losing the real hand must immediately clear its visible skeleton/action.
    engine.detector._run_candidates = lambda *args, **kwargs: [(.1, ear_points(), None, None)]
    clock[0] += .1
    output = engine.process_frame(jpeg, session)
    assert output["status"] == "no_hand"
    assert output["hands"] == [] and output.get("landmarks") is None
    assert output["runtime_prediction"] == "no_gesture"
    assert output["runtime_action"] == "Wait / No Action"


def test_large_foreground_hand_and_fingertip_overlap_are_allowed():
    assert not overlaps_face_as_small_hand(representative_hand(), SHAPE, [FACE])
    hand = (representative_hand() - .5) * .4 + [.38, .3]
    hand[8] = [.6, .3]  # Just one fingertip crosses the face.
    assert not overlaps_face_as_small_hand(hand, SHAPE, [FACE])
    assert overlaps_face_as_small_hand(ear_points(), SHAPE, [FACE])
    assert not overlaps_face_as_small_hand(ear_points(), SHAPE, [(350, 80, 15, 20)])


def test_valid_world_joints_rescue_edge_on_projection():
    world = np.column_stack((representative_hand(), np.zeros(21)))
    assert invalid_hand_geometry(collapsed_points(), SHAPE)
    assert not invalid_hand_geometry(collapsed_points(), SHAPE, world)
    assert not invalid_hand_geometry(representative_hand(), SHAPE)


def test_face_failure_pauses_candidates_without_crashing(monkeypatch):
    detector = guarded_detector(monkeypatch, [representative_hand()], boxes=())
    detector._face_guard.error = "face inference unavailable"
    rows = detector.detect_candidates(np.full(SHAPE, 140, np.uint8), state={}, candidate_score=lambda row: 1.)
    assert rows == []
    assert detector.diagnostics()["rejection_reason"] == "face_context_unavailable"
    assert detector.diagnostics()["face_guard_error"] == "face inference unavailable"


def test_face_refresh_uses_recorded_time_and_session_cache(monkeypatch):
    guard = FaceGuard.__new__(FaceGuard)
    calls = []
    result = SimpleNamespace(detections=[SimpleNamespace(bounding_box=SimpleNamespace(
        origin_x=100, origin_y=40, width=70, height=90))])
    guard.mp = SimpleNamespace(Image=lambda **kwargs: None, ImageFormat=SimpleNamespace(SRGB=1))
    guard.detector = SimpleNamespace(detect=lambda image: (calls.append(True) or result))
    frame = np.zeros((240, 320, 3), np.uint8)
    state = {"frame_time": 1.}
    assert guard.detect(frame, state=state)
    result.detections = []
    state["frame_time"] = 1.1
    assert guard.detect(frame, state=state)
    assert len(calls) == 1
    # Another stream gets no cached face, even at the same timestamp.
    assert guard.detect(frame, state={"frame_time": 1.1}) == []
    state["frame_time"] = 1.21
    assert guard.detect(frame, state=state)
    state["frame_time"] = 1.51
    assert guard.detect(frame, state=state) == []
    # Rewinding or changing resolution must discard the old coordinates.
    state["frame_time"] = .5
    assert guard.detect(frame, state=state) == []
    result.detections = [SimpleNamespace(bounding_box=SimpleNamespace(
        origin_x=100, origin_y=40, width=70, height=90))]
    state["frame_time"] = .8
    assert guard.detect(frame, state=state)
    result.detections = []
    assert guard.detect(np.zeros((480, 640, 3), np.uint8), state=state) == []
