"""Two-hand identity, handedness consensus, distance jitter and fresh overlays."""
from types import SimpleNamespace
import threading

import cv2
import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.hand_selection import AutomaticHandSelector
from backend.model_runtime import InferenceEngine, RuntimeSession
from backend.tests.test_hand_detection import simulated_detector
from backend.tests.test_hand_selection import SHAPE, feed, observation, projected_world


def test_one_missing_frame_preserves_peer_hold_without_exposing_a_missing_hand():
    selector = AutomaticHandSelector(RuntimeConfig())
    owner = observation(scale=.6)
    peer = observation(x=.75, scale=.3, hand='Left')
    _, _, previews = feed(selector, [owner, peer])[-1]
    peer_id = previews[1]['track_id']
    history = len(selector.tracks[peer_id].gate._history)
    _, _, missing = selector.update([owner], SHAPE, 1.)
    assert len(missing) == 1
    assert len(selector.tracks[peer_id].gate._history) == history
    selector.update([owner, peer], SHAPE, 1.1)
    assert len(selector.tracks[peer_id].gate._history) == history + 1
    selector.update([owner], SHAPE, 1.2)
    selector.update([owner], SHAPE, 1.3)
    assert not selector.tracks[peer_id].gate._history


def test_live_owner_resumes_after_one_miss_but_never_acts_without_fresh_landmarks():
    config = RuntimeConfig()
    probability = observation()['probabilities']
    quality = dict(known_gesture_mass=1.)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine._lock = config, threading.Lock()
    engine.model_manager = SimpleNamespace(
        ready=True, models={'ONNX': object()}, resolve_name=lambda name: 'ONNX',
        predict_detailed=lambda *args: (probability.copy(), 'ONNX', .01, quality),
        predict_many=lambda *args: {'ONNX': (probability.copy(), .01, quality)})
    engine.resolver = SimpleNamespace(resolve=lambda p, points, **kwargs: (
        p, {'pose_validation': dict(valid=True, geometry_supported=True)}))
    engine.detector = simulated_detector()
    engine.detector._last_diagnostics = {}
    rows = [observation()]
    engine.detector.detect_candidates = lambda *args, **kwargs: rows
    image = cv2.imencode('.jpg', np.full(SHAPE, 145, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    for frame in range(10):
        result = engine.process_frame(image, session, now=1 + frame * .1)
    owner = result['hand_selection']['selected_track_id']
    before = result['stable_frames']
    rows = []
    missing = engine.process_frame(image, session, now=2.)
    assert missing['hands'] == [] and missing['runtime_prediction'] == 'no_gesture'
    assert missing['runtime_action'] == 'Wait / No Action'
    rows = [observation()]
    returned = engine.process_frame(image, session, now=2.1)
    assert returned['hand_selection']['selected_track_id'] == owner
    assert returned['runtime_prediction'] == 'like'
    assert returned['stable_frames'] > before
    assert returned['runtime_action'] == 'Wait / No Action'  # No duplicate edge.


@pytest.mark.parametrize("owner_side", ["Right", "Left"])
@pytest.mark.parametrize("noise", ["duplicate_labels", "swapped_labels", "uncertain"])
@pytest.mark.parametrize("interval", [.1, .4])
def test_clear_two_hand_tracks_keep_identity_and_side_through_label_noise(owner_side, noise, interval):
    selector = AutomaticHandSelector(RuntimeConfig())
    peer_side = "Left" if owner_side == "Right" else "Right"
    original = [observation(scale=.6, hand=owner_side),
                observation(x=.75, scale=.3, hand=peer_side)]
    _, selection, previews = feed(selector, original)[-1]
    owner = selection["selected_track_id"]
    ids = {hand["handedness"]: hand["track_id"] for hand in previews}
    for frame in range(10, 70):
        near = observation(scale=.6, hand=owner_side)
        far = observation(x=.75, scale=.3, hand=peer_side)
        if frame % 3:
            near["handedness"] = peer_side
            if noise == "swapped_labels":
                far["handedness"] = owner_side
            elif noise == "uncertain":
                near["handedness_confidence"] = .55
        rows = [near, far] if frame % 2 else [far, near]
        chosen, result, previews = selector.update(rows, SHAPE, .9 + (frame - 9) * interval)
        assert chosen is near
        assert result["selected_track_id"] == owner
        assert not result["ownership_reset"]
        assert len(selector.tracks) == 2
        assert {hand["handedness"]: hand["track_id"] for hand in previews} == ids
        assert all(hand["handedness_stable"] for hand in previews)


def test_initial_wrong_side_can_correct_before_consensus_locks():
    selector = AutomaticHandSelector(RuntimeConfig())
    selector.update([observation(scale=.6, hand="Left"), observation(x=.75, scale=.3, hand="Left")], SHAPE, 0.)
    for frame in range(1, 8):
        _, _, previews = selector.update([
            observation(scale=.6), observation(x=.75, scale=.3, hand="Left")], SHAPE, frame * .1)
    assert [(hand["track_id"], hand["handedness"]) for hand in previews] == [(1, "Right"), (2, "Left")]
    assert all(hand["handedness_stable"] for hand in previews)


def test_ambiguous_overlap_still_pauses_and_does_not_rewrite_hand_labels():
    selector = AutomaticHandSelector(RuntimeConfig())
    original = [observation(scale=.6), observation(x=.75, scale=.3, hand="Left")]
    _, selected, _ = feed(selector, original)[-1]
    owner = selected["selected_track_id"]
    sides = {key: track.handedness for key, track in selector.tracks.items()}
    for frame in range(10, 13):
        chosen, selection, previews = selector.update([
            observation(x=.5, scale=.4, hand="Left"),
            observation(x=.51, scale=.4, hand="Left")], SHAPE, frame * .1)
        assert chosen is None and selection["active_track_id"] == owner
        assert all(not hand["eligible"] for hand in previews)
        assert {key: track.handedness for key, track in selector.tracks.items()} == sides
    restored = feed(selector, original, start=13)[-1]
    assert restored[1]["selected_track_id"] == owner


def test_noisy_distance_bursts_cannot_switch_owner_but_sustained_lead_can():
    selector = AutomaticHandSelector(RuntimeConfig())
    _, selected, _ = feed(selector, [observation(scale=.6), observation(x=.75, scale=.3, hand="Left")])[-1]
    owner = selected["selected_track_id"]
    def pair(peer_scale):
        return [projected_world(observation(scale=.5), 1200.),
                projected_world(observation(x=.75, scale=.5, hand="Left"), peer_scale)]
    feed(selector, pair(1200.), start=10, count=15)
    for frame in range(25, 89):
        # Four just-over-threshold frames used to complete a switch hold.
        # Their filtered evidence must agree before ownership can transfer.
        scale = 1344. if (frame - 25) % 8 < 4 else 1200.
        rows = pair(scale)
        chosen, selection, _ = selector.update(rows, SHAPE, frame * .1)
        assert chosen is rows[0]
        assert selection["selected_track_id"] == owner
        assert not selection["ownership_reset"]
    results = feed(selector, pair(1500.), start=89, count=12)
    assert results[-1][1]["selected_track_id"] != owner
    assert results[-1][0]["handedness"] == "Left"


def test_dorsal_foreground_clearance_has_hysteresis_after_identity_assignment():
    selector = AutomaticHandSelector(RuntimeConfig())
    peer = observation(x=.75, scale=.4, hand="Left", label="dorsal")
    _, selected, _ = feed(selector, [observation(scale=.6, label="dorsal"), peer])[-1]
    owner = selected["selected_track_id"]
    assert selected["dorsal_foreground_clear"]
    for frame in range(10, 30):
        near = observation(scale=.50 if frame % 2 else .52, label="dorsal")
        _, selection, _ = selector.update([peer, near], SHAPE, frame * .1)
        assert selection["selected_track_id"] == owner
        assert selection["dorsal_foreground_clear"]
    _, selection, _ = selector.update([observation(scale=.44, label="dorsal"), peer], SHAPE, 3.)
    assert not selection["dorsal_foreground_clear"]
    _, selection, _ = selector.update([observation(scale=.50, label="dorsal"), peer], SHAPE, 3.1)
    assert not selection["dorsal_foreground_clear"]  # Must reacquire 1.30x clearance.


def test_preview_smoothing_reduces_jitter_without_changing_current_landmarks():
    selector = AutomaticHandSelector(RuntimeConfig())
    feed(selector, [observation(scale=.6), observation(x=.75, scale=.3, hand="Left")])
    raw, displayed = [], []
    for frame in range(10, 50):
        near = observation(x=.3 + (.003 if frame % 2 else -.003), scale=.6)
        peer = observation(x=.75, scale=.3, hand="Left")
        original = near["points"].copy()
        chosen, _, previews = selector.update([near, peer], SHAPE, frame * .1)
        assert chosen is near
        np.testing.assert_array_equal(chosen["points"], original)
        np.testing.assert_allclose(previews[0]["landmarks"], original, atol=1e-6)
        raw.append(original[0, 0])
        displayed.append(previews[0]["display_landmarks"][0][0])
    assert np.std(displayed) < .6 * np.std(raw)
    chosen, _, previews = selector.update([], SHAPE, 5.)
    assert chosen is None and previews == []  # No synthetic observations on a miss.


@pytest.mark.parametrize("label", ["left", "right"])
def test_live_direction_stays_recognized_when_two_hand_labels_flip(label):
    config = RuntimeConfig()
    probabilities = np.eye(len(config.class_names))[config.class_to_idx[label]]
    quality = dict(known_gesture_mass=1.)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine._lock = config, threading.Lock()
    engine.detector = simulated_detector()
    engine.model_manager = SimpleNamespace(
        ready=True, models={"ONNX": object()}, resolve_name=lambda name: "ONNX",
        predict_detailed=lambda *args: (probabilities.copy(), "ONNX", .01, quality),
        predict_many=lambda *args: {"ONNX": (probabilities.copy(), .01, quality)})
    def resolve(p, points, *, handedness, **kwargs):
        expected = "Right" if points[0, 0] < .5 else "Left"
        if expected == "Left":
            assert kwargs.get("preferred_direction") is None
        return p, {"pose_validation": dict(valid=handedness == expected,
                                            geometry_supported=True, reason="side check")}
    engine.resolver = SimpleNamespace(resolve=resolve)
    native_rows = []
    engine.detector._run_candidates = lambda *args, **kwargs: native_rows
    session = RuntimeSession(config)
    image = cv2.imencode(".jpg", np.full(SHAPE, 145, np.uint8))[1].tobytes()
    owner = None
    for frame in range(35):
        near = observation(scale=.6, label=label)
        peer = observation(x=.75, scale=.3, hand="Left", label=label)
        if frame >= 10 and frame % 2:
            near["handedness"] = "Left"
        native_rows = [(row["area"], row["points"], SimpleNamespace(
            category_name=row["handedness"], score=.99), None) for row in (near, peer)]
        if frame % 2:
            native_rows.reverse()
        result = engine.process_frame(image, session, now=1 + frame * .1)
        if frame == 9:
            owner = result["hand_selection"]["selected_track_id"]
        if frame >= 10:
            assert result["runtime_prediction"] == label
            assert result["hand_selection"]["selected_track_id"] == owner
            assert result["quality"]["detector"]["selected_handedness"] == "Right"
            assert not result["quality"]["detector"]["reacquired"]


@pytest.mark.parametrize("label", ["left", "right"])
def test_uncertain_distance_keeps_fresh_selected_hand_visible_and_pauses_actions(label):
    config = RuntimeConfig()
    probability = np.eye(len(config.class_names))[config.class_to_idx[label]]
    quality = dict(known_gesture_mass=1.)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine._lock = config, threading.Lock()
    engine.detector = simulated_detector()
    engine.model_manager = SimpleNamespace(
        ready=True, models={"ONNX": object()}, resolve_name=lambda name: "ONNX",
        predict_detailed=lambda *args: (probability.copy(), "ONNX", .01, quality),
        predict_many=lambda *args: {"ONNX": (probability.copy(), .01, quality)})
    engine.resolver = SimpleNamespace(resolve=lambda p, points, **kwargs: (
        p, {"pose_validation": dict(valid=True, geometry_supported=True)}))
    rows = [observation(scale=.6, label=label),
            observation(x=.75, scale=.3, hand="Left", label=label)]
    engine.detector.detect_candidates = lambda *args, **kwargs: rows
    engine.detector._last_diagnostics = {}
    image = cv2.imencode(".jpg", np.full(SHAPE, 145, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    for frame in range(10):
        result = engine.process_frame(image, session, now=1 + frame * .1)
    owner = result["hand_selection"]["selected_track_id"]
    for frame in range(10, 16):
        near = observation(x=.3 + .001 * frame, scale=.6, label=label)
        peer = observation(x=.75, scale=.3, hand="Left", label=label)
        near["metrics"]["palm_segments_px"] = [70., 35., 75., 20.]
        peer["metrics"]["palm_segments_px"] = [35., 35., 35., 35.]
        rows = [near, peer]
        result = engine.process_frame(image, session, now=1 + frame * .1)
        assert result["hand_selection"]["selected_track_id"] == owner
        assert result["hand_selection"]["switch_pending"]
        assert result["runtime_prediction"] == label
        assert result["runtime_action"] == "Wait / No Action"
        assert result["action_reason"] == "hand distance uncertain"
        np.testing.assert_allclose(result["landmarks"], near["points"], atol=1e-6)
