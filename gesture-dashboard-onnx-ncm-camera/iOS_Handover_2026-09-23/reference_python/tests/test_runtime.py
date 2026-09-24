import asyncio

import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.model_runtime import (
    InferenceEngine,
    ModelManager,
    RuntimeSession,
    estimate_hand_distance_m,
    hand_plane_angles,
)
from backend.online_learning import OnlineLearningAdapter
from backend.runtime import InferenceStartLimiter, TemporalGate, probability_ema


def test_default_runtime_contract_is_backend_safe_ten_fps():
    config = RuntimeConfig()

    assert config.target_fps == 10.0
    assert config.frame_interval_ms == 100
    assert config.frame_budget_ms == 100.0
    assert config.stable_frames_required == 3


def test_inference_start_limiter_serializes_aggregate_starts_at_ten_fps():
    now = 0.0
    sleep_delays: list[float] = []

    def clock() -> float:
        return now

    async def sleep(delay: float) -> None:
        nonlocal now
        sleep_delays.append(delay)
        now += delay
        await asyncio.sleep(0)

    limiter = InferenceStartLimiter(100, clock=clock, sleep=sleep)

    async def scenario() -> list[float]:
        return list(
            await asyncio.gather(
                limiter.wait_for_start(),
                limiter.wait_for_start(),
                limiter.wait_for_start(),
            )
        )

    starts = asyncio.run(scenario())

    assert np.allclose(starts, [0.0, 0.1, 0.2])
    assert np.allclose(sleep_delays, [0.1, 0.1])


def test_inference_start_limiter_keeps_reservation_after_failure():
    now = 5.0

    def clock() -> float:
        return now

    async def sleep(delay: float) -> None:
        nonlocal now
        now += delay

    # A faster requested interval is still clamped to the hard 10 FPS ceiling.
    limiter = InferenceStartLimiter(1, clock=clock, sleep=sleep)

    async def scenario() -> tuple[float, float]:
        first = await limiter.wait_for_start()
        try:
            raise RuntimeError("simulated inference failure")
        except RuntimeError:
            pass
        second = await limiter.wait_for_start()
        return first, second

    first, second = asyncio.run(scenario())

    assert limiter.interval_seconds == 0.1
    assert abs(second - first - 0.1) < 1e-12


def test_probability_ema_keeps_rows_normalized():
    rows = probability_ema(np.asarray([[0.9, 0.1], [0.7, 0.3]]), alpha=0.65)
    assert rows.shape == (2, 2)
    assert np.allclose(rows.sum(axis=1), 1.0)
    assert rows[-1, 0] > rows[-1, 1]


def test_temporal_gate_requires_stability_confidence_margin_and_hold_time():
    config = RuntimeConfig()
    gate = TemporalGate(config)
    row = np.zeros(len(config.class_names), dtype=np.float64)
    row[config.class_to_idx["left"]] = 0.95
    row[config.class_to_idx["right"]] = 0.05
    first = gate.update(row, now=1.0)
    second = gate.update(row, now=1.1)
    third = gate.update(row, now=1.2)
    assert first.execute is False
    assert second.execute is False
    assert third.execute is True
    assert third.predicted_gesture == "left"


def test_temporal_hold_time_handles_zero_timestamp():
    config = RuntimeConfig(stable_frames_required=2, minimum_hold_seconds=0.10)
    gate = TemporalGate(config)
    row = np.zeros(len(config.class_names), dtype=np.float64)
    row[config.class_to_idx["left"]] = 1.0

    first = gate.update(row, now=0.0)
    second = gate.update(row, now=0.20)

    assert first.execute is False
    assert second.held_seconds == 0.20
    assert second.execute is True


def test_action_latch_requires_configured_consecutive_release_frames():
    config = RuntimeConfig(release_frames_required=3)
    session = RuntimeSession(config)
    session.last_action_gesture = "like"

    assert session.observe_action_release(None) is False
    assert session.observe_action_release(None) is False
    assert session.last_action_gesture == "like"
    # Seeing the held gesture again cancels a transient release sequence.
    assert session.observe_action_release("like") is False
    assert session.observe_action_release(None) is False
    assert session.observe_action_release(None) is False
    assert session.observe_action_release(None) is True
    assert session.last_action_gesture is None


def test_open_set_rejection_never_executes():
    config = RuntimeConfig()
    gate = TemporalGate(config)
    row = np.zeros(len(config.class_names), dtype=np.float64)
    row[config.class_to_idx["left"]] = 1.0
    decision = gate.update(row, now=1.0, known_gesture_mass=0.1)
    assert decision.execute is False
    assert decision.predicted_gesture == "no_gesture"
    assert decision.reason == "outside the ten-gesture vocabulary"


@pytest.mark.parametrize("fps", [5, 8, 10, 15, 20, 30, 60])
def test_fast_frames_cannot_skip_hold_duration(fps):
    config = RuntimeConfig()
    gate = TemporalGate(config)
    row = np.zeros(len(config.class_names))
    row[config.class_to_idx["dorsal"]] = 1.0
    executed = []
    for frame in range(fps + 1):
        decision = gate.update(row, now=frame / fps)
        if decision.execute:
            executed.append(frame / fps)
    assert executed
    assert executed[0] >= gate.config.minimum_hold_seconds


def test_confident_pose_transition_does_not_execute_stale_action():
    config = RuntimeConfig()
    gate = TemporalGate(config)
    left = np.eye(len(config.class_names))[config.class_to_idx["left"]]
    dorsal = np.eye(len(config.class_names))[config.class_to_idx["dorsal"]]
    for now in (1.0, 1.1, 1.2):
        decision = gate.update(left, now=now)
    assert decision.execute
    decision = gate.update(dorsal, now=1.3)
    assert decision.predicted_gesture == "dorsal"
    assert not decision.execute
    assert decision.stable_frames == 1
    assert not gate.update(dorsal, now=1.4).execute
    assert gate.update(dorsal, now=1.5).execute


def test_capture_outage_requires_new_confirmation():
    config = RuntimeConfig()
    gate = TemporalGate(config)
    dorsal = np.eye(len(config.class_names))[config.class_to_idx["dorsal"]]
    for now in (1.0, 1.1, 1.2):
        gate.update(dorsal, now=now)
    decision = gate.update(dorsal, now=2.0)
    assert not decision.execute
    assert decision.stable_frames == 1


@pytest.mark.parametrize("gesture", ["open_palm", "like", "fist", "thumb_down"])
def test_geometry_supported_hand_commands_can_recover_from_low_known_mass(gesture):
    config = RuntimeConfig(known_mass_floor=0.95)
    gate = TemporalGate(config)
    row = np.eye(len(config.class_names))[config.class_to_idx[gesture]]

    rejected = gate.update(
        row, now=0.0, known_gesture_mass=0.1, geometry_supported=True
    )
    assert rejected.predicted_gesture == config.reject_label

    for now in (1.0, 1.1, 1.2, 1.3, 1.4):
        decision = gate.update(
            row,
            now=now,
            known_gesture_mass=0.2,
            geometry_supported=True,
        )
        assert decision.execute is False
    decision = gate.update(
        row,
        now=1.5,
        known_gesture_mass=0.2,
        geometry_supported=True,
    )
    assert decision.execute is True
    assert decision.predicted_gesture == gesture


def test_nonfinite_classifier_evidence_fails_closed():
    config = RuntimeConfig()
    row = np.full(len(config.class_names), 1.0 / len(config.class_names))
    row[0] = np.nan

    decision = TemporalGate(config).update(row, known_gesture_mass=np.nan)

    assert decision.execute is False
    assert decision.predicted_gesture == config.reject_label
    assert decision.reason == "invalid classifier probabilities"
    assert all(np.isfinite(list(decision.probabilities.values())))


def test_reviewed_negative_feedback_reaches_runtime_known_mass(tmp_path):
    config = RuntimeConfig()
    feature = np.zeros(len(config.feature_names), dtype=np.float32)
    probabilities = np.full(len(config.class_names), 0.01, dtype=np.float64)
    probabilities[config.class_to_idx["left"]] = 0.93
    probabilities /= probabilities.sum()

    class Classifier:
        classes_ = np.arange(len(config.class_names), dtype=int)

        def predict_with_quality(self, features):
            return (
                np.repeat(probabilities[None, :], len(features), axis=0),
                np.full((len(features),), 0.95, dtype=np.float64),
            )

    manager = ModelManager(config, tmp_path, load_models=False)
    manager.models["ONNX"] = Classifier()
    manager.selected_model_name = "ONNX"
    adapter = OnlineLearningAdapter(config, models_directory=tmp_path)
    update = adapter.learn(
        feature,
        config.reject_label,
        probabilities=probabilities,
    )
    assert update["status"] == "accepted"
    manager.set_online_adapter(adapter)

    adapted, _, _, quality = manager.predict_detailed(feature)

    assert np.allclose(adapted, probabilities)
    assert quality["online_adapter_applied"] is True
    assert quality["known_gesture_mass"] < config.known_mass_floor
    decision = TemporalGate(config).update(
        adapted,
        now=1.0,
        known_gesture_mass=quality["known_gesture_mass"],
    )
    assert decision.predicted_gesture == config.reject_label
    assert decision.execute is False


def test_configured_fps_budget_verdict_uses_total_pipeline_time():
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config = RuntimeConfig(target_fps=10.0)
    passing = engine._timing(
        mediapipe_ms=33.9,
        feature_ms=0.2,
        classifier_ms=27.761,
        diagnostics_ms=27.761,
        total_ms=61.66,
        effective_fps=10.0,
    )
    assert passing["ten_fps_capacity_pass"] is True
    assert passing["target_fps_capacity_pass"] is True
    assert passing["verdict"] == "PASS"
    assert abs(passing["uncapped_fps"] - 16.218) < 0.01
    assert abs(passing["budget_used_percent"] - 61.66) < 0.01

    failing = engine._timing(
        mediapipe_ms=80.0,
        feature_ms=1.0,
        classifier_ms=30.0,
        diagnostics_ms=30.0,
        total_ms=111.0,
        effective_fps=9.0,
    )
    assert failing["ten_fps_capacity_pass"] is False
    assert failing["target_fps_capacity_pass"] is False
    assert failing["verdict"] == "FAIL"


def test_hand_plane_angles_computes_xy_yz_xz_and_zx():
    world_landmarks = np.zeros((21, 3), dtype=np.float32)
    world_landmarks[[5, 9, 13, 17]] = [0.0, 1.0, 0.5]
    angles = hand_plane_angles(world_landmarks)
    assert angles is not None
    assert "xy" in angles
    assert "yz" in angles
    assert "xz" in angles
    assert "zx" in angles
    assert np.isclose(angles["xy"], 90.0)
    assert np.isclose(angles["yz"], np.degrees(np.arctan2(0.5, 1.0)))
    assert np.isclose(angles["zx"], np.degrees(np.arctan2(0.0, 0.5)))
    assert np.isclose(angles["xz"], np.degrees(np.arctan2(0.5, 0.0)))


def test_estimate_hand_distance_m():
    quality = {"palm_scale_px": 80.0, "analysis_width": 640}
    dist = estimate_hand_distance_m(quality)
    assert dist is not None
    assert np.isclose(dist, 0.51, atol=0.01)

    assert estimate_hand_distance_m(None) is None
    assert estimate_hand_distance_m({"palm_scale_px": 4}) is None

