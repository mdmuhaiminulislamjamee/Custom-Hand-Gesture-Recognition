import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.config import CLASS_NAMES
from backend.online_learning import OnlineLearningAdapter


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        class_names=CLASS_NAMES,
        feature_names=[f"feature_{index}" for index in range(76)],
    )


def _base_row(target: int = 0, wrong: int = 1) -> np.ndarray:
    row = np.full(len(CLASS_NAMES), 0.01, dtype=np.float64)
    row[target] = 0.05
    row[wrong] = 0.89
    return row / row.sum()


def test_requires_exact_ten_class_and_76_feature_contract(tmp_path: Path):
    with pytest.raises(ValueError, match="ten-command"):
        OnlineLearningAdapter(CLASS_NAMES[:-1], models_directory=tmp_path)
    bad_features = SimpleNamespace(class_names=CLASS_NAMES, feature_names=["x"] * 76)
    with pytest.raises(ValueError, match="76-D"):
        OnlineLearningAdapter(bad_features, models_directory=tmp_path)


def test_safe_reviewed_update_is_applied_persisted_and_backed_up(tmp_path: Path):
    adapter = OnlineLearningAdapter(_config(), models_directory=tmp_path)
    feature = np.zeros(76, dtype=np.float32)
    base = _base_row()

    update = adapter.learn(feature, "left", probabilities=base, mode="safe")

    assert update["status"] == "accepted"
    assert update["safety_gate_passed"] is True
    assert update["probability_after"] > update["probability_before"]
    assert update["validation_source"] == "reviewed_sample_only"
    assert adapter.status()["accepted_updates"] == 1
    assert adapter.status()["active_prototypes"] == 1
    assert len(adapter.list_backups()) == 1
    assert not list(tmp_path.glob("*.tmp"))

    adapted = adapter.apply(base, feature)
    assert adapted.shape == (len(CLASS_NAMES),)
    assert adapted.sum() == pytest.approx(1.0)
    assert adapted[0] > base[0]

    reloaded = OnlineLearningAdapter(_config(), models_directory=tmp_path)
    assert reloaded.status()["accepted_updates"] == 1
    assert np.allclose(reloaded.apply(base, feature), adapted)


def test_holdout_regression_rejects_candidate_and_persists_counter(tmp_path: Path):
    feature = np.zeros(76, dtype=np.float32)
    holdout_p = _base_row(target=1, wrong=0).reshape(1, -1)
    np.savez_compressed(
        tmp_path / "gesture_online_validation_cache.npz",
        X=feature.reshape(1, -1),
        y=np.asarray([1], dtype=np.int64),
        base_probabilities=holdout_p,
        class_names=np.asarray(CLASS_NAMES),
    )
    adapter = OnlineLearningAdapter(_config(), models_directory=tmp_path)
    base = _base_row(target=0, wrong=1)

    update = adapter.learn(feature, "left", probabilities=base)

    assert update["status"] == "rejected"
    assert update["changes"]["log_loss_increase"] > 0.005
    assert np.allclose(adapter.apply(base, feature), base)
    status = adapter.status()
    assert status["attempted_updates"] == 1
    assert status["rejected_updates"] == 1
    assert status["active_prototypes"] == 0

    reloaded = OnlineLearningAdapter(_config(), models_directory=tmp_path)
    assert reloaded.status()["rejected_updates"] == 1


def test_cache_without_probabilities_can_use_base_predictor(tmp_path: Path):
    rows = np.vstack(
        [np.full(76, index / 10.0) for index in range(len(CLASS_NAMES) + 1)]
    )
    labels = np.arange(-1, len(CLASS_NAMES), dtype=np.int64)
    np.savez_compressed(
        tmp_path / "gesture_online_replay_cache.npz",
        X=rows,
        y=labels,
        class_names=np.asarray(CLASS_NAMES),
    )

    def predictor(features: np.ndarray) -> np.ndarray:
        probabilities = np.full(
            (len(features), len(CLASS_NAMES)), 0.01, dtype=np.float64
        )
        indexes = np.clip(
            np.rint(features[:, 0] * 10).astype(int), 0, len(CLASS_NAMES) - 1
        )
        probabilities[np.arange(len(features)), indexes] = 0.93
        return probabilities

    adapter = OnlineLearningAdapter(
        _config(), models_directory=tmp_path, base_predictor=predictor
    )
    cache = adapter.status()["replay_cache"]
    assert cache["valid"] is True
    assert cache["usable"] is True
    assert cache["probabilities_stored"] is False
    assert cache["covered_classes"] == len(CLASS_NAMES)
    assert cache["has_reject_rows"] is True


def test_reviewed_no_gesture_creates_local_known_mass_rejection(tmp_path: Path):
    adapter = OnlineLearningAdapter(_config(), models_directory=tmp_path)
    feature = np.zeros(76)
    base = _base_row()

    update = adapter.learn(feature, "no_gesture", probabilities=base)

    assert update["status"] == "accepted"
    assert update["probability_before"] is None
    assert update["rejection_score_after"] > update["rejection_score_before"]
    adapted, quality = adapter.apply_with_quality(
        base, feature, known_gesture_mass=0.95
    )
    assert np.allclose(adapted, base)
    assert quality["acceptance_scale"] < 0.2
    assert quality["known_gesture_mass"] < 0.2
    assert quality["rejection_score"] > 0.8
    assert adapter.status()["negative_prototypes"] == 1

    far_feature = np.full(76, 2.0)
    _, far_quality = adapter.apply_with_quality(
        base, far_feature, known_gesture_mass=0.95
    )
    assert far_quality["acceptance_scale"] == pytest.approx(1.0)
    assert far_quality["known_gesture_mass"] == pytest.approx(0.95)


def test_require_holdout_fails_closed_when_optional_caches_are_absent(tmp_path: Path):
    adapter = OnlineLearningAdapter(
        _config(), models_directory=tmp_path, require_holdout_for_safe=True
    )
    assert adapter.status()["ready"] is True
    assert adapter.status()["safe_mode_ready"] is False
    with pytest.raises(RuntimeError, match="requires a usable holdout"):
        adapter.learn(np.zeros(76), "left", probabilities=_base_row())


def test_rollback_restores_pre_update_adapter_and_keeps_lifetime_counters(tmp_path: Path):
    adapter = OnlineLearningAdapter(_config(), models_directory=tmp_path)
    feature = np.zeros(76)
    base = _base_row()
    update = adapter.learn(feature, "left", probabilities=base)
    assert adapter.apply(base, feature)[0] > base[0]

    result = adapter.rollback(update["backup"])

    assert result["status"] == "rolled_back"
    assert np.allclose(adapter.apply(base, feature), base)
    assert adapter.status()["accepted_updates"] == 1
    assert adapter.status()["rollback_count"] == 1


def test_corrupt_state_fails_open_for_inference_and_closed_for_learning(tmp_path: Path):
    (tmp_path / "gesture_online_adapter_state.npz").write_bytes(b"not-an-npz")
    adapter = OnlineLearningAdapter(_config(), models_directory=tmp_path)
    base = _base_row()
    feature = np.zeros(76)

    assert adapter.status()["ready"] is False
    assert np.allclose(adapter.apply(base, feature), base)
    with pytest.raises(RuntimeError, match="state is invalid"):
        adapter.learn(feature, "left", probabilities=base)


def test_old_direction_semantics_are_not_loaded_after_left_right_calibration(
    tmp_path: Path,
):
    adapter = OnlineLearningAdapter(_config(), models_directory=tmp_path)
    adapter.learn(np.zeros(76), "left", probabilities=_base_row())
    state_path = tmp_path / "gesture_online_adapter_state.npz"
    with np.load(state_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
    metadata = json.loads(str(payload["metadata_json"].item()))
    metadata["direction_semantic_version"] = "legacy_horizontal_mapping"
    payload["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(state_path, **payload)

    reloaded = OnlineLearningAdapter(_config(), models_directory=tmp_path)

    status = reloaded.status()
    assert status["ready"] is True
    assert status["safe_mode_ready"] is True
    assert status["state_error"] is None
    assert "Safe Learn started fresh" in str(status["state_notice"])
    assert Path(str(status["quarantined_state_path"])).is_file()
    assert not state_path.exists()
    assert np.allclose(reloaded.apply(_base_row(), np.zeros(76)), _base_row())


def test_adapter_state_is_bound_to_the_base_onnx_model(tmp_path: Path):
    first_manager = SimpleNamespace(
        bundle_metadata={"model_sha256": "model-a"}, models={}
    )
    first = OnlineLearningAdapter(
        _config(), model_manager=first_manager, models_directory=tmp_path
    )
    first.learn(np.zeros(76), "left", probabilities=_base_row())

    next_manager = SimpleNamespace(
        bundle_metadata={"model_sha256": "model-b"}, models={}
    )
    reloaded = OnlineLearningAdapter(
        _config(), model_manager=next_manager, models_directory=tmp_path
    )

    status = reloaded.status()
    assert status["ready"] is True
    assert status["safe_mode_ready"] is True
    assert status["state_error"] is None
    assert "Safe Learn started fresh" in str(status["state_notice"])
    assert Path(str(status["quarantined_state_path"])).is_file()


def test_apply_supports_batches_and_rejects_unsafe_learning_modes(tmp_path: Path):
    adapter = OnlineLearningAdapter(_config(), models_directory=tmp_path)
    base = _base_row()
    rows = np.vstack([base, base])
    features = np.zeros((2, 76))
    assert adapter.apply(rows, features).shape == (2, len(CLASS_NAMES))
    with pytest.raises(PermissionError, match="safe learning"):
        adapter.learn(features[0], "left", probabilities=base, force=True)
    with pytest.raises(ValueError, match="explicitly reviewed"):
        adapter.learn(features[0], "left", probabilities=base, reviewed=False)
