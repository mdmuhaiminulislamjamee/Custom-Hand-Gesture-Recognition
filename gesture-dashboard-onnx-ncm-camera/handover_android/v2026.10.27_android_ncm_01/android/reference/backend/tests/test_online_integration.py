from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi import HTTPException

import backend.app as app_module
from backend.config import RuntimeConfig
from backend.model_runtime import ModelManager


class _Classifier:
    def __init__(self, probabilities: np.ndarray, known_mass: float):
        self.probabilities = probabilities
        self.known_mass = known_mass
        self.classes_ = np.arange(len(probabilities), dtype=int)

    def predict_with_quality(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.repeat(self.probabilities[None, :], len(features), axis=0),
            np.full(len(features), self.known_mass, dtype=np.float64),
        )


def _manager(tmp_path, adapter: Any) -> tuple[ModelManager, np.ndarray, np.ndarray]:
    config = RuntimeConfig()
    probabilities = np.full(len(config.class_names), 0.01, dtype=np.float64)
    probabilities[config.class_to_idx["left"]] = 0.93
    probabilities /= probabilities.sum()
    feature = np.zeros(len(config.feature_names), dtype=np.float32)
    manager = ModelManager(config, tmp_path, load_models=False)
    manager.models["ONNX"] = _Classifier(probabilities, known_mass=0.95)
    manager.selected_model_name = "ONNX"
    manager.set_online_adapter(adapter)
    return manager, probabilities, feature


def test_model_manager_consumes_adapter_rejection_mass(tmp_path):
    class RejectingAdapter:
        call: tuple[np.ndarray, np.ndarray, float] | None = None

        def apply_with_quality(
            self,
            probabilities: np.ndarray,
            feature: np.ndarray,
            *,
            known_gesture_mass: float,
        ) -> tuple[np.ndarray, dict[str, float]]:
            self.call = (probabilities.copy(), feature.copy(), known_gesture_mass)
            return probabilities, {
                "base_known_gesture_mass": known_gesture_mass,
                "known_gesture_mass": 0.10,
                "acceptance_scale": 0.10 / known_gesture_mass,
                "rejection_score": 1.0 - 0.10 / known_gesture_mass,
            }

    adapter = RejectingAdapter()
    manager, base, feature = _manager(tmp_path, adapter)

    probabilities, _, _, quality = manager.predict_detailed(feature)

    assert adapter.call is not None
    assert np.allclose(adapter.call[0], base)
    assert np.allclose(adapter.call[1], feature)
    assert adapter.call[2] == pytest.approx(0.95)
    assert np.allclose(probabilities, base)
    assert quality["known_gesture_mass"] == pytest.approx(0.10)
    assert quality["online_adapter_applied"] is True


def test_model_manager_restores_base_output_if_adapter_result_is_invalid(tmp_path):
    class InvalidAdapter:
        def apply_with_quality(
            self,
            probabilities: np.ndarray,
            feature: np.ndarray,
            *,
            known_gesture_mass: float,
        ) -> tuple[np.ndarray, dict[str, float]]:
            del feature, known_gesture_mass
            # The bad probability shape fails after the candidate mass is read.
            return probabilities[:-1], {"known_gesture_mass": 0.01}

    manager, base, feature = _manager(tmp_path, InvalidAdapter())

    probabilities, _, _, quality = manager.predict_detailed(feature)

    assert np.allclose(probabilities, base)
    assert quality["known_gesture_mass"] == pytest.approx(0.95)
    assert quality["online_adapter_applied"] is False
    assert quality["online_adapter"]["state_error"]


def _feedback_payload(*, learning_mode: str = "safe") -> app_module.FeedbackPayload:
    config = RuntimeConfig()
    displayed = {
        name: (0.86 if name == "left" else 0.02) for name in config.class_names
    }
    base = {
        name: (0.86 if name == "right" else 0.02) for name in config.class_names
    }
    return app_module.FeedbackPayload(
        actual_label="left",
        predicted_label="right",
        confidence=0.86,
        model_name="ONNX",
        probabilities=displayed,
        base_probabilities=base,
        feature_vector=[0.0] * 76,
        learning_mode=learning_mode,
    )


def test_feedback_service_saves_review_and_learns_from_base_probabilities(monkeypatch):
    class Store:
        record: Any | None = None

        def save(self, record):
            self.record = record
            return {"status": "saved", "feedback_id": "review-1", "image_saved": False}

    class Learning:
        call: dict[str, Any] | None = None

        def learn(self, feature, label, **kwargs):
            self.call = {"feature": feature, "label": label, **kwargs}
            return {"status": "accepted", "message": "accepted"}

    store = Store()
    learning = Learning()
    monkeypatch.setattr(app_module, "feedback_store", store)
    monkeypatch.setattr(app_module, "online_learning", learning)

    result = app_module.save_feedback(_feedback_payload())

    assert result["status"] == "saved"
    assert result["online_update"]["status"] == "accepted"
    assert store.record is not None
    assert store.record.probabilities["left"] == pytest.approx(0.86)
    assert learning.call is not None
    assert learning.call["label"] == "left"
    assert learning.call["probabilities"]["right"] == pytest.approx(0.86)
    assert learning.call["mode"] == "safe"
    assert learning.call["reviewed"] is True


def test_feedback_service_reports_post_save_learning_failure_in_band(monkeypatch):
    class Store:
        saves = 0

        def save(self, record):
            del record
            self.saves += 1
            return {"status": "saved", "feedback_id": "review-2", "image_saved": False}

    class Learning:
        def learn(self, feature, label, **kwargs):
            del feature, label, kwargs
            raise RuntimeError("validation cache unavailable")

    store = Store()
    monkeypatch.setattr(app_module, "feedback_store", store)
    monkeypatch.setattr(app_module, "online_learning", Learning())

    result = app_module.save_feedback(_feedback_payload())

    assert store.saves == 1
    assert result["status"] == "saved"
    assert result["online_update"]["status"] == "error"
    assert "validation cache unavailable" in result["online_update"]["message"]


def test_feedback_service_keeps_force_learning_disabled(monkeypatch):
    class Store:
        saves = 0

        def save(self, record):
            del record
            self.saves += 1
            return {"status": "saved"}

    store = Store()
    monkeypatch.setattr(app_module, "feedback_store", store)

    with pytest.raises(HTTPException) as raised:
        app_module.save_feedback(_feedback_payload(learning_mode="force"))

    assert raised.value.status_code == 403
    assert store.saves == 0
