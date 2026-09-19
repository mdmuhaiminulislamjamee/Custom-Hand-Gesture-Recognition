from __future__ import annotations

import json

import numpy as np

from backend.artifact_integrity import ArtifactRegistry
from backend.config import CLASS_NAMES, MODELS_DIRECTORY, load_runtime_config
from backend.model_runtime import ModelManager
from backend.app import _onnx_qualification


def test_artifact_manifest_and_onnx_metadata_are_current():
    verification = ArtifactRegistry().verify()
    assert verification["verified"], verification["errors"]
    metadata = json.loads(
        (MODELS_DIRECTORY / "gesture_mlp_onnx_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["format"] == "ONNX"
    assert metadata["parity"]["passed"] is True
    assert metadata["parity"]["prediction_agreement"] >= 0.99
    assert metadata["output_class_order"] == CLASS_NAMES
    assert metadata["known_mass_output"] == "known_gesture_mass"


def test_health_qualification_uses_the_current_ten_gesture_gate_schema():
    qualification = _onnx_qualification()

    assert qualification["status"] == "passed", qualification["failing_gates"]
    assert qualification["current"] is True
    assert qualification["pc_release_ready"] is True
    assert qualification["failing_gates"] == []
    assert qualification["gates"]["complete_release_checks"] is True
    assert qualification["gates"]["complete_confirmation_checks"] is True
    assert qualification["gates"]["selected_thresholds_match_runtime"] is True


def test_model_manager_loads_only_qualified_onnx_runtime():
    config = load_runtime_config()
    manager = ModelManager(config)
    assert manager.ready, manager.errors
    assert manager.status()["available_models"] == ["ONNX"]
    assert manager.status()["selectable_models"] == ["ONNX"]
    probabilities, model_name, _ = manager.predict(
        np.zeros(len(config.feature_names), dtype=np.float32)
    )
    assert model_name == "ONNX"
    assert probabilities.shape == (len(config.class_names),)
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1.0)
    _, _, _, quality = manager.predict_detailed(
        np.zeros(len(config.feature_names), dtype=np.float32)
    )
    assert 0.0 <= quality["known_gesture_mass"] <= 1.0


def test_handover_folder_contains_no_python_pickle_model():
    assert not list(MODELS_DIRECTORY.glob("*.joblib"))
    assert not list(MODELS_DIRECTORY.glob("*.pkl"))


def test_open_set_output_stays_finite_when_retained_mass_underflows():
    config = load_runtime_config()
    classifier = ModelManager(config).models["ONNX"]
    rows = np.vstack([
        np.full((1, len(config.feature_names)), value, dtype=np.float32)
        for value in (-1e6, -1e3, 1e3, 1e6)
    ])

    probabilities, known_mass = classifier.predict_with_quality(rows)

    assert np.isfinite(probabilities).all()
    assert np.isfinite(known_mass).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((known_mass >= 0.0) & (known_mass <= 1.0)).all()
