from __future__ import annotations

import json

import numpy as np

from backend.config import MODELS_DIRECTORY, load_runtime_config
from backend.model_runtime import ModelManager


def _macro_f1(truth: np.ndarray, predicted: np.ndarray, class_count: int) -> float:
    scores = []
    for class_index in range(class_count):
        actual = truth == class_index
        guessed = predicted == class_index
        true_positive = int(np.count_nonzero(actual & guessed))
        false_positive = int(np.count_nonzero(~actual & guessed))
        false_negative = int(np.count_nonzero(actual & ~guessed))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def test_untouched_cache_qualifies_ten_class_model_and_open_set_gate():
    config = load_runtime_config()
    manager = ModelManager(config)
    assert manager.ready, manager.errors
    with np.load(
        MODELS_DIRECTORY / "gesture_online_untouched_test_cache.npz",
        allow_pickle=False,
    ) as cache:
        features = cache["X"].astype(np.float32)
        labels = cache["y"].astype(np.int64)
        cache_class_names = [str(value) for value in cache["class_names"]]
    assert cache_class_names == config.class_names

    classifier = manager.models["ONNX"]
    probabilities, known_mass = classifier.predict_with_quality(features)
    known = labels >= 0
    unknown = ~known
    predicted = probabilities.argmax(axis=1)

    accuracy = float(np.mean(predicted[known] == labels[known]))
    macro_f1 = _macro_f1(labels[known], predicted[known], len(config.class_names))
    accepted = known_mass >= config.known_mass_floor
    runtime_predicted = np.where(accepted, predicted, -1)
    false_acceptance = float(np.mean(accepted[unknown]))

    assert accuracy >= 0.985
    assert macro_f1 >= 0.98
    assert float(np.mean(accepted[known])) >= 0.96
    assert false_acceptance <= 0.03

    for gesture in ("down", "dorsal", "fist", "thumb_down"):
        index = config.class_to_idx[gesture]
        actual = labels == index
        true_positive = np.count_nonzero(actual & (runtime_predicted == index))
        false_positive = np.count_nonzero(~actual & (runtime_predicted == index))
        false_negative = np.count_nonzero(actual & (runtime_predicted != index))
        assert 2 * true_positive / (2 * true_positive + false_positive + false_negative) >= 0.95


def test_direction_source_columns_are_ncm_calibrated_without_pixel_mirroring():
    config = load_runtime_config()
    metadata = json.loads(
        (MODELS_DIRECTORY / "gesture_mlp_onnx_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["source_class_mapping"]["left"] == "one_right"
    assert metadata["source_class_mapping"]["right"] == "one_left"
    calibration = metadata["horizontal_direction_calibration"]
    assert calibration["camera_pixels_mirrored"] is False
    assert calibration["positive_index_dx_command"] == "left"
    assert calibration["negative_index_dx_command"] == "right"
    assert (calibration.get("source_columns_swapped") is True
            or calibration.get("training_labels_ncm_calibrated") is True)
    with np.load(
        MODELS_DIRECTORY / "gesture_online_validation_cache.npz",
        allow_pickle=False,
    ) as cache:
        labels = cache["y"].astype(np.int64)
        landmarks = cache["landmarks"].astype(np.float32)

    # The NCM calibration deliberately swaps the old training columns while the
    # underlying landmark coordinates remain untouched.
    for label, expected_sign in (("left", 1), ("right", -1)):
        rows = landmarks[labels == config.class_to_idx[label]]
        index_direction_x = rows[:, 8, 0] - rows[:, 5, 0]
        assert np.all(np.sign(index_direction_x) == expected_sign)
