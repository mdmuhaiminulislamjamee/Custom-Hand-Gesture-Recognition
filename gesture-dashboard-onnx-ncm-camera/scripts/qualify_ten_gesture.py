"""Select rejection gates, qualify the ten-command model, and deploy atomically.

Thresholds are selected only on the participant-disjoint validation split.  The
public test split is opened afterwards and is the deployment gate.  ``fist`` and
``thumb_down`` must each clear their own quality floor; aggregate accuracy cannot
hide a weak new command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from backend.artifact_integrity import ArtifactRegistry
from backend.config import CLASS_NAMES, FEATURE_NAMES, GESTURE_TO_ACTION, RuntimeConfig
from backend.hand_detection import DETECTOR_SETTINGS
from research.v18_20 import (
    ARTIFACT_DIRECTORY,
    COMMAND_COUNT,
    INTERNAL_COUNT,
    LABELS,
    REJECT_INDEX,
    benchmark,
    evaluate,
    load_npz,
    predict_onnx,
)


def _runtime_config(settings: dict[str, float]) -> RuntimeConfig:
    return RuntimeConfig(
        known_mass_floor=settings["known_mass_floor"],
        confidence_floor=settings["confidence_floor"],
        probability_margin_floor=settings["probability_margin_floor"],
        raw={
            "open_palm_orientation": {
                "minimum_up_alignment": 0.70,
                "minimum_axis_dominance": 0.70,
            }
        },
    )


def select_thresholds(candidate: Path, validation: dict) -> tuple[dict, dict]:
    """Grid-search open-set gates on validation only, with safety constraints.

    Pose geometry is deliberately applied after selection.  The three scalar
    model gates can therefore be searched vectorially and the selected setting
    is then checked end-to-end, including geometry, on validation and test.
    """
    from sklearn.metrics import f1_score

    probabilities, mass = predict_onnx(candidate, validation["X"])
    truth = np.asarray(validation["y"], dtype=np.int64)
    known = truth < COMMAND_COUNT
    raw = probabilities.argmax(axis=1)
    ordered = np.sort(probabilities, axis=1)
    confidence_values = ordered[:, -1]
    margin_values = ordered[:, -1] - ordered[:, -2]
    rows: list[dict] = []
    for known_mass in (0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95):
        for confidence in (0.70, 0.75, 0.80, 0.85):
            for margin in (0.08, 0.12, 0.16, 0.18, 0.22):
                settings = {
                    "known_mass_floor": known_mass,
                    "confidence_floor": confidence,
                    "probability_margin_floor": margin,
                }
                accepted = (
                    (mass >= known_mass)
                    & (confidence_values >= confidence)
                    & (margin_values >= margin)
                )
                predicted = np.where(accepted, raw, REJECT_INDEX)
                metrics = {
                    "runtime_macro_f1": float(
                        f1_score(
                            truth, predicted, labels=np.arange(INTERNAL_COUNT),
                            average="macro", zero_division=0,
                        )
                    ),
                    "correct_command_rate": float(
                        np.mean(accepted[known] & (predicted[known] == truth[known]))
                    ),
                    "accepted_precision": float(
                        np.mean(predicted[accepted] == truth[accepted])
                    ) if accepted.any() else 0.0,
                    "unknown_false_accept_rate": float(np.mean(accepted[~known])),
                    "known_rejection_rate": float(np.mean(~accepted[known])),
                }
                feasible = (
                    metrics["accepted_precision"] >= 0.99
                    and metrics["unknown_false_accept_rate"] <= 0.02
                )
                score = (
                    metrics["runtime_macro_f1"]
                    + metrics["correct_command_rate"]
                    - 3.0 * max(0.0, metrics["unknown_false_accept_rate"] - 0.02)
                    - 2.0 * max(0.0, 0.99 - metrics["accepted_precision"])
                )
                rows.append({**settings, **metrics, "feasible": feasible, "score": score})
    feasible = [row for row in rows if row["feasible"]]
    pool = feasible or rows
    chosen = max(
        pool,
        key=lambda row: (
            row["score"], row["runtime_macro_f1"],
            -row["known_rejection_rate"], -row["known_mass_floor"],
        ),
    )
    settings = {
        key: float(chosen[key])
        for key in (
            "known_mass_floor", "confidence_floor", "probability_margin_floor"
        )
    }
    return settings, {"selected": chosen, "grid": rows}


def _evaluate_split(candidate: Path, data: dict, config: RuntimeConfig) -> dict:
    probabilities, mass = predict_onnx(candidate, data["X"])
    return evaluate(probabilities, mass, data, config)


def _confusion_rates(result: dict) -> dict[str, float]:
    confusion = np.asarray(result["confusion"], dtype=np.float64)

    def rate(actual: str, predicted: str) -> float:
        row = confusion[CLASS_NAMES.index(actual)]
        return float(row[CLASS_NAMES.index(predicted)] / max(row.sum(), 1.0))

    return {
        "thumb_down_as_like_rate": rate("thumb_down", "like"),
        "like_as_thumb_down_rate": rate("like", "thumb_down"),
        "thumb_down_as_fist_rate": rate("thumb_down", "fist"),
        "fist_as_thumb_down_rate": rate("fist", "thumb_down"),
    }


def _quality_gate(result: dict, latency: dict) -> tuple[bool, dict]:
    metrics = result["metrics"]
    report = result["report"]
    rates = _confusion_rates(result)
    like_f1 = float(report["like"]["f1-score"])
    thumb_f1 = float(report["thumb_down"]["f1-score"])
    checks = {
        "runtime_macro_f1_at_least_0_97": metrics["runtime_macro_f1"] >= 0.97,
        "accepted_precision_at_least_0_99": metrics["accepted_precision"] >= 0.99,
        "unknown_false_accept_rate_at_most_0_02": metrics["unknown_false_accept_rate"] <= 0.02,
        "fist_f1_at_least_0_95": float(report["fist"]["f1-score"]) >= 0.95,
        "fist_recall_at_least_0_95": float(report["fist"]["recall"]) >= 0.95,
        "thumb_down_f1_at_least_0_95": thumb_f1 >= 0.95,
        "thumb_down_recall_at_least_0_95": float(report["thumb_down"]["recall"]) >= 0.95,
        "thumb_down_within_0_02_of_like_f1": thumb_f1 >= like_f1 - 0.02,
        "thumb_down_like_confusion_at_most_0_02": rates["thumb_down_as_like_rate"] <= 0.02,
        "thumb_down_fist_confusion_at_most_0_02": rates["thumb_down_as_fist_rate"] <= 0.02,
        "classifier_p95_below_5_ms": latency["p95_ms"] < 5.0,
    }
    return bool(all(checks.values())), {"checks": checks, "confusion_rates": rates}


def _confirmation_gate(result: dict) -> tuple[bool, dict]:
    metrics = result["metrics"]
    report = result["report"]
    present = [
        name for name in LABELS if float(report[name]["support"]) > 0
    ]
    present_macro_f1 = float(
        np.mean([float(report[name]["f1-score"]) for name in present])
    )
    thumb_f1 = float(report["thumb_down"]["f1-score"])
    like_f1 = float(report["like"]["f1-score"])
    checks = {
        "new_subject_present_class_macro_f1_at_least_0_95": present_macro_f1 >= 0.95,
        "new_subject_accepted_precision_at_least_0_99": metrics["accepted_precision"] >= 0.99,
        "new_subject_unknown_false_accept_rate_at_most_0_02": metrics["unknown_false_accept_rate"] <= 0.02,
        "new_subject_fist_recall_at_least_0_95": float(report["fist"]["recall"]) >= 0.95,
        "new_subject_thumb_down_recall_at_least_0_95": float(report["thumb_down"]["recall"]) >= 0.95,
        "new_subject_thumb_down_within_0_02_of_like_f1": thumb_f1 >= like_f1 - 0.02,
        "new_subject_upward_open_palm_recall_at_least_0_98": float(report["open_palm"]["recall"]) >= 0.98,
    }
    return bool(all(checks.values())), {
        "checks": checks,
        "present_classes": present,
        "present_class_macro_f1": present_macro_f1,
        "confusion_rates": _confusion_rates(result),
    }


def _write_evidence(output: Path, prefix: str, result: dict) -> None:
    pd.DataFrame(result["report"]).T.to_csv(output / f"{prefix}_classification.csv")
    pd.DataFrame(
        result["confusion"], index=LABELS, columns=LABELS
    ).rename_axis("actual").to_csv(output / f"{prefix}_confusion.csv")


def _build_online_caches(root: Path, candidate: Path) -> None:
    """Publish ten-class safe-learning anchors without leaking test into training."""
    output = root / ARTIFACT_DIRECTORY
    models = root / "models"
    balanced = load_npz(output / "balanced_train.npz")
    rng = np.random.default_rng(20260916)
    selected: list[int] = []
    for label in range(INTERNAL_COUNT):
        indexes = np.flatnonzero(balanced["y"] == label)
        selected.extend(rng.permutation(indexes)[:128].tolist())
    selected_array = np.asarray(selected, dtype=np.int64)
    replay_y = balanced["y"][selected_array].astype(np.int64)
    replay_y = np.where(replay_y == REJECT_INDEX, -1, replay_y)
    np.savez_compressed(
        models / "gesture_online_replay_cache.npz",
        X=balanced["X"][selected_array].astype(np.float32),
        y=replay_y,
        class_names=np.asarray(CLASS_NAMES),
    )
    for source_name, destination_name in (
        ("public_val.npz", "gesture_online_validation_cache.npz"),
        ("public_test.npz", "gesture_online_untouched_test_cache.npz"),
    ):
        data = load_npz(output / source_name)
        y = np.where(data["y"] == REJECT_INDEX, -1, data["y"]).astype(np.int64)
        np.savez_compressed(
            models / destination_name,
            X=data["X"].astype(np.float32),
            y=y,
            class_names=np.asarray(CLASS_NAMES),
            landmarks=data["landmarks"].astype(np.float32),
        )


def _deploy(
    root: Path,
    candidate: Path,
    selection: dict,
    test_result: dict,
    latency: dict,
    audit: dict,
    qualification: dict,
    confirmation_result: dict | None,
) -> None:
    models = root / "models"
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    parity = json.loads(candidate.with_suffix(".parity.json").read_text(encoding="utf-8"))
    report = test_result["report"]
    metrics = test_result["metrics"]
    quality = {
        "accuracy": metrics["raw_known_accuracy"],
        "macro_f1": metrics["raw_known_macro_f1"],
        "runtime_macro_f1": metrics["runtime_macro_f1"],
        "accepted_precision": metrics["accepted_precision"],
        "known_acceptance_rate": 1.0 - metrics["known_rejection_rate"],
        "unknown_false_acceptance_rate": metrics["unknown_false_accept_rate"],
        "minimum_per_class_f1": min(
            float(report[name]["f1-score"]) for name in CLASS_NAMES
        ),
        "evaluation_source": (
            "participant-disjoint public test split; used as a release "
            "regression gate after candidate iteration"
        ),
    }
    confirmation_evidence = (
        None
        if confirmation_result is None
        else {
            "metrics": confirmation_result["metrics"],
            "per_class": {
                name: confirmation_result["report"][name] for name in LABELS
            },
            "participant_count": 2600,
            "participant_overlap_with_train_validation_test": 0,
            "role": (
                "fail-closed deployment and regression gate; excluded from "
                "model fitting and scalar-threshold selection, but an earlier "
                "failure informed later training and geometry revisions"
            ),
        }
    )
    metadata = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "ONNX",
        "model_name": "TenGestureBalancedMLP",
        "onnx_file": "gesture_mlp_production.onnx",
        "onnx_bytes": candidate.stat().st_size,
        "onnx_sha256": digest,
        "onnx_ir_version": 8,
        "opsets": {"ai.onnx": 16},
        "input": {"name": "landmark_features", "dtype": "float32", "shape": [None, 76]},
        "probability_output": "probabilities",
        "known_mass_output": "known_gesture_mass",
        "output_class_order": CLASS_NAMES,
        "internal_class_order": LABELS,
        "source_class_mapping": {
            "left": "one_right",
            "right": "one_left",
            "up": "one",
            "down": "one_down",
            "open_palm": "palm_up_only",
            "like": "like",
            "dorsal": "11K/Dorsal hands",
            "ok": "ok",
            "fist": "fist",
            "thumb_down": "dislike",
        },
        "horizontal_direction_calibration": {
            "camera_pixels_mirrored": False,
            "positive_index_dx_command": "left",
            "negative_index_dx_command": "right",
            "training_labels_ncm_calibrated": True,
        },
        "feature_names": FEATURE_NAMES,
        "parity": parity,
        "quality": quality,
        "qualification": qualification,
        "independent_confirmation": confirmation_evidence,
        "public_evaluation": metrics,
        "training": audit,
        "runtime_selection": selection,
        "classifier_latency": latency,
        "landmark_detector": DETECTOR_SETTINGS,
        "runtime_geometry": {
            "open_palm_upward_only": True,
            "fist_validation": True,
            "foreshortened_camera_facing_fist_recovery": True,
            "thumb_down_validation": True,
            "dorsal_independent_support": True,
            "reviewed_negative_feedback_veto": True,
            "geometry_recovery_mass_floor": 0.15,
        },
        "runtime_note": (
            "Ten exposed command probabilities. no_gesture remains an internal "
            "negative class; known_gesture_mass carries its rejection evidence."
        ),
    }
    config_path = models / "gesture_mobile_runtime_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        schema_version=3,
        model_format="ONNX float32, ten command outputs plus open-set score",
        class_names=CLASS_NAMES,
        feedback_labels=[*CLASS_NAMES, "no_gesture"],
        gesture_to_action=GESTURE_TO_ACTION,
    )
    config["temporal_smoothing"].update(selection)
    config["temporal_smoothing"]["geometry_recovery_mass_floor"] = 0.15
    config["pose_validation"].update(
        fist_maximum_mean_finger_extension=0.44,
        thumb_down_minimum_thumb_extension=0.45,
        thumb_down_maximum_other_finger_extension=0.58,
    )
    config["open_palm_orientation"] = {
        "minimum_up_alignment": 0.70,
        "minimum_axis_dominance": 0.70,
        "reject_left_right_down": True,
    }
    config["qualification"] = {
        "parity": parity,
        "test_metrics": quality,
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "deployment_gate": qualification,
        "independent_confirmation": confirmation_evidence,
        "classifier_latency": latency,
    }
    config["ten_gesture_release"] = {
        "training_examples_per_internal_class": audit["per_class"],
        "internal_classes": INTERNAL_COUNT,
        "classifier_threads": 1,
        "public_test_runtime_macro_f1": metrics["runtime_macro_f1"],
    }

    shutil.copy2(candidate, models / "gesture_mlp_production.onnx")
    (models / "gesture_mlp_onnx_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    _build_online_caches(root, candidate)

    pd.DataFrame(
        [
            {
                "class": name,
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1_score": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in CLASS_NAMES
        ]
    ).to_csv(models / "ten_gesture_classification_report.csv", index=False)
    pd.DataFrame(
        test_result["confusion"], index=LABELS, columns=LABELS
    ).rename_axis("actual").to_csv(models / "ten_gesture_confusion_matrix.csv")
    pd.DataFrame([{"model": "ten_gesture", **quality}]).to_csv(
        models / "ten_gesture_test_metrics.csv", index=False
    )
    pd.DataFrame([_confusion_rates(test_result)]).to_csv(
        models / "ten_gesture_hard_case_metrics.csv", index=False
    )
    ArtifactRegistry(models).build()


def qualify(root: Path, deploy: bool = False) -> dict:
    root = Path(root)
    output = root / ARTIFACT_DIRECTORY
    candidate = output / "candidate.onnx"
    validation = load_npz(output / "public_val.npz")
    test = load_npz(output / "public_test.npz")
    selection, selection_evidence = select_thresholds(candidate, validation)
    selection_record = {
        **selection,
        "source": "participant-disjoint public validation split; test excluded",
        "model_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    }
    (output / "threshold_selection.json").write_text(
        json.dumps(selection_evidence, indent=2) + "\n", encoding="utf-8"
    )
    config = _runtime_config(selection)
    validation_result = _evaluate_split(candidate, validation, config)
    test_result = _evaluate_split(candidate, test, config)
    _write_evidence(output, "public_validation", validation_result)
    _write_evidence(output, "public_test", test_result)
    latency = benchmark(candidate, test["X"])
    passed, gate_details = _quality_gate(test_result, latency)
    confirmation_path = output / "independent_confirmation.npz"
    confirmation_result = None
    confirmation_gate = None
    if confirmation_path.is_file():
        confirmation = load_npz(confirmation_path)
        confirmation_result = _evaluate_split(candidate, confirmation, config)
        _write_evidence(output, "independent_confirmation", confirmation_result)
        confirmation_passed, confirmation_gate = _confirmation_gate(
            confirmation_result
        )
        passed = bool(passed and confirmation_passed)
    qualification = {
        "passed": passed,
        **gate_details,
        "independent_confirmation": confirmation_gate,
        "selection": selection_record,
        "note": (
            "The public test is participant-disjoint within imported public data. "
            "It is evidence, not a guarantee for every person, camera, or angle."
        ),
    }
    (output / "qualification.json").write_text(
        json.dumps(qualification, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "selection": selection_record,
        "validation": validation_result["metrics"],
        "test": test_result["metrics"],
        "test_per_class": {
            name: test_result["report"][name] for name in LABELS
        },
        "independent_confirmation": (
            None
            if confirmation_result is None
            else {
                "metrics": confirmation_result["metrics"],
                "per_class": {
                    name: confirmation_result["report"][name]
                    for name in LABELS
                },
                "gate": confirmation_gate,
            }
        ),
        "latency": latency,
        "qualification": qualification,
    }
    (output / "evaluation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if deploy:
        if not passed:
            raise RuntimeError("Candidate failed at least one ten-gesture deployment gate.")
        audit = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
        _deploy(
            root, candidate, selection, test_result, latency, audit,
            qualification, confirmation_result,
        )
        print("Deployed the qualified ten-command release.", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true")
    arguments = parser.parse_args()
    qualify(Path.cwd(), deploy=arguments.deploy)
