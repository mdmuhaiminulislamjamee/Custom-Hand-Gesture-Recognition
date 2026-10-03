"""Install the evaluated twelve-command ONNX model and matching runtime bundle.

The existing ten-command files are replaced only after candidate integrity and
offline release checks pass. Camera-domain validation remains a separate task.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from backend.artifact_integrity import ArtifactRegistry
from backend.config import CLASS_NAMES, FEATURE_NAMES, FEEDBACK_LABELS, GESTURE_TO_ACTION
from research.v18_20 import predict_onnx
from scripts.train_twelve_gesture_candidate import prepare


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SOURCE = ROOT / "artifacts" / "twelve_gesture"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _cache(path: Path, data: dict) -> None:
    # -1 is the runtime's open-set/feedback sentinel; ONNX has 12 outputs.
    labels = np.where(data["y"] == len(CLASS_NAMES), -1, data["y"])
    payload = dict(
        X=np.asarray(data["X"], dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        class_names=np.asarray(CLASS_NAMES),
    )
    if "landmarks" in data:
        payload["landmarks"] = np.asarray(data["landmarks"], dtype=np.float32)
    np.savez_compressed(
        path,
        **payload,
    )


def main() -> None:
    evaluation = _read(SOURCE / "evaluation.json")
    contract = _read(SOURCE / "candidate_contract.json")
    comparison = _read(SOURCE / "runtime_comparison.json")
    candidate = SOURCE / "candidate.onnx"
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if contract["class_names"] != CLASS_NAMES or contract["gesture_to_action"] != GESTURE_TO_ACTION:
        raise ValueError("Candidate class/action contract is not the 12-command runtime contract.")
    if digest != contract["model_sha256"] or digest != evaluation["artifact"]["sha256"]:
        raise ValueError("Candidate model digest does not match evaluated artifact.")
    parity = evaluation["parity"]
    test = evaluation["test"]
    checks = {
        "onnx_parity": parity["label_agreement"] >= .999 and parity["max_probability_error"] < 1e-4,
        "offline_macro_f1_at_least_0_97": test["macro_f1"] >= .97,
        "offline_accepted_precision_at_least_0_99": test["accepted_precision"] >= .99,
        "offline_unknown_false_accept_at_most_0_02": test["unknown_false_accept_rate"] <= .02,
        "old_command_correct_accept_at_least_0_98": test["old_command_correct_accept_rate"] >= .98,
        "new_command_recall_at_least_0_97": all(value >= .97 for value in test["new_command_recall"].values()),
        "runtime_peace_recall_at_least_0_95": comparison["test"]["candidate"]["new_recall"]["peace"] >= .95,
        "runtime_rock_recall_at_least_0_97": comparison["test"]["candidate"]["new_recall"]["rock"] >= .97,
        "runtime_unknown_false_accept_at_most_0_02": comparison["test"]["candidate"]["true_negative_false_accept"] <= .02,
    }
    if not all(checks.values()):
        raise ValueError(f"Offline release checks failed: {checks}")

    train, validation, untouched_test, audit = prepare()
    if audit != evaluation["audit"]:
        raise ValueError("Prepared data audit does not reproduce candidate evaluation.")
    retired_mask = untouched_test["variant"] == "source_call"
    if not np.any(retired_mask):
        raise ValueError("Untouched test has no retired-call negative examples.")
    retired_scores, retired_mass = predict_onnx(candidate, untouched_test["X"][retired_mask])
    ordered_retired = np.sort(retired_scores, axis=1)
    retired_false_accept = float(np.mean(
        (retired_mass >= .70)
        & (ordered_retired[:, -1] >= .70)
        & (ordered_retired[:, -1] - ordered_retired[:, -2] >= .08)
    ))
    evaluation["test"]["retired_call_negative_count"] = int(retired_mask.sum())
    evaluation["test"]["retired_call_false_accept_rate"] = retired_false_accept
    checks["retired_call_false_accept_at_most_0_02"] = retired_false_accept <= .02
    if not checks["retired_call_false_accept_at_most_0_02"]:
        raise ValueError(f"Retired call sign false acceptance exceeds 0.02: {retired_false_accept}")
    if train["X"].shape[1] != len(FEATURE_NAMES):
        raise ValueError("Prepared replay features are not 76-D.")
    # The replay cache is deliberately small and class-balanced; the untouched
    # cache is participant-separated from training and is never used for fitting.
    replay_indices = np.concatenate([
        np.flatnonzero(train["y"] == index)[:128]
        for index in range(len(FEEDBACK_LABELS))
    ])
    _cache(MODELS / "gesture_online_replay_cache.npz", {key: value[replay_indices] for key, value in train.items()})
    _cache(MODELS / "gesture_online_validation_cache.npz", validation)
    _cache(MODELS / "gesture_online_untouched_test_cache.npz", untouched_test)

    destination = MODELS / "gesture_mlp_production.onnx"
    shutil.copyfile(candidate, destination)
    config = _read(MODELS / "gesture_mobile_runtime_config.json")
    config.update({
        "model_format": "ONNX float32, twelve command outputs plus open-set score",
        "class_names": CLASS_NAMES,
        "feedback_labels": FEEDBACK_LABELS,
        "gesture_to_action": GESTURE_TO_ACTION,
    })
    config.pop("ten_gesture_release", None)
    config.pop("qualification", None)
    config.pop("thirteen_gesture_release", None)
    config["twelve_gesture_release"] = {
        "model_sha256": digest,
        "source": "artifacts/twelve_gesture/evaluation.json",
        "offline_checks": checks,
        "live_camera_validation_pending": True,
        "limitations": audit["limitations"],
    }
    _write(MODELS / "gesture_mobile_runtime_config.json", config)

    metadata = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "ONNX",
        "model_name": "TwelveGestureBalancedMLP",
        "onnx_file": destination.name,
        "onnx_bytes": destination.stat().st_size,
        "onnx_sha256": digest,
        "input": {"name": "landmark_features", "dtype": "float32", "shape": [None, len(FEATURE_NAMES)]},
        "probability_output": "probabilities",
        "known_mass_output": "known_gesture_mass",
        "output_class_order": CLASS_NAMES,
        "internal_class_order": FEEDBACK_LABELS,
        "feature_names": FEATURE_NAMES,
        "source_class_mapping": {
            "left": "one_right", "right": "one_left", "up": "one",
            "down": "one_down", "open_palm": "palm_up_only", "like": "like",
            "dorsal": "11K/Dorsal hands", "ok": "ok", "fist": "fist",
            "thumb_down": "dislike", **contract["source_class_mapping"],
        },
        "horizontal_direction_calibration": {
            "camera_pixels_mirrored": False,
            "positive_index_dx_command": "left",
            "negative_index_dx_command": "right",
            "training_labels_ncm_calibrated": True,
        },
        "parity": {
            "passed": checks["onnx_parity"],
            "prediction_agreement": parity["label_agreement"],
            "maximum_absolute_probability_error": parity["max_probability_error"],
            "maximum_absolute_mass_error": parity["max_mass_error"],
        },
        "quality": {
            "macro_f1": test["macro_f1"],
            "accepted_precision": test["accepted_precision"],
            "unknown_false_acceptance_rate": test["unknown_false_accept_rate"],
            "old_command_correct_accept_rate": test["old_command_correct_accept_rate"],
            "new_command_recall": test["new_command_recall"],
            "retired_call_false_accept_rate": retired_false_accept,
        },
        "qualification": {
            "offline_passed": True,
            "live_camera_validation_pending": True,
            "checks": checks,
            "selection": {
                "model_sha256": digest,
                "known_mass_floor": config["temporal_smoothing"]["known_mass_floor"],
                "confidence_floor": config["temporal_smoothing"]["confidence_floor"],
                "probability_margin_floor": config["temporal_smoothing"]["probability_margin_floor"],
            },
            "data_audit": audit,
            "comparison_to_previous_ten_command_model": {
                "validation": {key: comparison["validation"][key] for key in ("production", "candidate")},
                "test": {key: comparison["test"][key] for key in ("production", "candidate")},
            },
        },
        "runtime_note": "Twelve-command offline-qualified bundle; live camera validation remains pending.",
    }
    _write(MODELS / "gesture_mlp_onnx_metadata.json", metadata)
    manifest = ArtifactRegistry(MODELS).build()
    contract["deployment_status"] = "active twelve-command ONNX runtime; live-camera validation pending"
    evaluation["deployed"] = True
    _write(SOURCE / "candidate_contract.json", contract)
    _write(SOURCE / "evaluation.json", evaluation)
    print(f"Installed {len(CLASS_NAMES)}-command model {digest}; manifest {manifest['release_fingerprint']}")


if __name__ == "__main__":
    main()
