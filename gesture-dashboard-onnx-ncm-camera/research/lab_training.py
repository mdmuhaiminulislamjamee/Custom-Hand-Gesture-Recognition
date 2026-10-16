"""Twelve-command candidate training from reviewed dashboard captures.

The image classifier receives only hand features. Context/detector captures are
retained by the dataset reader but excluded from fitting. All selection uses
validation people; test scores are produced by a separate evaluation job.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from backend.config import CLASS_NAMES, RuntimeConfig
from backend.data_collection import atomic_json
from research.collection_dataset import load_collection_dataset

LABELS = [*CLASS_NAMES, "no_gesture"]


def audit_dataset(root: Path):
    if not (root / "dataset.json").is_file():
        raise ValueError("Collect and review photos or extract reviewed video frames first.")
    data, ingestion = load_collection_dataset(root)
    coverage = {}
    for split in ("train", "val", "test"):
        mask = data["split"] == split
        coverage[split] = {
            "rows": int(mask.sum()),
            "participants": len(set(data["participant_id"][mask])),
            "classes": {label: int(np.sum(mask & (data["y"] == i)))
                        for i, label in enumerate(LABELS)},
        }
    issues = []
    from research.v18_20 import feature_keys
    split_keys = {split: set(feature_keys(data["X"][data["split"] == split]))
                  for split in ("train", "val", "test")}
    overlaps = {f"{left}/{right}": len(split_keys[left] & split_keys[right])
                for left, right in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlaps.values()):
        issues.append("Identical hand features occur in different participant splits; review duplicate captures before training.")
    for split in ("train", "val", "test"):
        if coverage[split]["participants"] < 2:
            issues.append(f"{split}: fewer than two independent participants with usable hand features")
        missing = [label for label, count in coverage[split]["classes"].items() if count == 0]
        if missing:
            issues.append(f"{split}: missing classes: {', '.join(missing)}")
    result = {
        "class_names": LABELS, "ingestion": ingestion, "coverage": coverage,
        "participants": {str(person): {"split": str(data["split"][data["participant_id"] == person][0]),
                                       "rows": int(np.sum(data["participant_id"] == person))}
                         for person in sorted(set(data["participant_id"]))},
        "dataset_fingerprint": hashlib.sha256("\n".join(sorted(
            f"{person}:{sample}:{split}:{label}:{feature}" for person, sample, split, label, feature in
            zip(data["participant_id"], data["sample_id"], data["split"], data["y"],
                feature_keys(data["X"])))).encode()).hexdigest(),
        "coverage_passed": not issues, "issues": issues,
        "cross_split_feature_overlap": overlaps,
        "can_train": coverage["train"]["rows"] > 0 and coverage["val"]["rows"] > 0 and not any(overlaps.values()),
        "scope": "Single-frame hand classifier; relaxed posture remains a runtime body check.",
        "split_policy": "Participant-isolated; all sessions and video-derived frames stay together.",
    }
    return data, result


def mix_training(base, captured, per_class=1024, seed=1900):
    """Keep every class, cap captured share and repeats, then shuffle."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    counts = {}
    for index, label in enumerate(LABELS):
        public = np.flatnonzero(base["y"] == index)
        local = np.flatnonzero(captured["y"] == index)
        if not len(public):
            raise ValueError(f"Base training archive lacks {label}")
        take_local = min(round(per_class * .25), len(local) * 4)
        public_ids = rng.choice(public, per_class - take_local,
                                replace=len(public) < per_class - take_local)
        rows.extend(base["X"][public_ids])
        labels.extend([index] * len(public_ids))
        if take_local:
            local_ids = np.resize(rng.permutation(local), take_local)
            rows.extend(captured["X"][local_ids])
            labels.extend([index] * take_local)
        counts[label] = {"public_rows": len(public_ids), "captured_rows": take_local,
                         "unique_captured_rows": len(local)}
    order = rng.permutation(len(rows))
    return {"X": np.asarray(rows, np.float32)[order],
            "y": np.asarray(labels, np.int64)[order]}, counts


def check_source_splits(public, captured):
    """Refuse copied/duplicate public frames crossing camera split boundaries."""
    from research.v18_20 import feature_keys
    public_keys = {split: set(feature_keys(data["X"])) for split, data in public.items()}
    for split, data in captured.items():
        captured_keys = set(feature_keys(data["X"]))
        for other, keys in public_keys.items():
            if split != other and captured_keys & keys:
                raise ValueError(f"Captured {split} features overlap public {other} data. Remove copied/duplicate examples to preserve an independent evaluation.")


def score_model(path: Path, data, config=None):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from research.v18_20 import predict_onnx

    if not len(data["y"]):
        return {"rows": 0, "available": False}
    config = config or RuntimeConfig()
    probabilities, mass = predict_onnx(path, data["X"])
    if probabilities.shape != (len(data["y"]), len(CLASS_NAMES)):
        raise ValueError("Candidate output does not match the twelve-command contract")
    ordered = np.sort(probabilities, axis=1)
    accepted = ((mass >= config.known_mass_floor)
                & (ordered[:, -1] >= config.confidence_floor)
                & (ordered[:, -1] - ordered[:, -2] >= config.probability_margin_floor))
    predicted = np.where(accepted, probabilities.argmax(1), len(CLASS_NAMES))
    truth = data["y"]
    negative = truth == len(CLASS_NAMES)
    report = classification_report(truth, predicted, labels=np.arange(len(LABELS)),
                                   target_names=LABELS, output_dict=True, zero_division=0)
    return {
        "rows": len(truth), "available": True,
        "macro_f1": float(f1_score(truth, predicted, labels=np.arange(len(LABELS)),
                                   average="macro", zero_division=0)),
        "accepted_precision": float(np.mean(predicted[accepted] == truth[accepted])) if accepted.any() else None,
        "unknown_false_accept_rate": float(np.mean(accepted[negative])) if negative.any() else None,
        "per_class": {name: report[name] for name in LABELS},
        "confusion": confusion_matrix(truth, predicted, labels=np.arange(len(LABELS))).tolist(),
        "class_names": LABELS,
        "scope": "Landmark classifier thresholds only; run Video Lab for body/geometry/actions.",
    }


def train(root: Path, output: Path, project: Path, epochs=50, log: Callable = print):
    from sklearn.metrics import f1_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from scripts.train_twelve_gesture_candidate import prepare, export_onnx
    from research.v18_20 import predict_onnx

    captured, audit = audit_dataset(root)
    atomic_json(output / "audit.json", audit)
    if not audit["can_train"]:
        raise ValueError("Training needs reviewed usable frames from independent training and validation participants. Run the dataset audit for details.")
    log("Preparing participant-isolated public data for all twelve commands and rejection.", flush=True)
    base, public_val, public_test, base_audit = prepare(per_class=1024)
    subsets = {split: {"X": captured["X"][captured["split"] == split],
                       "y": captured["y"][captured["split"] == split]}
               for split in ("train", "val", "test")}
    check_source_splits({"train": base, "val": public_val, "test": public_test}, subsets)
    mixed, balance = mix_training(base, subsets["train"])
    val = {key: np.concatenate([public_val[key], subsets["val"][key]]) for key in ("X", "y")}
    # The test snapshot is immutable and is evaluated only by an explicit job.
    np.savez_compressed(output / "public_test.npz", X=public_test["X"], y=public_test["y"])
    np.savez_compressed(output / "camera_test.npz", **subsets["test"])
    np.savez_compressed(output / "validation.npz", **val)
    scaler = StandardScaler().fit(mixed["X"])
    x = scaler.transform(mixed["X"]).astype(np.float32)
    validation_x = scaler.transform(val["X"]).astype(np.float32)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=.003, batch_size=min(256, len(x)),
                          learning_rate_init=.0007, random_state=1900, max_iter=1)
    best, best_score, stale, history = None, -1., 0, []
    with threadpool_limits(limits=1):
        for epoch in range(epochs):
            model.partial_fit(x, mixed["y"], classes=np.arange(len(LABELS)))
            f1 = float(f1_score(val["y"], model.predict(validation_x),
                                labels=np.arange(len(LABELS)), average="macro", zero_division=0))
            history.append({"epoch": epoch + 1, "validation_raw_macro_f1": f1})
            log(f"Epoch {epoch + 1}/{epochs}: validation macro F1 {f1:.5f}", flush=True)
            if f1 > best_score + .0001:
                best, best_score, stale = copy.deepcopy(model), f1, 0
            else:
                stale += 1
            if stale >= 15:
                log("Validation improvement stopped; keeping the best epoch.", flush=True)
                break
    candidate = output / "candidate.onnx"
    artifact = export_onnx(best, scaler, candidate)
    expected = best.predict_proba(validation_x)
    actual, mass = predict_onnx(candidate, val["X"])
    expected_mass = expected[:, :len(CLASS_NAMES)].sum(1)
    expected_known = expected[:, :len(CLASS_NAMES)] / expected_mass[:, None]
    parity = {"label_agreement": float(np.mean(expected_known.argmax(1) == actual.argmax(1))),
              "max_probability_error": float(np.max(np.abs(expected_known - actual))),
              "max_mass_error": float(np.max(np.abs(expected_mass - mass)))}
    if parity["label_agreement"] < 1 or max(parity["max_probability_error"], parity["max_mass_error"]) > 1e-4:
        raise ValueError("Candidate ONNX export failed numerical parity checks")
    atomic_json(output / "contract.json", {"class_names": CLASS_NAMES, "feature_count": 76,
                "sha256": artifact["sha256"], "feature_version": "xy_radius_screen_direction_plus_hand_shape_v6"})
    from backend.config import FEATURE_NAMES
    atomic_json(output / "candidate_metadata.json", {
        "format": "ONNX", "model_name": "Reviewed camera candidate",
        "output_class_order": CLASS_NAMES, "feature_names": FEATURE_NAMES,
        "onnx_sha256": artifact["sha256"], "input": {"name": "landmark_features"},
        "known_mass_output": "known_gesture_mass", "parity": {"passed": True, **parity},
    })
    production = project / "models/gesture_mlp_production.onnx"
    config = RuntimeConfig()
    result = {
        "kind": "train", "artifact": artifact, "parity": parity,
        "validation": {"production": score_model(production, val, config),
                       "candidate": score_model(candidate, val, config)},
        "camera_validation": {"production": score_model(production, subsets["val"], config),
                              "candidate": score_model(candidate, subsets["val"], config)},
        "audit": audit, "history": history, "training_balance": balance, "base_audit": base_audit,
        "exploratory": not audit["coverage_passed"], "test_evaluated": False,
        "production_model_changed": False,
        "next_step": "Evaluate the frozen test snapshot, then replay independently labeled camera clips.",
    }
    atomic_json(output / "result.json", result)
    return result


def evaluate(candidate_dir: Path, output: Path, project: Path):
    contract = json.loads((candidate_dir / "contract.json").read_text(encoding="utf-8"))
    candidate = candidate_dir / "candidate.onnx"
    if contract.get("class_names") != CLASS_NAMES or hashlib.sha256(candidate.read_bytes()).hexdigest() != contract.get("sha256"):
        raise ValueError("Candidate artifact failed contract/checksum validation")
    result = {"kind": "evaluate", "production_model_changed": False,
              "candidate_job_id": candidate_dir.name, "class_names": LABELS}
    for filename, key in (("public_test.npz", "public_test"), ("camera_test.npz", "camera_test")):
        with np.load(candidate_dir / filename, allow_pickle=False) as archive:
            data = {"X": archive["X"], "y": archive["y"]}
        result[key] = {"production": score_model(project / "models/gesture_mlp_production.onnx", data),
                       "candidate": score_model(candidate, data)}
    result["live_camera_qualification_pending"] = True
    atomic_json(output / "result.json", result)
    return result
