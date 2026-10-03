"""Train a twelve-command model with the retired call sign as no_gesture.

Peace and Rock were hard negatives in the ten-command release. HaGRID's Call
stays in the rejection class. This script holds out people before augmentation,
and never changes the production model.  The held-out new-command examples all
come from HaGRID's published *training* partition; they are not an independent
camera-domain qualification set.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from backend.config import CLASS_NAMES, GESTURE_TO_ACTION
from backend.geometry import landmarks_to_feature
from research.v18_20 import load_npz, predict_onnx


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "twelve_gesture"
# The ten-command source archive predates this deployment; its rejection
# sentinel is index 10 even after the runtime taxonomy becomes twelve-way.
BASE_COMMANDS = CLASS_NAMES[:10]
COMMANDS = [*BASE_COMMANDS, "peace", "rock"]
REJECT = len(COMMANDS)
LABELS = [*COMMANDS, "no_gesture"]
SOURCE_LABELS = {"source_peace": "peace", "source_rock": "rock"}
ACTIONS = GESTURE_TO_ACTION.copy()


def _partition(group: str) -> str:
    value = int.from_bytes(hashlib.sha256(("12-command-v1:" + group).encode()).digest()[:8], "big") % 100
    return "train" if value < 70 else "val" if value < 85 else "test"


def _relabel(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {key: value.copy() for key, value in data.items()}
    result["y"][result["y"] == len(BASE_COMMANDS)] = REJECT
    for variant, name in SOURCE_LABELS.items():
        result["y"][result["variant"] == variant] = COMMANDS.index(name)
    return result


def _subset(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in data.items()}


def _join(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    common = set.intersection(*(set(part) for part in parts))
    return {key: np.concatenate([part[key] for part in parts]) for key in common}


def prepare(seed: int = 1302, per_class: int = 4096) -> tuple[dict, dict, dict, dict]:
    from research.v18_20 import feature_keys

    source = _relabel(load_npz(ROOT / "data/ten_gesture/hagrid_train.npz"))
    original_val = _relabel(load_npz(ROOT / "artifacts/ten_gesture/public_val.npz"))
    original_test = _relabel(load_npz(ROOT / "artifacts/ten_gesture/public_test.npz"))
    # The older evaluation archives collapsed 100 Peace + 100 Call + 100
    # Rock + 100 true negatives into one unlabeled rejection bucket per split.
    # Their source category was not retained, so none of those 400 rows can be
    # safely relabeled. Exclude all, rather than call real commands negatives.
    ambiguous_removed = {}
    for split, data in (("val", original_val), ("test", original_test)):
        ambiguous_removed[split] = int(np.sum(data["y"] == REJECT))
    original_val = _subset(original_val, original_val["y"] != REJECT)
    original_test = _subset(original_test, original_test["y"] != REJECT)
    # The publisher's val/test people stay in those original splits. The new
    # shapes only exist locally in its training partition, so reserve whole
    # participants there for the candidate's validation and test sets.
    external_groups = set(original_val["group"]) | set(original_test["group"])
    eligible = ~np.isin(source["group"], list(external_groups))
    source = _subset(source, eligible)
    assignments = np.asarray([_partition(str(group)) for group in source["group"]])
    groups = {
        split: _subset(source, assignments == split)
        for split in ("train", "val", "test")
    }
    # The 11K dorsal source has real participant metadata. Only the people
    # absent from original evaluation can enter training.
    import pandas as pd

    dorsal = load_npz(ROOT / "data/ten_gesture/dorsal.npz")
    metadata = pd.read_csv(ROOT / "data/ten_gesture/HandInfo.csv", dtype={"id": str}).set_index("imageName")
    dorsal["group"] = np.asarray(["11k:" + metadata.loc[name, "id"] for name in dorsal["source_id"]])
    dorsal = _relabel(dorsal)
    held_dorsal = set(original_val["group"]) | set(original_test["group"])
    dorsal_train = _subset(dorsal, ~np.isin(dorsal["group"], list(held_dorsal)))
    groups["train"] = _join([groups["train"], dorsal_train])
    groups["val"] = _join([groups["val"], original_val])
    groups["test"] = _join([groups["test"], original_test])

    # Test > validation > training precedence for both people and exact poses.
    seen_groups: set[str] = set()
    seen_features: set[str] = set()
    removed: dict[str, int] = {}
    for split in ("test", "val", "train"):
        data = groups[split]
        keys = feature_keys(data["X"])
        local: set[str] = set()
        keep = []
        for i, key in enumerate(keys):
            if str(data["group"][i]) in seen_groups or key in seen_features or key in local:
                continue
            keep.append(i)
            local.add(key)
        removed[split] = len(keys) - len(keep)
        groups[split] = _subset(data, np.asarray(keep, dtype=int))
        seen_groups.update(map(str, groups[split]["group"]))
        seen_features.update(local)

    rng = np.random.default_rng(seed)
    rows, labels, parents = [], [], []
    training = groups["train"]
    for label, name in enumerate(LABELS):
        available = np.flatnonzero(training["y"] == label)
        if not len(available):
            raise ValueError(f"No training data for {name}")
        chosen = np.resize(rng.permutation(available), per_class)
        for number, index in enumerate(chosen):
            points = training["landmarks"][index].copy()
            if number >= len(available):
                points -= points[0]
                # New poses intentionally cover the full in-plane compass. A
                # rotated 2-D example is not a new camera viewpoint/person.
                angle = (rng.uniform(-np.pi, np.pi) if name in SOURCE_LABELS.values()
                         else rng.uniform(-.25, .25))
                cosine, sine = np.cos(angle), np.sin(angle)
                points = points @ np.asarray([[cosine, -sine], [sine, cosine]]).T
                points *= rng.uniform(.90, 1.10, size=(1, 2))
                scale = np.linalg.norm(points[9] - points[0])
                points += rng.normal(0, max(scale, 1e-6) * .006, points.shape)
            rows.append(landmarks_to_feature(points))
            labels.append(label)
            parents.append(str(training["source_id"][index]))
    order = rng.permutation(len(rows))
    balanced = {
        "X": np.asarray(rows, np.float32)[order],
        "y": np.asarray(labels, np.int64)[order],
        "parent_id": np.asarray(parents)[order],
    }
    audit = {
        "seed": seed,
        "per_class": per_class,
        "labels": LABELS,
        "counts": {split: {name: int(np.sum(data["y"] == index)) for index, name in enumerate(LABELS)}
                   for split, data in groups.items()},
        "subjects": {split: len(set(data["group"])) for split, data in groups.items()},
        "removed_overlap": removed,
        "ambiguous_legacy_negatives_excluded": ambiguous_removed,
        "limitations": [
            "New-command validation/test examples are participant-held-out rows from HaGRID's training partition.",
            "Legacy evaluation negatives mixed the new valid commands with true negatives without source labels; all were excluded.",
            "In-plane rotation is synthetic; it cannot prove all camera viewing angles, distances, lighting, or occlusions.",
            "No new live low-resolution or low-FPS camera capture is available for qualification.",
        ],
    }
    return balanced, groups["val"], groups["test"], audit


def export_onnx(model: MLPClassifier, scaler: StandardScaler, destination: Path) -> dict:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    initializers = []
    nodes = []

    def constant(name: str, value: np.ndarray) -> None:
        initializers.append(numpy_helper.from_array(np.asarray(value), name))

    constant("mean", scaler.mean_.astype(np.float32))
    constant("scale", scaler.scale_.astype(np.float32))
    nodes.extend([
        helper.make_node("Sub", ["landmark_features", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "scale"], ["scaled"]),
    ])
    current = "scaled"
    for i, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
        constant(f"w{i}", weights.astype(np.float32))
        constant(f"b{i}", bias.astype(np.float32))
        nodes.extend([
            helper.make_node("MatMul", [current, f"w{i}"], [f"mat{i}"]),
            helper.make_node("Add", [f"mat{i}", f"b{i}"], [f"add{i}"]),
        ])
        current = f"add{i}"
        if i < len(model.coefs_) - 1:
            nodes.append(helper.make_node("Relu", [current], [f"act{i}"]))
            current = f"act{i}"
    nodes.append(helper.make_node("Softmax", [current], ["all_probabilities"], axis=1))
    constant("known_indices", np.arange(len(COMMANDS), dtype=np.int64))
    constant("axis", np.asarray([1], dtype=np.int64))
    constant("epsilon", np.asarray([1e-30], dtype=np.float32))
    nodes.extend([
        helper.make_node("Gather", ["all_probabilities", "known_indices"], ["known"], axis=1),
        helper.make_node("ReduceSum", ["known", "axis"], ["known_gesture_mass"], keepdims=1),
        helper.make_node("Add", ["known", "epsilon"], ["safe_known"]),
        helper.make_node("ReduceSum", ["safe_known", "axis"], ["denominator"], keepdims=1),
        helper.make_node("Div", ["safe_known", "denominator"], ["probabilities"]),
        helper.make_node("ArgMax", ["probabilities"], ["label"], axis=1, keepdims=0),
    ])
    graph = helper.make_graph(
        nodes, "twelve_gesture_candidate",
        [helper.make_tensor_value_info("landmark_features", TensorProto.FLOAT, [None, 76])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
         helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, len(COMMANDS)]),
         helper.make_tensor_value_info("known_gesture_mass", TensorProto.FLOAT, [None, 1])],
        initializers,
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=8)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, destination)
    return {"bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def _score(model_path: Path, data: dict, mass_floor: float = .70) -> dict:
    probabilities, mass = predict_onnx(model_path, data["X"])
    raw = probabilities.argmax(1)
    ordered = np.sort(probabilities, axis=1)
    accepted = ((mass >= mass_floor) & (ordered[:, -1] >= .70)
                & ((ordered[:, -1] - ordered[:, -2]) >= .08))
    predicted = np.where(accepted, raw, REJECT)
    truth = data["y"]
    report = classification_report(truth, predicted, labels=np.arange(len(LABELS)),
                                   target_names=LABELS, output_dict=True, zero_division=0)
    old = truth < len(BASE_COMMANDS)
    unknown = truth == REJECT
    return {
        "raw_old_command_accuracy": float(np.mean(raw[old] == truth[old])),
        "old_command_correct_accept_rate": float(np.mean(predicted[old] == truth[old])),
        "new_command_recall": {name: float(report[name]["recall"]) for name in SOURCE_LABELS.values()},
        "new_command_precision": {name: float(report[name]["precision"]) for name in SOURCE_LABELS.values()},
        "unknown_false_accept_rate": float(np.mean(accepted[unknown])),
        "accepted_precision": float(np.mean(predicted[accepted] == truth[accepted])),
        "macro_f1": float(f1_score(truth, predicted, labels=np.arange(len(LABELS)), average="macro")),
        "per_class": report,
        "confusion": confusion_matrix(truth, predicted, labels=np.arange(len(LABELS))).tolist(),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, validation, test, audit = prepare()
    scaler = StandardScaler().fit(train["X"])
    train_x = scaler.transform(train["X"]).astype(np.float32)
    val_x = scaler.transform(validation["X"]).astype(np.float32)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=.003, batch_size=256,
                          learning_rate_init=.0007, random_state=1302, max_iter=1)
    best, best_f1, stale, history = None, -1., 0, []
    with threadpool_limits(limits=1):
        for epoch in range(80):
            model.partial_fit(train_x, train["y"], classes=np.arange(len(LABELS)))
            f1 = float(f1_score(validation["y"], model.predict(val_x),
                                labels=np.arange(len(LABELS)), average="macro", zero_division=0))
            history.append({"epoch": epoch + 1, "val_raw_macro_f1": f1})
            if f1 > best_f1 + 1e-4:
                best, best_f1, stale = copy.deepcopy(model), f1, 0
            else:
                stale += 1
            if epoch % 5 == 0:
                print(f"epoch {epoch + 1}: validation raw macro F1={f1:.4f}", flush=True)
            if stale >= 15:
                break
    candidate = OUTPUT / "candidate.onnx"
    artifact = export_onnx(best, scaler, candidate)
    expected = best.predict_proba(scaler.transform(validation["X"]).astype(np.float32))
    actual, mass = predict_onnx(candidate, validation["X"])
    expected_mass = expected[:, :len(COMMANDS)].sum(1)
    expected_conditional = expected[:, :len(COMMANDS)] / expected_mass[:, None]
    parity = {
        "label_agreement": float(np.mean(expected_conditional.argmax(1) == actual.argmax(1))),
        "max_probability_error": float(np.max(np.abs(expected_conditional - actual))),
        "max_mass_error": float(np.max(np.abs(expected_mass - mass))),
    }
    evaluation = {
        "artifact": artifact,
        "parity": parity,
        "audit": audit,
        "history": history,
        "validation": _score(candidate, validation),
        "test": _score(candidate, test),
        "deployed": False,
    }
    (OUTPUT / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    (OUTPUT / "candidate_contract.json").write_text(json.dumps({
        "class_names": COMMANDS,
        "internal_class_names": LABELS,
        "gesture_to_action": ACTIONS,
        "source_class_mapping": {"peace": "peace", "rock": "rock"},
        "feature_count": 76,
        "model_file": "candidate.onnx",
        "model_sha256": artifact["sha256"],
        "deployment_status": "candidate_only; production model unchanged",
    }, indent=2) + "\n")
    print(json.dumps({key: evaluation[key] for key in ("artifact", "parity", "audit")}, indent=2))
    print("validation:", {key: value for key, value in evaluation["validation"].items() if key not in {"per_class", "confusion"}})
    print("test:", {key: value for key, value in evaluation["test"].items() if key not in {"per_class", "confusion"}})


if __name__ == "__main__":
    main()
