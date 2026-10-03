"""Evaluate the twelve-command graph against the archived baseline."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import confusion_matrix

from backend.config import RuntimeConfig
from backend.geometry import GeometryResolver, landmarks_to_feature
from research.v18_20 import predict_onnx
from scripts.train_twelve_gesture_candidate import COMMANDS, LABELS, OUTPUT, ROOT, prepare


def decide(model_path, data, config):
    probabilities, mass = predict_onnx(model_path, data["X"])
    resolver = GeometryResolver(config)
    predicted = np.full(len(data["y"]), len(config.class_names), dtype=np.int64)
    raw = probabilities.argmax(1)
    reclassified = []
    for index, (row, points) in enumerate(zip(probabilities, data["landmarks"])):
        resolved, details = resolver.resolve(row, points)
        pose = details["pose_validation"]
        direction = details["directional"] or {}
        top = int(resolved.argmax())
        name = config.class_names[top]
        recovery_mass = mass[index] >= config.geometry_recovery_mass_floor
        geometry = bool(pose.get("geometry_supported", False))
        directional_recovery = bool(geometry and direction.get("strong_geometry", False)
                                    and direction.get("model_supported_pose", False))
        support_allowed = bool(
            (name == "dorsal" and geometry and recovery_mass)
            or (name in {"left", "right", "up", "down"} and directional_recovery
                and (name == "down" or recovery_mass))
            or (name in {"open_palm", "like", "fist", "thumb_down"} and geometry and recovery_mass)
        )
        ordered = np.sort(resolved)
        accept = bool(pose["valid"] and (mass[index] >= config.known_mass_floor or support_allowed)
                      and ordered[-1] >= config.confidence_floor
                      and ordered[-1] - ordered[-2] >= config.probability_margin_floor)
        if accept:
            predicted[index] = top
        if top != raw[index]:
            reclassified.append((int(data["y"][index]), int(raw[index]), top))
    return predicted, raw, mass, reclassified


def summary(data, predicted, names):
    truth = data["y"]
    rejection = len(names)
    old = truth < 10
    negative = truth == rejection
    result = {
        "old_correct_accept": float(np.mean(predicted[old] == truth[old])),
        "old_per_class_recall": {
            name: float(np.mean(predicted[truth == index] == index))
            for index, name in enumerate(names[:10])
        },
        "true_negative_false_accept": float(np.mean(predicted[negative] != rejection)),
        "new_recall": {name: float(np.mean(predicted[truth == names.index(name)] == names.index(name)))
                       for name in ("peace", "rock") if name in names},
        "confusion": confusion_matrix(truth, predicted, labels=np.arange(rejection + 1)).tolist(),
    }
    return result


def main():
    _, validation, test, _ = prepare()
    legacy_raw = json.loads((ROOT / "models/gesture_mobile_runtime_config.json").read_text())
    legacy_classes = legacy_raw["class_names"]
    existing = RuntimeConfig(class_names=legacy_classes, raw=legacy_raw)
    new_config = RuntimeConfig(class_names=COMMANDS, raw=legacy_raw)
    results = {}
    for split, data in (("validation", validation), ("test", test)):
        old_data = {key: value.copy() for key, value in data.items()}
        old_data["y"] = np.where(data["y"] == COMMANDS.index("rock"),
                                 legacy_classes.index("rock"), data["y"])
        old_data["y"] = np.where(data["y"] == len(COMMANDS), len(legacy_classes), old_data["y"])
        original, _, _, _ = decide(ROOT / "models/gesture_mlp_production.onnx", old_data, existing)
        old_mask = data["y"] < 10
        negative_mask = data["y"] == len(COMMANDS)
        baseline = {
            "old_correct_accept": float(np.mean(original[old_mask] == data["y"][old_mask])),
            "old_per_class_recall": {
                name: float(np.mean(original[data["y"] == index] == index))
                for index, name in enumerate(existing.class_names[:10])
            },
            "true_negative_false_accept": float(np.mean(original[negative_mask] != len(existing.class_names))),
        }
        candidate, raw, _, changed = decide(OUTPUT / "candidate.onnx", data, new_config)
        results[split] = {
            "production": baseline,
            "candidate": summary(data, candidate, COMMANDS),
            "candidate_raw_to_resolved": {
                "peace_to_direction": int(sum(data["y"][i] == COMMANDS.index("peace")
                                              and raw[i] == COMMANDS.index("peace")
                                              and candidate[i] < 4 for i in range(len(raw)))),
                "reclassified_total": len(changed),
            },
        }
        if split == "test":
            rock = data["y"] == COMMANDS.index("rock")
            thumb = data["X"][:, 68]
            results[split]["candidate"]["rock_thumb_recall"] = {
                name: {"count": int(mask.sum()),
                       "recall": float(np.mean(candidate[mask] == COMMANDS.index("rock"))) if mask.any() else None}
                for name, mask in {
                    "folded_below_0.4": rock & (thumb < .4),
                    "middle_0.4_to_0.7": rock & (thumb >= .4) & (thumb < .7),
                    "extended_at_least_0.7": rock & (thumb >= .7),
                }.items()
            }
            results[split]["candidate"]["in_plane_quadrant_recall"] = {}
            vectors = data["landmarks"][:, 8] - data["landmarks"][:, 5]
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            quadrants = np.floor((angles + np.pi) / (np.pi / 2)).astype(int).clip(0, 3)
            for name in ("peace", "rock"):
                label = COMMANDS.index(name)
                results[split]["candidate"]["in_plane_quadrant_recall"][name] = {
                    str(quadrant): {
                        "count": int(mask.sum()),
                        "recall": float(np.mean(candidate[mask] == label)) if mask.any() else None,
                    }
                    for quadrant in range(4)
                    for mask in [((data["y"] == label) & (quadrants == quadrant))]
                }
            new_only = (data["y"] >= 10) & (data["y"] < len(COMMANDS))
            original_points = data["landmarks"][new_only]
            results[split]["candidate"]["synthetic_rotation_recall"] = {}
            for degrees in (0, 90, 180, 270):
                angle = np.deg2rad(degrees)
                rotation = np.asarray([[np.cos(angle), -np.sin(angle)],
                                       [np.sin(angle), np.cos(angle)]], dtype=np.float32)
                rotated = (original_points - original_points[:, :1]) @ rotation.T
                rotated_data = {
                    "landmarks": rotated,
                    "X": np.asarray([landmarks_to_feature(points) for points in rotated], dtype=np.float32),
                    "y": data["y"][new_only],
                }
                rotated_prediction, _, _, _ = decide(OUTPUT / "candidate.onnx", rotated_data, new_config)
                results[split]["candidate"]["synthetic_rotation_recall"][str(degrees)] = {
                    name: float(np.mean(rotated_prediction[rotated_data["y"] == COMMANDS.index(name)]
                                        == COMMANDS.index(name)))
                    for name in ("peace", "rock")
                }
            # Landmark jitter is a narrow proxy for imperfect low-resolution
            # localization, not a simulation of missed fingers or low FPS.
            peace_mask = data["y"] == COMMANDS.index("peace")
            rng = np.random.default_rng(1302)
            results[split]["candidate"]["synthetic_peace_jitter"] = {}
            for pixels in (1, 2, 3):
                jittered = data["landmarks"][peace_mask] + rng.normal(
                    0, pixels / 320.0,
                    size=data["landmarks"][peace_mask].shape,
                ).astype(np.float32)
                jittered_data = {
                    "landmarks": jittered,
                    "X": np.asarray([landmarks_to_feature(points) for points in jittered], dtype=np.float32),
                    "y": data["y"][peace_mask],
                }
                jittered_prediction, _, _, _ = decide(OUTPUT / "candidate.onnx", jittered_data, new_config)
                results[split]["candidate"]["synthetic_peace_jitter"][str(pixels)] = {
                    "peace_recall": float(np.mean(jittered_prediction == COMMANDS.index("peace"))),
                    "wrong_direction_rate": float(np.mean(jittered_prediction < 4)),
                }
    (OUTPUT / "runtime_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    for split, result in results.items():
        print(split)
        for name in ("production", "candidate"):
            print(name, {key: value for key, value in result[name].items()
                         if key not in {"confusion", "in_plane_quadrant_recall"}})
        print(result["candidate_raw_to_resolved"])


if __name__ == "__main__":
    main()
