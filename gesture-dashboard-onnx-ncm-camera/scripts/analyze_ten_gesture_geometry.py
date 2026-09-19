"""Print geometry-gate distributions for the ten-command qualification set."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
from pathlib import Path

import numpy as np

from backend.config import CLASS_NAMES, RuntimeConfig
from backend.geometry import GeometryResolver
from research.v18_20 import COMMAND_COUNT, load_npz, predict_onnx


def _quantiles(values: list[float]) -> list[float]:
    if not values:
        return []
    return np.quantile(values, [0, .05, .1, .25, .5, .75, .9, .95, 1]).round(4).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--path', type=Path)
    arguments = parser.parse_args()
    root = Path.cwd()
    data = load_npz(
        arguments.path
        if arguments.path is not None
        else root / f"artifacts/ten_gesture/public_{arguments.split}.npz"
    )
    probabilities, mass = predict_onnx(
        root / "artifacts/ten_gesture/candidate.onnx", data["X"]
    )
    config = RuntimeConfig(
        known_mass_floor=.8,
        confidence_floor=.7,
        probability_margin_floor=.08,
        raw={"open_palm_orientation": {"minimum_up_alignment": .7, "minimum_axis_dominance": .7}},
    )
    resolver = GeometryResolver(config)
    records = []
    fields: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for index, points in enumerate(data["landmarks"]):
        adjusted, details = resolver.resolve(probabilities[index], points)
        truth_index = int(data["y"][index])
        truth = "no_gesture" if truth_index == COMMAND_COUNT else CLASS_NAMES[truth_index]
        pose = details["pose_validation"]
        shape = details.get("hand_shape") or {}
        ordered = np.sort(adjusted)
        accepted = bool(
            pose["valid"]
            and (mass[index] >= config.known_mass_floor or pose.get("geometry_supported", False))
            and ordered[-1] >= config.confidence_floor
            and ordered[-1] - ordered[-2] >= config.probability_margin_floor
        )
        raw_name = CLASS_NAMES[int(probabilities[index].argmax())]
        resolved_name = CLASS_NAMES[int(adjusted.argmax())]
        records.append((truth, raw_name, resolved_name, accepted, pose, shape, float(mass[index]), float(probabilities[index].max()), str(data.get('variant', np.asarray(['unknown'] * len(data['y'])))[index])))
        if truth in {"fist", "thumb_down", "like", "no_gesture"}:
            for key in (
                "thumb_extension", "maximum_other_finger_extension",
                "mean_other_finger_extension", "thumb_down_score",
                "thumb_down_lead_ratio", "fist_score", "open_palm_up_alignment",
            ):
                if key in shape:
                    fields[truth][key].append(float(shape[key]))
    for truth in ("fist", "thumb_down", "like", "no_gesture"):
        subset = [row for row in records if row[0] == truth]
        print("\n", truth, "rows", len(subset))
        print("raw", Counter(row[1] for row in subset))
        print("resolved", Counter(row[2] for row in subset))
        print("accepted", sum(row[3] for row in subset))
        print("invalid reasons", Counter(row[4].get("reason") for row in subset if not row[4]["valid"]))
        print("geometry supported", Counter(bool(row[4].get("geometry_supported")) for row in subset))
        print("known-mass", _quantiles([row[6] for row in subset]))
        for key, values in fields[truth].items():
            print(key, _quantiles(values))
        if truth in {'thumb_down', 'fist', 'like'}:
            for row in subset:
                if row[1] == truth and not row[4]['valid']:
                    shape = row[5]
                    mean_non_thumb = float(np.mean([
                        shape['index_extension'], shape['middle_extension'],
                        shape['ring_extension'], shape['pinky_extension'],
                    ]))
                    print('invalid-shape', {
                        'truth': truth,
                        'confidence': round(row[7], 4), 'mass': round(row[6], 4),
                        'extension': round(shape['thumb_extension'], 4),
                        'down_score': round(shape['thumb_down_score'], 4),
                        'lead': round(shape['thumb_down_lead_ratio'], 4),
                        'mean_other': round(mean_non_thumb, 4),
                    })
    unknown = [row for row in records if row[0] == "no_gesture" and row[3]]
    print("\naccepted unknown", len(unknown))
    print("raw", Counter(row[1] for row in unknown))
    print("resolved", Counter(row[2] for row in unknown))
    print("geometry supported", Counter(bool(row[4].get("geometry_supported")) for row in unknown))
    print("known-mass", _quantiles([row[6] for row in unknown]))
    for name in sorted(set(row[2] for row in unknown)):
        print(name, "mass", _quantiles([row[6] for row in unknown if row[2] == name]))
        print(name, "mass-values", [round(row[6], 4) for row in unknown if row[2] == name])
    print("reasons", Counter(row[4].get("reason") for row in unknown))
    print("variants", Counter(row[8] for row in unknown))


if __name__ == "__main__":
    main()
