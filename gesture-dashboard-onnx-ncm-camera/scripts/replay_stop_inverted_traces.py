"""Replay reviewed gesture screen-overlay traces through the shipped classifier and gate.

These are approximate 2-D traces, with synthetic confident handedness and no
depth. This checks the runtime correction, not camera detector performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.artifact_integrity import sha256_file
from backend.config import PROJECT_ROOT, load_runtime_config
from backend.geometry import GeometryResolver, landmarks_to_feature
from backend.hand_selection import gate_arguments
from backend.model_runtime import ModelManager
from backend.runtime import TemporalGate


DEFAULT_TRACES = PROJECT_ROOT / "artifacts/stop_inverted/reverse_palm_review/overlay_traces.json"


def replay(traces_path: Path):
    review = json.loads(traces_path.read_text(encoding="utf-8"))
    viewport = review["viewport"]
    shape = (viewport["height"], viewport["width"], 3)
    config = load_runtime_config()
    manager = ModelManager(config)
    if not manager.ready:
        raise RuntimeError(manager.errors)
    resolver = GeometryResolver(config)
    results = []
    for frame in review["frames"]:
        for mirror in (False, True):
            points = (np.asarray(frame["points"], np.float32) - [viewport["x"], viewport["y"]]) / [shape[1], shape[0]]
            if mirror:
                points[:, 0] = 1. - points[:, 0]
            side = "Left" if mirror else review["handedness"]
            expected = review["intended_gesture"]
            if mirror:
                expected = {"left": "right", "right": "left"}.get(expected, expected)
            probabilities, _, _, quality = manager.predict_detailed(landmarks_to_feature(points))
            row, details = resolver.resolve(probabilities, points, handedness=side,
                handedness_confidence=.99, image_shape=shape)
            # Test the actual classifier mass and the observed low-mass miss
            # separately. The latter was displayed by the original recording,
            # whose precise detector landmarks cannot be recovered here.
            outcomes = {}
            for name, mass in (("shipped_onnx", quality["known_gesture_mass"]),
                               ("recorded_mass", frame["displayed_mass"])):
                gate = TemporalGate(config)
                args = gate_arguments(details, {"known_gesture_mass": mass})
                for tick in range(7):
                    decision = gate.update(row, now=1 + tick * .1, **args)
                outcomes[name] = dict(prediction=decision.predicted_gesture,
                    execute=decision.execute, known_gesture_mass=float(mass), reason=decision.reason)
            results.append(dict(seconds=frame["seconds"], mirror=mirror,
                expected_gesture=expected,
                displayed_gesture=frame["displayed_gesture"],
                raw_classifier_label=config.class_names[int(probabilities.argmax())],
                geometry=(details["directional"].get("horizontal_pointing_geometry")
                          if expected in {"left", "right"} else
                          details["hand_shape"]["stop_inverted_geometry"]), outcomes=outcomes))
    return dict(source=review["source"], traced_frames=len(review["frames"]),
        originally_missed_traced_frames=sum(f["displayed_gesture"] == "no_gesture" for f in review["frames"]),
        replay_cases=len(results),
        passed_cases=sum(all(o["execute"] and o["prediction"] == r["expected_gesture"]
                             for o in r["outcomes"].values()) for r in results),
        model_sha256=sha256_file(manager.models[manager.selected_model_name].model_path),
        limitations=[review["method"], review["handedness_note"],
            "Mirrors are synthetic coverage; original detector handedness confidence and depth are unavailable.",
            "This checks sampled landmark traces, not every video frame or live-camera performance."],
        results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = replay(args.traces)
    output = args.output or args.traces.with_name("replay_report.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['passed_cases']}/{report['replay_cases']} traced poses passed; {output}")
    if report["passed_cases"] != report["replay_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
