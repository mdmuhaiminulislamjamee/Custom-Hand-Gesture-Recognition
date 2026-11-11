"""Replay reviewed feedback against a classifier and the production hold gate.

These are training corrections, not an independent camera accuracy estimate.
The CSV lacks world landmarks and detector handedness; record that limitation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Auto-detect virtualenv if running outside .venv
try:
    import onnxruntime
except ModuleNotFoundError:
    venv_py = (PROJECT_ROOT / ".venv" / "Scripts" / "python.exe") if os.name == "nt" else (PROJECT_ROOT / ".venv" / "bin" / "python")
    if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve():
        import subprocess
        res = subprocess.run([str(venv_py)] + sys.argv, cwd=str(PROJECT_ROOT))
        sys.exit(res.returncode)
    raise

import numpy as np
from PIL import Image

from backend.config import load_runtime_config
from backend.geometry import GeometryResolver, landmarks_to_feature
from backend.hand_selection import gate_arguments
from backend.runtime import TemporalGate
from research.v18_20 import predict_onnx


def audit(model_path, feedback_path=None):
    config = load_runtime_config()
    feedback_path = feedback_path or PROJECT_ROOT / "feedback/misclassification_log.csv"
    with Path(feedback_path).open(encoding="utf-8-sig", newline="") as stream:
        raw_records = list(csv.DictReader(stream))
    records, points = [], []
    for r in raw_records:
        try:
            pts = np.asarray(json.loads(r["landmarks_json"]), np.float32)
            if pts.shape == (21, 2) and np.isfinite(pts).all():
                landmarks_to_feature(pts)
                records.append(r)
                points.append(pts)
        except Exception:
            continue
    probabilities, masses = predict_onnx(Path(model_path), np.asarray([landmarks_to_feature(p) for p in points]))
    resolver = GeometryResolver(config)
    rows = []
    for record, pose, probability, mass in zip(records, points, probabilities, masses):
        image = PROJECT_ROOT / "feedback/images" / Path(record["image_path"]).name
        shape = None
        if image.is_file():
            with Image.open(image) as source:
                shape = (source.height, source.width, 3)
        resolved, details = resolver.resolve(probability, pose, image_shape=shape)
        gate = TemporalGate(config)
        arguments = gate_arguments(details, {"known_gesture_mass": float(mass)})
        for tick in range(9):
            decision = gate.update(resolved, now=1. + tick * .1, **arguments)
        vector = pose[8] - pose[5]
        if shape:
            vector = vector * [shape[1], shape[0]]
        rows.append(dict(id=record["feedback_id"], actual=record["actual_label"],
                         raw=config.class_names[int(probability.argmax())],
                         confidence=float(probability.max()), mass=float(mass),
                         resolved=details["pose_validation"]["gesture"],
                         predicted=decision.predicted_gesture, executed=decision.execute,
                         reason=decision.reason, image_shape=shape,
                         index_xy_degrees=float(np.degrees(np.arctan2(vector[1], vector[0]))),
                         directional=details["directional"], pose=details["pose_validation"]))
    summary = {}
    for label in sorted({r["actual"] for r in rows}):
        subset = [r for r in rows if r["actual"] == label]
        summary[label] = dict(rows=len(subset), raw_labels=dict(Counter(r["raw"] for r in subset)),
                              raw_accepted_correct=sum(r["raw"] == label and r["mass"] >= config.known_mass_floor and r["confidence"] >= config.confidence_floor for r in subset),
                              runtime_correct=sum(r["predicted"] == label for r in subset),
                              executed_correct=sum(r["predicted"] == label and r["executed"] for r in subset),
                              failures=[r["id"] for r in subset if r["predicted"] != label])
    return dict(model=str(model_path), model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                runtime_sources={name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
                    for name in ("backend/geometry.py", "backend/runtime.py", "backend/model_runtime.py",
                                 "backend/hand_selection.py", "backend/direction_stability.py",
                                 "models/gesture_mobile_runtime_config.json")},
                feedback_sha256=hashlib.sha256(Path(feedback_path).read_bytes()).hexdigest(),
                summary=summary, rows=rows,
                limitation="Reviewed training corrections; world/depth and handedness were not retained. Static hold replay is not raw-image detector replay.")


def stress_audit(model_path):
    """Unseen transformations of the training corrections, with fixed grids."""
    config = load_runtime_config()
    with (PROJECT_ROOT / "feedback/misclassification_log.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = [r for r in csv.DictReader(stream) if r["actual_label"] in {"left", "right"}]
    poses, cases = [], []
    for record in records:
        with Image.open(PROJECT_ROOT / "feedback/images" / Path(record["image_path"]).name) as image:
            aspect = image.width / image.height
        points = np.asarray(json.loads(record["landmarks_json"]), np.float32)
        points = (points - points[0]) * [aspect, 1.]
        for mirror in (False, True):
            pose = points.copy()
            if mirror:
                pose[:, 0] *= -1
            label = record["actual_label"]
            if mirror:
                label = "right" if label == "left" else "left"
            for tilt in (-35., -20., 0., 20., 35.):
                vector = pose[8] - pose[5]
                target = np.deg2rad(tilt) + (0. if label == "left" else np.pi)
                angle = target - np.arctan2(vector[1], vector[0])
                c, s = np.cos(angle), np.sin(angle)
                pixels = pose @ np.asarray([[c, -s], [s, c]]).T
                for width, height in ((600, 600), (800, 600), (960, 540), (540, 720)):
                    poses.append(pixels / [width / height, 1.])
                    cases.append(dict(id=record["feedback_id"], label=label, mirror=mirror,
                                      tilt=tilt, shape=(height, width, 3)))
    probabilities, masses = predict_onnx(Path(model_path), np.asarray([landmarks_to_feature(p) for p in poses]))
    resolver = GeometryResolver(config)
    counts, failures = Counter(), []
    for points, probability, mass, case in zip(poses, probabilities, masses, cases):
        row, details = resolver.resolve(probability, points, image_shape=case["shape"])
        gate = TemporalGate(config)
        args = gate_arguments(details, {"known_gesture_mass": float(mass)})
        for tick in range(9):
            decision = gate.update(row, now=1. + tick * .1, **args)
        correct = decision.execute and decision.predicted_gesture == case["label"]
        counts[case["label"] + "_total"] += 1
        counts[case["label"] + "_correct"] += int(correct)
        if not correct:
            failures.append(dict(**case, predicted=decision.predicted_gesture, reason=decision.reason,
                                 raw=config.class_names[int(probability.argmax())], mass=float(mass),
                                 directional=details["directional"]))
    return dict(counts=dict(counts), failures=failures,
                limitation="Synthetic transforms of training feedback, not independent people or measured 3-D camera viewpoints.")


def replay_images(model_path, output_directory):
    """Redetect actual saved JPEGs, then replay each as a static held image."""
    from backend.model_runtime import InferenceEngine, OnnxClassifier, RuntimeSession
    config = load_runtime_config()
    metadata = json.loads((PROJECT_ROOT / "models/gesture_mlp_onnx_metadata.json").read_text(encoding="utf-8"))
    metadata["onnx_sha256"] = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    metadata_path = output_directory / "replay_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    engine = InferenceEngine(config, load_models=False)
    engine.model_manager.models["ONNX"] = OnnxClassifier(Path(model_path), metadata_path, config)
    engine.model_manager.errors.clear()
    if not engine.detector.ready:
        raise RuntimeError(engine.detector.error)
    with (PROJECT_ROOT / "feedback/misclassification_log.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = [r for r in csv.DictReader(stream) if r["actual_label"] in {"left", "right"}]
    rows = []
    try:
        for number, record in enumerate(records):
            image_path = PROJECT_ROOT / "feedback/images" / Path(record["image_path"]).name
            image_bytes = image_path.read_bytes()
            session = RuntimeSession(config)
            results = [engine.process_frame(image_bytes, session, now=1. + number * 4 + tick * .1)
                       for tick in range(12)]
            last = results[-1]
            expected_action = config.gesture_to_action[record["actual_label"]]
            rows.append(dict(id=record["feedback_id"], actual=record["actual_label"],
                             predicted=last.get("runtime_prediction"), status=last.get("status"),
                             action_seen=any(r.get("runtime_action") == expected_action for r in results),
                             reason=last.get("action_reason"), quality=last.get("quality"),
                             raw=last.get("raw_prediction"), mass=last.get("known_gesture_mass"),
                             landmarks=last.get("landmarks"), world_landmarks=last.get("world_landmarks"),
                             resolvers=last.get("resolvers"),
                             annotated_landmarks=json.loads(record["landmarks_json"])))
            if number % 10 == 0:
                print(f"JPEG replay {number + 1}/{len(records)}", flush=True)
    finally:
        engine.close()
    return dict(rows=rows, total=len(rows),
                correct=sum(r["predicted"] == r["actual"] for r in rows),
                correct_action=sum(r["action_seen"] for r in rows),
                limitation="Stored reviewed images are training evidence. Repeated still images test detection and the hold/action pipeline, not live motion or new environments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/gesture_mlp_production.onnx")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--images", action="store_true")
    args = parser.parse_args()
    report = audit(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stress:
        report["stress"] = stress_audit(args.model)
    if args.images:
        report["image_replay"] = replay_images(args.model, args.output.parent)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
