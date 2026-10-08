"""Export deterministic geometry/model fixtures from the current Python runtime."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.config import load_runtime_config
from backend.geometry import GeometryResolver, landmarks_to_feature
from backend.model_runtime import ModelManager
from backend.runtime import TemporalGate


def main():
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "swift/tests/parity_vectors.json"
    config = load_runtime_config()
    original = GeometryResolver.resolve
    captured = {}

    def record(self, probabilities, landmarks, **kwargs):
        result = original(self, probabilities, landmarks, **kwargs)
        p, xy = np.asarray(probabilities), np.asarray(landmarks)
        if p.shape != (12,) or xy.shape != (21, 2) or not np.isfinite(xy).all():
            return result
        case = dict(probabilities=p.tolist(), landmarks=xy.tolist(),
                    handedness=kwargs.get("handedness"),
                    handedness_confidence=kwargs.get("handedness_confidence"),
                    world_landmarks=None if kwargs.get("world_landmarks") is None else np.asarray(kwargs["world_landmarks"]).tolist(),
                    require_fist_depth=kwargs.get("require_fist_depth", False))
        key = json.dumps(case, sort_keys=True)
        captured.setdefault(key, (case, os.environ.get("PYTEST_CURRENT_TEST", "unknown").split(" (")[0]))
        return result

    GeometryResolver.resolve = record
    try:
        status = pytest.main(["-q", "-o", "addopts=", "backend/tests/test_reported_fist_views.py",
                              "backend/tests/test_reported_peace_dorsal.py", "backend/tests/test_geometry.py"])
    finally:
        GeometryResolver.resolve = original
    if status:
        raise SystemExit(status)
    resolver = GeometryResolver(config)
    cases = []
    for case, source in captured.values():
        xy = np.asarray(case["landmarks"], dtype=np.float32)
        kwargs = {k: case[k] for k in ("handedness", "handedness_confidence", "world_landmarks", "require_fist_depth")}
        if kwargs["world_landmarks"] is not None:
            kwargs["world_landmarks"] = np.asarray(kwargs["world_landmarks"], dtype=np.float32)
        adjusted, details = resolver.resolve(np.asarray(case["probabilities"]), xy, **kwargs)
        pose, directional = details["pose_validation"], details["directional"]
        recovery = bool(pose["geometry_supported"] and directional.get("strong_geometry") and directional.get("model_supported_pose"))
        temporal = []
        for mass in (1.0, 0.2, 0.0):
            gate = TemporalGate(config)
            decisions = []
            for timestamp in (10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6):
                result = gate.update(adjusted, now=timestamp, known_gesture_mass=mass,
                    pose_valid=pose["valid"], rejection_reason=pose["reason"], geometry_supported=pose["geometry_supported"],
                    directional_recovery=recovery, sideways_palm_recovery=pose["sideways_palm_recovery"],
                    dorsal_range_recovery=pose["dorsal_range_recovery"])
                decisions.append(result.execute)
            temporal.append(dict(mass=mass, execute=decisions))
        case.update(id=len(cases), source=source, features=landmarks_to_feature(xy).tolist(),
                    expected=dict(probabilities=adjusted.tolist(), gesture=pose["gesture"], valid=pose["valid"],
                        geometry_supported=pose["geometry_supported"], directional_recovery=recovery,
                        sideways_palm_recovery=pose["sideways_palm_recovery"], dorsal_range_recovery=pose["dorsal_range_recovery"],
                        temporal=temporal))
        cases.append(case)
    classifier = ModelManager(config).models["ONNX"]
    model_cases = []
    with np.load(ROOT / "models/gesture_online_untouched_test_cache.npz", allow_pickle=False) as cache:
        for label in range(-1, 12):
            for row in np.flatnonzero(cache["y"] == label)[:2]:
                features = cache["X"][row].astype(np.float32)
                probabilities, mass = classifier.predict_with_quality(features.reshape(1,76))
                model_cases.append(dict(label=int(label), features=features.tolist(),
                    probabilities=probabilities[0].tolist(), known_gesture_mass=float(np.asarray(mass).reshape(-1)[0])))
    payload = dict(schema_version=1, model_sha256=hashlib.sha256((ROOT / "models/gesture_mlp_production.onnx").read_bytes()).hexdigest(),
                   feature_tolerance=0.0001, probability_tolerance=0.0001,
                   geometry_cases=cases, model_cases=model_cases)
    payload["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ["backend/geometry.py", "backend/runtime.py", "models/gesture_mobile_runtime_config.json"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(cases)} geometry cases and {len(model_cases)} ONNX cases to {destination}")

if __name__ == "__main__":
    main()
