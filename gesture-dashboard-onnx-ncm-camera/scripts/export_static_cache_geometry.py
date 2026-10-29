"""Export deterministic real held-out landmark cases for all static classes.

Raw probabilities are one-hot deliberately: this checks geometry independently
of classifier error. Image dimensions/world depth are unavailable in this
cache, so this does not qualify the live detector or live Fist depth checks.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from backend.config import CLASS_NAMES, load_runtime_config
from backend.geometry import GeometryResolver, landmarks_to_feature


def main():
    config = load_runtime_config()
    resolver = GeometryResolver(config)
    source = ROOT / "models/gesture_online_validation_cache.npz"
    cases = []
    rng = np.random.default_rng(20261029)
    with np.load(source, allow_pickle=False) as cache:
        for label in sorted(set(cache["y"].tolist())):
            for index in rng.permutation(np.flatnonzero(cache["y"] == label))[:16]:
                points = cache["landmarks"][index]
                side = str(cache["handedness"][index])
                side = side if side.lower() in ("left", "right") else None
                # Unknown rows are tested against every command rather than
                # granting them their ground-truth label during inference.
                labels = range(13) if label == -1 else [int(label)]
                for raw in labels:
                    probabilities = np.eye(13)[raw]
                    adjusted, details = resolver.resolve(probabilities, points,
                        handedness=side, handedness_confidence=.99 if side else None,
                        require_fist_depth=False, image_shape=(600, 600, 3))
                    pose = details["pose_validation"]
                    cases.append(dict(id=len(cases), source=f"validation_cache/row={index}/truth={label}/raw={CLASS_NAMES[raw]}",
                        probabilities=probabilities.tolist(), landmarks=points.tolist(),
                        features=landmarks_to_feature(points).tolist(), handedness=side,
                        handedness_confidence=.99 if side else None, world_landmarks=None,
                        require_fist_depth=False,image_shape=[600,600,3],
                        expected=dict(probabilities=adjusted.tolist(),gesture=pose["gesture"],
                            valid=pose["valid"],geometry_supported=pose["geometry_supported"],
                            pose_validation=pose,temporal=[])))
    destination=ROOT / "static_gesture_runtime/validation/cache_geometry_reference.json"
    destination.write_text(json.dumps(dict(schema_version=1,geometry_cases=cases,
        limitations="Held-out landmark rows with forced classifier labels; no original image sizes or world depth. Not camera replay."),indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(cases=len(cases),valid_by_class=Counter(row["expected"]["gesture"] for row in cases if row["expected"]["valid"])),indent=2))


if __name__ == "__main__":
    main()
