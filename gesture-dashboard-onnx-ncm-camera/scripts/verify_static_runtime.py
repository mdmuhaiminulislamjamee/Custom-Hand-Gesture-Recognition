"""Verify a source checkout or extracted static runtime handover.

Needs requirements-static-conversion.txt for native LiteRT fixture execution.
Does not require the Boss app, training images, a camera, or TensorFlow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    models = root / "flutter/assets/models"
    baseline = json.loads((root / "validation/model_parity.json").read_text(encoding="utf-8"))
    provenance = json.loads((root / "contracts/detector_provenance.json").read_text(encoding="utf-8"))
    for relative, expected in provenance["files"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Detector source differs from dynamic delivery: {relative}")
    results = {}
    for name in ("float32", "int8"):
        manifest = json.loads((models / f"gesture_static_{name}_manifest.json").read_text(encoding="utf-8"))
        path = models / manifest["model_file"]
        if not path.resolve().is_relative_to(models.resolve()):
            raise ValueError("Unsafe model filename")
        if digest(path) != manifest["model_sha256"] or path.stat().st_size != manifest["model_bytes"]:
            raise ValueError(f"Invalid {name} model hash or size")
        if manifest["source_onnx_sha256"] != baseline["model_sha256"] or manifest["labels"] != baseline["class_names"]:
            raise ValueError("Mixed model contract")
        runtime = Interpreter(model_path=str(path), num_threads=1)
        runtime.allocate_tensors()
        inp = runtime.get_input_details()[0]
        outs = runtime.get_output_details()
        if inp["shape"].tolist() != [1, 76] or len(outs) != 2:
            raise ValueError("Unexpected tensor contract")
        maximum_probability_error, maximum_mass_error, agreement = 0., 0., 0
        for case in baseline["model_cases"]:
            row = np.asarray([case["features"]], np.float32)
            runtime.set_tensor(inp["index"], row)
            runtime.invoke()
            output = {n: runtime.get_tensor(outs[c["index"]]["index"]) for n, c in manifest["outputs"].items()}
            p, m = output["probabilities"][0], float(output["known_gesture_mass"][0, 0])
            maximum_probability_error = max(maximum_probability_error, float(np.abs(p-case["probabilities"]).max()))
            maximum_mass_error = max(maximum_mass_error, abs(m-case["known_gesture_mass"]))
            agreement += int(int(p.argmax()) == int(np.argmax(case["probabilities"])))
        passed = manifest["qualification"]["passed"]
        if name == "float32" and (not passed or maximum_probability_error > .0001 or maximum_mass_error > .0001 or agreement != len(baseline["model_cases"])):
            raise ValueError("FP32 fixture parity failed")
        results[name] = {"qualified_for_loading": passed, "fixture_rows": len(baseline["model_cases"]), "top1_agreement": agreement/len(baseline["model_cases"]), "maximum_probability_error": maximum_probability_error, "maximum_mass_error": maximum_mass_error}
    sums_path = root / "SHA256SUMS.json"
    if sums_path.exists():
        for relative, expected in json.loads(sums_path.read_text(encoding="utf-8")).items():
            if digest(root / relative) != expected:
                raise ValueError(f"Package file differs from checksum: {relative}")
    return {"passed": True, "detector_files_verified": len(provenance["files"]), "models": results, "physical_ios_tested": False}


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=base if (base / "flutter").is_dir() else base / "static_gesture_runtime")
    print(json.dumps(verify(parser.parse_args().package_root), indent=2))
