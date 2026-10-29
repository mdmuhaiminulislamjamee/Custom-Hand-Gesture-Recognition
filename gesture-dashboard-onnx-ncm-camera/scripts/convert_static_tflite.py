"""Convert the installed MLP, preserving its open-set probability contract.

Run in a separate environment (see requirements-static-conversion.txt). This
does not train a model or modify models/. Calibration uses the replay cache;
validation and untouched-test caches are only used for evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter
from static_tflite_graph import build_graph

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "static_gesture_runtime/flutter/assets/models"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_model():
    path = ROOT / "models/gesture_mlp_production.onnx"
    metadata = json.loads((path.parent / "gesture_mlp_onnx_metadata.json").read_text(encoding="utf-8"))
    if sha(path) != metadata["onnx_sha256"]:
        raise ValueError("Production ONNX does not match its metadata")
    graph = onnx.load(path)
    expected = ["Sub", "Div", "MatMul", "Add", "Relu", "MatMul", "Add", "Relu", "MatMul", "Add", "Softmax", "Gather", "ReduceSum", "Add", "ReduceSum", "Div", "ArgMax"]
    if [n.op_type for n in graph.graph.node] != expected:
        raise ValueError("Source graph changed: review conversion before proceeding")
    weights = {v.name: numpy_helper.to_array(v) for v in graph.graph.initializer}
    if [weights[f"w{i}"].shape for i in range(3)] != [(76, 128), (128, 64), (64, 14)]:
        raise ValueError("Unexpected classifier topology")
    if metadata["output_class_order"][6] != "dorsal" or len(metadata["feature_names"]) != 76:
        raise ValueError("Unexpected feature or output taxonomy")
    return path, metadata, weights


def calibration_rows(max_rows: int):
    path = ROOT / "models/gesture_online_replay_cache.npz"
    with np.load(path, allow_pickle=False) as data:
        x, y = data["X"].copy(), data["y"].copy()
    if x.shape[1] != 76 or not np.isfinite(x).all():
        raise ValueError("Invalid calibration cache")
    rng = np.random.default_rng(20261029)
    # Include all labels, including unknowns, without consulting test rows.
    count = max(1, max_rows // len(np.unique(y)))
    chosen = np.concatenate([rng.permutation(np.flatnonzero(y == label))[:count] for label in np.unique(y)])
    chosen.sort()
    return x[chosen].astype(np.float32), {"source": str(path.relative_to(ROOT)), "sha256": sha(path), "rows": len(chosen), "indices_sha256": hashlib.sha256(chosen.tobytes()).hexdigest(), "test_used_for_calibration": False}


def tensor_record(detail, position):
    scale, zero = detail["quantization"]
    return {"name": detail["name"], "dtype": np.dtype(detail["dtype"]).name, "shape": detail["shape"].tolist(), "index": position, "tensor_index": int(detail["index"]), "scale": float(scale), "zero_point": int(zero)}


def load_interpreter(path):
    interpreter = Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    outputs = interpreter.get_output_details()
    contracts = {d["name"]: tensor_record(d, i) for i, d in enumerate(outputs)}
    if set(contracts) != {"probabilities", "known_gesture_mass"}:
        raise ValueError("Missing named output contract")
    return interpreter, contracts


def convert(weights, calibration, destination, quantized):
    destination.write_bytes(build_graph(weights, calibration, quantized))
    return load_interpreter(destination)


def invoke(interpreter, contracts, rows):
    input_detail = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()
    results = {name: np.empty((len(rows), record["shape"][-1]), dtype=np.float32) for name, record in contracts.items()}
    timings = []
    for i, row in enumerate(rows):
        values = row[None].astype(np.float32)
        if input_detail["dtype"] == np.int8:
            scale, zero = input_detail["quantization"]
            values = np.clip(np.rint(values / scale + zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], values)
        start = time.perf_counter_ns()
        interpreter.invoke()
        timings.append((time.perf_counter_ns() - start) / 1e6)
        for name, record in contracts.items():
            values = interpreter.get_tensor(outputs[record["index"]]["index"])
            if record["dtype"] == "int8":
                values = (values.astype(np.float32) - record["zero_point"]) * record["scale"]
            results[name][i] = values[0]
    return results["probabilities"], results["known_gesture_mass"].ravel(), {"p50_ms": float(np.median(timings[10:])), "p95_ms": float(np.percentile(timings[10:], 95)), "scope": "warm host CPU invoke only; not iPhone or camera latency"}


def accepted(p, mass):
    rank = np.sort(p, axis=1)
    arg = p.argmax(axis=1)
    return np.where((mass >= .70) & (rank[:, -1] >= .70) & (rank[:, -1] - rank[:, -2] >= .08) & (arg != 6), arg, 13)


def metrics(p, mass, labels):
    pred = accepted(p, mass)
    selected = pred != 13
    unknown = labels == 13
    return {"accepted_precision": float(np.mean(pred[selected] == labels[selected])) if selected.any() else 0., "unknown_false_accept_rate": float(np.mean(selected[unknown])) if unknown.any() else None, "acceptance_rate": float(selected.mean())}


def evaluate(path, interpreter, contracts, session):
    with np.load(path, allow_pickle=False) as data:
        rows, y = data["X"].copy(), data["y"].copy()
    # Existing online caches encode hard negatives / unknowns as -1.
    y = np.where(y == -1, 13, y)
    if not np.isin(y, np.arange(14)).all():
        raise ValueError("Unexpected evaluation cache labels")
    base_p, base_m = session.run(["probabilities", "known_gesture_mass"], {"landmark_features": rows})
    base_m = base_m.ravel()
    p, m, timings = invoke(interpreter, contracts, rows)
    comparisons = {"rows": len(rows), "source_sha256": sha(path), "top1_agreement": float(np.mean(p.argmax(1) == base_p.argmax(1))), "maximum_probability_error": float(np.max(np.abs(p-base_p))), "mean_probability_error": float(np.mean(np.abs(p-base_p))), "maximum_mass_error": float(np.max(np.abs(m-base_m))), "scalar_gate_agreement": float(np.mean(accepted(p,m) == accepted(base_p,base_m))), "source_scalar_metrics": metrics(base_p,base_m,y), "tflite_scalar_metrics": metrics(p,m,y), "host_invoke": timings, "geometry_and_temporal_not_included": True}
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-rows", type=int, default=1400)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source, metadata, weights = source_model()
    calibration, calibration_info = calibration_rows(args.calibration_rows)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    report = {"schema_version": 1, "source_onnx_sha256": sha(source), "weights_retrained": False, "converter": "explicit_builtin_flatbuffer_graph", "calibration": calibration_info, "models": {}, "limitations": ["Landmark-level conversion checks do not measure camera detection or iPhone latency.", "New command-controller timing is tested separately; scalar-gate agreement is not full runtime parity.", "INT8 candidate quantizes the three dense layers; normalization and open-set probability tail remain FP32."]}
    for variant in ("float32", "int8"):
        print(f"Converting {variant}", flush=True)
        path = args.output / f"gesture_static_{variant}.tflite"
        try:
            interpreter, contracts = convert(weights, calibration, path, variant == "int8")
            evaluations = {split: evaluate(ROOT / f"models/{cache}", interpreter, contracts, session) for split, cache in (("validation", "gesture_online_validation_cache.npz"), ("untouched_test", "gesture_online_untouched_test_cache.npz"))}
            test = evaluations["untouched_test"]
            if variant == "float32":
                passed = all(v["maximum_probability_error"] <= 1e-4 and v["maximum_mass_error"] <= 1e-4 and v["top1_agreement"] == 1.0 and v["scalar_gate_agreement"] == 1.0 for v in evaluations.values())
            else:
                passed = all(v["top1_agreement"] >= .995 and v["scalar_gate_agreement"] >= .995 and v["tflite_scalar_metrics"]["accepted_precision"] >= v["source_scalar_metrics"]["accepted_precision"] - .002 and v["tflite_scalar_metrics"]["unknown_false_accept_rate"] <= v["source_scalar_metrics"]["unknown_false_accept_rate"] + .002 for v in evaluations.values())
            manifest = {"schema_version": 1, "model_name": metadata["model_name"], "source_onnx_sha256": sha(source), "model_file": path.name, "model_sha256": sha(path), "model_bytes": path.stat().st_size, "labels": metadata["output_class_order"], "feature_names": metadata["feature_names"], "input": tensor_record(interpreter.get_input_details()[0], 0), "outputs": contracts, "normalization_embedded": True, "precision": "int8_dense_float32_normalization_and_tail" if variant == "int8" else "float32", "qualification": {"passed": passed, "conversion_passed": passed, "recommended_default": variant == "float32" and passed, "physical_ios_tested": False}, "calibration": calibration_info if variant == "int8" else None}
            write_json(path.with_name(path.stem + "_manifest.json"), manifest)
            report["models"][variant] = {"passed": passed, "model_sha256": sha(path), "bytes": path.stat().st_size, "evaluation": evaluations}
            print(f"{variant}: passed={passed}; top1={test['top1_agreement']:.6f}; scalar gate={test['scalar_gate_agreement']:.6f}", flush=True)
        except Exception as error:
            report["models"][variant] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
            print(f"{variant} conversion failed: {error}", flush=True)
    report_path = args.output.parents[2] / "validation/conversion_report.json"
    write_json(report_path, report)
    if not report["models"]["float32"]["passed"]:
        raise SystemExit("FP32 parity failed; candidate must not replace the deployed model")
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
