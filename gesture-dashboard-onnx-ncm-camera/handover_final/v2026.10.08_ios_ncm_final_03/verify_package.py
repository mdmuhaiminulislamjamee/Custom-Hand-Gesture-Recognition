"""Read-only, standard-library handover checksum and model-contract verifier."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

CLASSES = ["left", "right", "up", "down", "open_palm", "like", "dorsal", "ok", "fist", "thumb_down", "peace", "rock"]

def safe(name):
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and ".." not in p.parts and "\\" not in name and ":" not in name

def verify(root):
    root = Path(root).resolve()
    listing = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    expected = set(listing)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.json"}
    if actual != expected:
        raise ValueError(f"Package file set differs: missing={expected-actual}, extra={actual-expected}")
    for name, info in listing.items():
        if not safe(name):
            raise ValueError(f"Unsafe path: {name}")
        p = root / name
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe link: {name}")
        payload = p.read_bytes()
        if len(payload) != info["bytes"] or hashlib.sha256(payload).hexdigest() != info["sha256"]:
            raise ValueError(f"Hash/size mismatch: {name}")
    model_root = root / "ios/models"
    metadata = json.loads((model_root / "gesture_mlp_onnx_metadata.json").read_text())
    runtime = json.loads((model_root / "gesture_mobile_runtime_config.json").read_text())
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text())
    digest = hashlib.sha256((model_root / "gesture_mlp_production.onnx").read_bytes()).hexdigest()
    if any(order != CLASSES for order in [metadata["output_class_order"], runtime["class_names"], manifest["class_names"]]):
        raise ValueError("Class order mismatch")
    if digest != metadata["onnx_sha256"] or digest != manifest["model_sha256"]:
        raise ValueError("Model hash mismatch")
    with ZipFile(root / "desktop_reference.zip") as archive:
        entries = [i.filename for i in archive.infolist() if not i.is_dir()]
        if len(entries) != len(set(entries)) or not all(safe(n) for n in entries):
            raise ValueError("Unsafe/duplicate desktop ZIP entries")
        if archive.testzip() is not None:
            raise ValueError("Desktop ZIP CRC mismatch")
        for name in ["gesture_mlp_production.onnx", "gesture_mlp_onnx_metadata.json", "gesture_mobile_runtime_config.json", "hand_landmarker.task", "blaze_face_short_range.tflite"]:
            if archive.read("desktop_reference/models/" + name) != (model_root / name).read_bytes():
                raise ValueError(f"Desktop/iOS asset mismatch: {name}")
        prohibited = ("/.venv/", "/node_modules/", "/.runtime/", "/feedback/", "/.git/", "gesture_online_adapter_state", "online_adapter_backups")
        if any(any(token in n for token in prohibited) for n in entries):
            raise ValueError("Local runtime/private state found in desktop ZIP")
    return {"verified_files": len(listing), "model_sha256": digest, "desktop_ios_assets_match": True}

def verify_onnx(root):
    import numpy as np
    import onnxruntime as ort
    root = Path(root)
    session = ort.InferenceSession(str(root / "ios/models/gesture_mlp_production.onnx"), providers=["CPUExecutionProvider"])
    vectors = json.loads((root / "ios/swift/tests/parity_vectors.json").read_text())
    errors = []
    for row in vectors["model_cases"]:
        probabilities, mass = session.run(["probabilities", "known_gesture_mass"],
            {"landmark_features": np.asarray([row["features"]], dtype=np.float32)})
        if probabilities.shape != (1,12) or mass.shape != (1,1):
            raise ValueError("Unexpected output shape")
        error = max(float(np.max(np.abs(probabilities[0] - row["probabilities"]))), abs(float(mass[0,0]) - row["known_gesture_mass"]))
        if error > 0.0001:
            raise ValueError("ONNX fixture mismatch")
        errors.append(error)
    return {"onnx_cases": len(errors), "maximum_absolute_error": max(errors)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--onnx", action="store_true", help="Also run fixtures; needs numpy and onnxruntime")
    args = parser.parse_args()
    result = verify(args.directory)
    if args.onnx:
        result.update(verify_onnx(args.directory))
    print(json.dumps(result, indent=2))
