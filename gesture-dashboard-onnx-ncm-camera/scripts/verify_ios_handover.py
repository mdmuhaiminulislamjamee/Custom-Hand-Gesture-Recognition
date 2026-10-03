"""Verify the staged iOS release ZIP without extracting or trusting its paths."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from backend.config import CLASS_NAMES, GESTURE_TO_ACTION


def verify(path: Path) -> dict:
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        files = {item.filename: item for item in archive.infolist() if not item.is_dir()}
        if not files or any(
            name.startswith("/") or ".." in PurePosixPath(name).parts
            for name in files
        ):
            raise ValueError("ZIP has an unsafe or empty file layout")
        prefix = "ios/"
        checksums = prefix + "SHA256SUMS.csv"
        if checksums not in files:
            raise ValueError("ZIP has no SHA256SUMS.csv")
        rows = list(csv.DictReader(io.StringIO(archive.read(checksums).decode("utf-8-sig"))))
        expected = {prefix + row["path"]: row for row in rows}
        if set(files) != set(expected) | {checksums}:
            raise ValueError("Checksum listing does not match ZIP contents")
        for name, row in expected.items():
            payload = archive.read(name)
            if len(payload) != int(row["bytes"]):
                raise ValueError(f"Size mismatch: {name}")
            if hashlib.sha256(payload).hexdigest() != row["sha256"].lower():
                raise ValueError(f"SHA256 mismatch: {name}")
        metadata = json.loads(archive.read(prefix + "models/gesture_mlp_onnx_metadata.json").decode("utf-8-sig"))
        runtime = json.loads(archive.read(prefix + "models/gesture_mobile_runtime_config.json").decode("utf-8-sig"))
        manifest = json.loads(archive.read(prefix + "contracts/IOS_RELEASE_MANIFEST.json").decode("utf-8-sig"))
        for order in (metadata["output_class_order"], runtime["class_names"], manifest["output_class_order"]):
            if order != CLASS_NAMES:
                raise ValueError("Class order mismatch in iOS ZIP")
        if runtime["gesture_to_action"] != GESTURE_TO_ACTION or manifest["gesture_to_action"] != GESTURE_TO_ACTION:
            raise ValueError("Action mapping mismatch in iOS ZIP")
        model = archive.read(prefix + "models/gesture_mlp_production.onnx")
        if hashlib.sha256(model).hexdigest() != metadata["onnx_sha256"]:
            raise ValueError("ONNX hash mismatch in iOS ZIP")
        if manifest["outputs"]["probabilities"] != [1, len(CLASS_NAMES)]:
            raise ValueError("Probability tensor shape mismatch in iOS ZIP")
        return {"archive": str(path), "verified_files": len(files), "model_sha256": metadata["onnx_sha256"]}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.verify_ios_handover path/to/handover.zip")
    print(json.dumps(verify(Path(sys.argv[1])), indent=2))
