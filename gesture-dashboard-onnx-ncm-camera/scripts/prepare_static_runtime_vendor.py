"""Copy the exact delivered dynamic detector into the isolated static package.

Copies source/assets only; no old temporal classifier or dynamic weights are
included. Records upstream hashes so integrations can identify the detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "dynamic_gesture_runtime_api_v10_2_20261025.zip")
    parser.add_argument("--destination", type=Path, default=ROOT / "static_gesture_runtime/flutter")
    args = parser.parse_args()
    destination = args.destination.resolve()
    recorded = {}
    with zipfile.ZipFile(args.archive) as archive:
        roots = {PurePosixPath(n).parts[0] for n in archive.namelist() if n.endswith("contracts/mediapipe_detector.json")}
        if len(roots) != 1:
            raise ValueError("Could not identify unique upstream detector contract")
        prefix = next(iter(roots)) + "/"
        mappings = {
            "flutter/lib/landmark_telemetry/camera_frame_adapter.dart": "lib/detector/camera_frame_adapter.dart",
            "flutter/lib/landmark_telemetry/hand_detector_factory.dart": "lib/detector/hand_detector_factory.dart",
            "contracts/mediapipe_detector.json": "../contracts/mediapipe_detector.json",
        }
        for member in archive.namelist():
            if member.endswith("/") or not member.startswith(prefix):
                continue
            relative = member[len(prefix):]
            if relative.startswith("flutter/third_party/hand_detection/"):
                target_relative = relative[len("flutter/"):]
            elif relative in mappings:
                target_relative = mappings[relative]
            else:
                continue
            target = (destination / target_relative).resolve()
            if not target.is_relative_to(destination.parent):
                raise ValueError("Unsafe archive destination")
            payload = archive.read(member)
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f"Refusing to overwrite modified vendor file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            recorded[str(target.relative_to(destination.parent)).replace("\\", "/")] = hashlib.sha256(payload).hexdigest()
    payload = {
        "schema_version": 1,
        "archive_name": args.archive.name,
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "content_version": prefix.rstrip("/"),
        "archive_filename_version_mismatch": "v10_2" in args.archive.name and "v10_3" in prefix,
        "files": recorded,
        "provenance": "Unmodified source and detector assets from the user-supplied dynamic runtime archive.",
    }
    report = destination.parent / "contracts/detector_provenance.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Copied {len(recorded)} detector files; provenance: {report}")


if __name__ == "__main__":
    main()
