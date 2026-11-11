"""Verify the extracted Boss handover using only Python's standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _legacy_digest(stream)


def _legacy_digest(stream) -> str:
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe payload path: {relative}")
    result = root.joinpath(*posix.parts).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Payload path escapes handover: {relative}")
    return result


def verify(root: Path) -> dict:
    sums = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for relative, expected in sums.items():
        path = checked_path(root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or changed payload: {relative}")
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    checks = json.loads((root / "HOST_CHECKS.json").read_text(encoding="utf-8"))
    if checks.get("status") != "passed":
        raise ValueError("Host qualification is not passed")
    for key in ("version_name", "version_code", "sha256"):
        if manifest["android_test_apk"][key] != checks["android_artifact"][key]:
            raise ValueError(f"APK metadata differs from checked artifact: {key}")
    for relative, expected in checks.get("source_sha256", {}).items():
        if sha256(checked_path(root, relative)) != expected:
            raise ValueError(f"Source differs from checked revision: {relative}")
    package = root / "static_gesture_runtime"
    runtime_sums = json.loads((package / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for relative, expected in runtime_sums.items():
        if sha256(checked_path(package, relative)) != expected:
            raise ValueError(f"Changed runtime file: {relative}")
    provenance = json.loads((package / "contracts/detector_provenance.json").read_text(encoding="utf-8"))
    detector_files = {**provenance["files"], **provenance.get("local_patches", {})}
    for relative, expected in detector_files.items():
        if sha256(checked_path(package, relative)) != expected:
            raise ValueError(f"Detector differs from supplied dynamic source: {relative}")

    models = package / "flutter/assets/models"
    source_sha = sha256(root / "source_model/gesture_mlp_production.onnx")
    if source_sha != manifest["source_onnx_sha256"]:
        raise ValueError("Source ONNX differs from the release manifest")
    if sha256(root / "source_model/gesture_mobile_runtime_config.json") != manifest["runtime_config_sha256"]:
        raise ValueError("Runtime configuration differs from the release manifest")
    reference = json.loads((package / "validation/model_parity.json").read_text(encoding="utf-8"))
    conversion = json.loads((package / "validation/conversion_report.json").read_text(encoding="utf-8"))
    if reference["model_sha256"] != source_sha or conversion["source_onnx_sha256"] != source_sha:
        raise ValueError("Fixtures or conversion report belong to a different source model")
    if conversion["runtime_config_sha256"] != manifest["runtime_config_sha256"]:
        raise ValueError("Conversion used another runtime configuration")
    for name in ("float32", "int8"):
        model_manifest = json.loads((models / f"gesture_static_{name}_manifest.json").read_text(encoding="utf-8"))
        model = checked_path(models, model_manifest["model_file"])
        if sha256(model) != model_manifest["model_sha256"] or model.stat().st_size != model_manifest["model_bytes"]:
            raise ValueError(f"Invalid {name} model")
        if model_manifest["source_onnx_sha256"] != source_sha:
            raise ValueError(f"The {name} model belongs to a different ONNX source")
        expected_qualified = True if name == "float32" else manifest["int8_qualified"]
        if bool(model_manifest["qualification"]["passed"]) != expected_qualified:
            raise ValueError("Unexpected model qualification; FP32 must be the qualified default")

    apk = checked_path(root, manifest["android_test_apk"]["path"])
    if sha256(apk) != manifest["android_test_apk"]["sha256"]:
        raise ValueError("Android APK differs from release manifest")
    asset_sources = {
        "assets/flutter_assets/packages/static_gesture_runtime/assets/models/gesture_static_float32.tflite": models / "gesture_static_float32.tflite",
        "assets/flutter_assets/packages/static_gesture_runtime/assets/models/gesture_static_int8.tflite": models / "gesture_static_int8.tflite",
        "assets/flutter_assets/packages/hand_detection/assets/models/hand_detection.tflite": package / "flutter/third_party/hand_detection/assets/models/hand_detection.tflite",
        "assets/flutter_assets/packages/hand_detection/assets/models/hand_landmark_full.tflite": package / "flutter/third_party/hand_detection/assets/models/hand_landmark_full.tflite",
        "assets/flutter_assets/packages/hand_detection/assets/models/blaze_face_short_range.tflite": package / "flutter/third_party/hand_detection/assets/models/blaze_face_short_range.tflite",
    }
    with zipfile.ZipFile(apk) as archive:
        if archive.testzip() is not None:
            raise ValueError("Android APK has corrupt ZIP entries")
        names = archive.namelist()
        abis = {name.split("/")[1] for name in names if name.startswith("lib/") and name.endswith(".so")}
        if abis != {"arm64-v8a"} or "lib/arm64-v8a/libapp.so" not in names:
            raise ValueError(f"Unexpected APK architectures: {abis}")
        if any("gesture_mlp" in name or name.endswith("hand_landmarker.task") for name in names):
            raise ValueError("Retired ONNX/MediaPipe assets are still packaged")
        for asset, source in asset_sources.items():
            if hashlib.sha256(archive.read(asset)).hexdigest() != sha256(source):
                raise ValueError(f"APK and runtime model differ: {asset}")
        for name in ("float32", "int8"):
            filename = f"gesture_static_{name}_manifest.json"
            asset = "assets/flutter_assets/packages/static_gesture_runtime/assets/models/" + filename
            if archive.read(asset) != (models / filename).read_bytes():
                raise ValueError(f"APK and runtime manifest differ: {filename}")

    api = (package / "flutter/lib/static_gesture_runtime_api.dart").read_text(encoding="utf-8")
    if "export 'src/detector_adapter.dart';" not in api:
        raise ValueError("Final public detector adapter export is missing")
    adapter = (package / "flutter/lib/src/detector_adapter.dart").read_text(encoding="utf-8")
    if "bool modelInputMirrored = false" not in adapter:
        raise ValueError("Final handedness adapter update is missing")
    for app in (root / "flutter_gesture_tester", package / "flutter/example"):
        pubspec = (app / "pubspec.yaml").read_text(encoding="utf-8")
        if "skip_build: true" in pubspec:
            raise ValueError(f"Host-only OpenCV skip hook shipped in {app.name}")
    return {
        "passed": True,
        "payload_files_verified": len(sums),
        "runtime_files_verified": len(runtime_sums),
        "dynamic_detector_files_verified": len(detector_files),
        "apk_model_assets_matched": len(asset_sources),
        "android_abi": "arm64-v8a",
        "default_classifier": "qualified FP32 TFLite",
        "physical_ios_tested": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2))
