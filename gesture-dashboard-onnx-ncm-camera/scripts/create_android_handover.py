"""Build an immutable Android NCM handover from an explicit source allowlist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.verify_android_handover import MODELS, verify, verify_onnx

DEFAULT_VERSION = "v2026.10.08_android_ncm_03"
KOTLIN = ["GestureNcmClient.kt", "NcmNetworkSelector.kt", "GestureHandLandmarker.kt",
          "GestureFeatureExtractor.kt", "GestureOnnxClassifier.kt", "GestureFrameAnalyzer.kt",
          "GestureParity.kt"]
SWIFT_REFERENCE = ["GestureFeatureExtractor.swift", "GestureGeometryResolver.swift",
                   "GestureTemporalGate.swift", "GestureActionLatch.swift",
                   "GestureDecisionPipeline.swift"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(relative: str, destination: Path) -> None:
    source = ROOT / relative
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Missing or linked release input: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def archive(folder: Path, target: Path) -> None:
    with ZipFile(target, "x", ZIP_DEFLATED, compresslevel=9) as archive_file:
        for file in sorted(folder.rglob("*")):
            if file.is_file():
                archive_file.write(file, folder.name + "/" + file.relative_to(folder).as_posix())
    with ZipFile(target) as archive_file:
        if archive_file.testzip() is not None:
            raise ValueError(f"ZIP CRC failed: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    args = parser.parse_args()
    if not re.fullmatch(r"v\d{4}\.\d{2}\.\d{2}_android_ncm_\d{2}", args.version):
        raise ValueError("Version must be vYYYY.MM.DD_android_ncm_NN")
    release_root = ROOT / "handover_android"
    final = release_root / args.version
    zip_path = release_root / ("gesture-android-handover-" + args.version + ".zip")
    if final.exists() or zip_path.exists():
        raise FileExistsError("Release exists; choose a new version. Existing deliveries are never overwritten.")

    subprocess.run([sys.executable, "-m", "scripts.onnx_preflight"], cwd=ROOT, check=True)
    stage = ROOT / ".runtime" / ("android-handover-build-" + uuid.uuid4().hex)
    package = stage / args.version
    package.mkdir(parents=True)
    for name in MODELS:
        copy("models/" + name, package / "android/models" / name)
    for name in KOTLIN:
        copy("android_handover_src/kotlin/" + name, package / "android/kotlin" / name)
    copy("android_handover_src/AndroidManifest.fragment.xml", package / "android/AndroidManifest.fragment.xml")
    copy("android_handover_src/app-build.gradle.kts.fragment", package / "android/app-build.gradle.kts.fragment")
    copy("swift/tests/parity_vectors.json", package / "android/tests/parity_vectors.json")
    for name in ["README.md", "ANDROID_INTEGRATION.md", "NCM_PROTOCOL.md", "VALIDATION.md"]:
        copy("android_handover_src/" + name, package / name)
    copy("handover_docs/MODEL_CARD.md", package / "MODEL_CARD.md")
    model_card = package / "MODEL_CARD.md"
    model_card.write_text(model_card.read_text(encoding="utf-8").replace("iPhone/NCM", "Android/NCM"), encoding="utf-8")
    for guide in package.glob("*.md"):
        guide.write_text(guide.read_text(encoding="utf-8").replace(DEFAULT_VERSION, args.version), encoding="utf-8")
    copy("scripts/verify_android_handover.py", package / "verify_package.py")
    for source in sorted((ROOT / "backend").rglob("*.py")):
        if "__pycache__" not in source.parts:
            copy(source.relative_to(ROOT).as_posix(), package / "android/reference/backend" / source.relative_to(ROOT / "backend"))
    for name in SWIFT_REFERENCE:
        copy("swift/" + name, package / "android/reference/swift" / name)
    copy("reference/NcmCameraTester.cpp", package / "android/reference/protocol/NcmCameraTester.cpp")
    copy("handover_final/v2026.10.08_ios_ncm_final_03/desktop_reference.zip", package / "desktop_reference.zip")

    model_metadata = json.loads((ROOT / "models/gesture_mlp_onnx_metadata.json").read_text())
    source_hashes = {
        p.relative_to(ROOT).as_posix(): digest(p)
        for p in sorted((ROOT / "android_handover_src").rglob("*")) if p.is_file()
    }
    source_hashes.update({"scripts/create_android_handover.py": digest(ROOT / "scripts/create_android_handover.py"),
                          "scripts/verify_android_handover.py": digest(ROOT / "scripts/verify_android_handover.py")})
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = dict(
        schema_version=1, release=args.version, prepared_date="2026-10-08", timezone="Asia/Dhaka",
        source_base_commit=revision, source_snapshot_includes_uncommitted_handover_changes=True,
        model_sha256=model_metadata["onnx_sha256"], class_names=model_metadata["output_class_order"],
        ios_reference_release="v2026.10.08_ios_ncm_final_03",
        android_min_sdk=24, android_kotlin_compiled=False, physical_android_ncm_validated=False,
        feature_onnx_android_parity_validated=False, full_android_decision_port_complete=False,
        android_image_source="USB-NCM JLIP/JPEG only; built-in Android camera excluded",
        source_sha256=source_hashes,
        scope="Native Android NCM/MediaPipe/ONNX diagnostic integration sources, complete model and behavior references; Android action path and device acceptance pending",
    )
    (package / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sums = {p.relative_to(package).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(package.rglob("*")) if p.is_file()}
    (package / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2) + "\n", encoding="utf-8")
    report = verify(package)
    report.update(verify_onnx(package))
    release_root.mkdir(exist_ok=True)
    shutil.copytree(package, final)
    archive(final, zip_path)
    verify(final)
    zip_path.with_suffix(".zip.sha256").write_text(digest(zip_path) + "  " + zip_path.name + "\n", encoding="utf-8")
    print(json.dumps({"directory": str(final), "archive": str(zip_path),
                      "archive_bytes": zip_path.stat().st_size, **report}, indent=2))


if __name__ == "__main__":
    main()
