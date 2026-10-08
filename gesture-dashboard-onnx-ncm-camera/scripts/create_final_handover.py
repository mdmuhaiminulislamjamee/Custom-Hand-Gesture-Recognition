"""Build the immutable app-team delivery from an explicit allowlist."""
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
from scripts.verify_final_handover import verify, verify_onnx

DEFAULT_VERSION = "v2026.10.08_ios_ncm_final_03"
MODEL_FILES = ["gesture_mlp_production.onnx", "gesture_mlp_onnx_metadata.json",
               "gesture_mobile_runtime_config.json", "hand_landmarker.task", "blaze_face_short_range.tflite"]
CACHE_FILES = ["gesture_artifact_manifest.json", "gesture_online_replay_cache.npz",
               "gesture_online_validation_cache.npz", "gesture_online_untouched_test_cache.npz"]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def copy(relative, destination):
    source = ROOT / relative
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Missing or linked release input: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)

def copy_tree(relative, destination, suffixes=None):
    for source in sorted((ROOT / relative).rglob("*")):
        if not source.is_file() or any(p.startswith(".") or p == "__pycache__" for p in source.relative_to(ROOT / relative).parts):
            continue
        if suffixes is not None and source.suffix not in suffixes:
            continue
        copy(source.relative_to(ROOT), destination / source.relative_to(ROOT / relative))

def archive(folder, output):
    with ZipFile(output, "x", ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, folder.name + "/" + p.relative_to(folder).as_posix())
    with ZipFile(output) as z:
        if z.testzip() is not None:
            raise ValueError(f"ZIP CRC failed: {output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    args = parser.parse_args()
    if not re.fullmatch(r"v\d{4}\.\d{2}\.\d{2}_ios_ncm_final_\d{2}", args.version):
        raise ValueError("Version must be vYYYY.MM.DD_ios_ncm_final_NN")
    release_root = ROOT / "handover_final"
    final = release_root / args.version
    zip_path = release_root / ("gesture-final-handover-" + args.version + ".zip")
    if final.exists() or zip_path.exists():
        raise FileExistsError("Release exists; choose a new version. Existing deliveries are never overwritten.")

    subprocess.run([sys.executable, "-m", "scripts.onnx_preflight"], cwd=ROOT, check=True)
    metadata = json.loads((ROOT / "models/gesture_mlp_onnx_metadata.json").read_text())
    fixtures = json.loads((ROOT / "swift/tests/parity_vectors.json").read_text())
    if sha(ROOT / "models/gesture_mlp_production.onnx") != fixtures["model_sha256"]:
        raise ValueError("Regenerate parity fixtures for the current model.")
    for name, digest in fixtures["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Regenerate parity fixtures after changing {name}")

    # Keep build staging under the workspace; no recursive cleanup/destructive move.
    stage = ROOT / ".runtime" / ("final-handover-build-" + uuid.uuid4().hex)
    package = stage / args.version
    desktop = stage / "desktop_reference"
    package.mkdir(parents=True)
    desktop.mkdir()
    for name in MODEL_FILES:
        copy("models/" + name, package / "ios/models" / name)
    copy_tree("swift", package / "ios/swift", {".swift", ".h", ".example", ".fragment", ".json", ".sh", ".md"})
    # Package-relative guide link instead of the repository-relative one.
    swift_readme = package / "ios/swift/README.md"
    swift_readme.write_text(swift_readme.read_text().replace("../handover_docs/IOS_INTEGRATION.md", "../../IOS_INTEGRATION.md"), encoding="utf-8")
    copy_tree("backend", package / "ios/reference/backend", {".py"})
    copy_tree("reference", package / "ios/reference/protocol", {".cpp", ".h"})
    copy_tree("handover_docs", package)
    # Repository guide sources are authoritative; replace the default version for new builds.
    for p in package.glob("*.md"):
        p.write_text(p.read_text(encoding="utf-8").replace(DEFAULT_VERSION,args.version), encoding="utf-8")
    copy("scripts/verify_final_handover.py", package / "verify_package.py")

    for directory, suffixes in [
        ("backend", {".py"}), ("scripts", {".py", ".ps1"}),
        ("research", {".py"}), ("app", {".ts", ".tsx", ".css", ".mjs"}),
        ("public", {".svg", ".png"}), ("reference", {".cpp", ".h"}),
        ("swift", {".swift", ".h", ".example", ".fragment", ".json", ".sh", ".md"}),
    ]:
        copy_tree(directory, desktop / directory, suffixes)
    for name in MODEL_FILES + CACHE_FILES:
        copy("models/" + name, desktop / "models" / name)
    for p in ROOT.glob("*.bat"):
        copy(p.name, desktop / p.name)
    for name in ["package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", "requirements.txt",
                 "requirements.lock.txt", "requirements-handover-checks.txt", "pytest.ini",
                 "tsconfig.json", "eslint.config.mjs", "next-env.d.ts", "next.config.ts",
                 "vite.config.ts", "vite.local.config.ts", ".env.example",
                 "DATA_COLLECTION.md", "HAND_DISTANCE.md", "MULTIVIEW_TRAINING.md"]:
        copy(name, desktop / name)
    copy("handover_docs/WINDOWS_RUN.md", desktop / "README.md")
    copy("handover_docs/MODEL_CARD.md", desktop / "MODEL_CARD.md")
    copy("handover_docs/VALIDATION.md", desktop / "HANDOVER_CHECKLIST.md")
    copy_tree("handover_docs", desktop / "handover_docs")
    copy_tree("examples", desktop / "examples", {".py"})
    # Minimal existing qualification input used by the shipped tests, no camera photos.
    copy("data/ten_gesture/hagrid_train.npz", desktop / "data/ten_gesture/hagrid_train.npz")
    copy("data/ten_gesture/sources.json", desktop / "data/ten_gesture/sources.json")
    for name in ["candidate_contract.json", "evaluation.json", "runtime_comparison.json"]:
        copy("artifacts/twelve_gesture/" + name, desktop / "artifacts/twelve_gesture" / name)
    archive(desktop, package / "desktop_reference.zip")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_hashes = {
        p.relative_to(ROOT).as_posix(): sha(p)
        for directory in ["backend", "swift"]
        for p in sorted((ROOT / directory).rglob("*"))
        if p.is_file() and p.suffix in {".py", ".swift"} and "__pycache__" not in p.parts
    }
    manifest = dict(
        schema_version=1, release=args.version, prepared_date="2026-10-08", timezone="Asia/Dhaka",
        source_base_commit=revision, source_snapshot_includes_uncommitted_handover_changes=True,
        model_sha256=metadata["onnx_sha256"], class_names=metadata["output_class_order"],
        gesture_to_action=json.loads((ROOT / "models/gesture_mobile_runtime_config.json").read_text())["gesture_to_action"],
        model_retrained_for_handover=False, mutable_online_adapter_included=False,
        native_swift_compiled=False, physical_ios_validated=False,
        frame_pipeline_port_complete=False, physical_ncm_ios_validated=False,
        geometry_fixture_count=len(fixtures["geometry_cases"]), onnx_fixture_count=len(fixtures["model_cases"]),
        source_sha256=source_hashes,
        ios_image_source="USB-NCM JLIP/JPEG only; built-in iOS camera excluded",
        scope="Runnable Windows NCM reference and native NCM-only iOS integration sources; Apple/device acceptance pending")
    (package / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    sums = {p.relative_to(package).as_posix(): dict(bytes=p.stat().st_size,sha256=sha(p))
            for p in sorted(package.rglob("*")) if p.is_file()}
    (package / "SHA256SUMS.json").write_text(json.dumps(sums,indent=2)+"\n",encoding="utf-8")
    report = verify(package)
    report.update(verify_onnx(package))
    release_root.mkdir(exist_ok=True)
    shutil.copytree(package, final)
    archive(final, zip_path)
    verify(final)
    # Outer hash is a companion file; the ZIP never contains its own digest.
    zip_path.with_suffix(".zip.sha256").write_text(sha(zip_path)+"  "+zip_path.name+"\n",encoding="utf-8")
    print(json.dumps(dict(directory=str(final), archive=str(zip_path), archive_bytes=zip_path.stat().st_size,
                         staged_desktop=str(desktop), **report), indent=2))

if __name__ == "__main__":
    main()
