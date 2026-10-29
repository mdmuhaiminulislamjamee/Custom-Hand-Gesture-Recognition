"""Assemble the integrity-checked static runtime handover."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "static_gesture_runtime"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "handover_static/static_gesture_runtime_20261029_fix1.zip")
    args = parser.parse_args()
    reference = json.loads((PACKAGE / "validation/python_reference.json").read_text(encoding="utf-8"))
    small = {key: reference[key] for key in ("schema_version", "model_sha256", "class_names", "feature_count", "probability_tolerance", "model_cases")}
    (PACKAGE / "validation/model_parity.json").write_text(json.dumps(small, indent=2)+"\n", encoding="utf-8")
    tools_dir = PACKAGE / "tools"
    tools_dir.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "scripts/verify_static_runtime.py", tools_dir / "verify_static_runtime.py")
    shutil.copyfile(ROOT / "requirements-static-conversion.txt", tools_dir / "requirements-verification.txt")
    ignored = {".dart_tool", "build", ".git", "__pycache__", ".idea"}
    files = sorted(path for path in PACKAGE.rglob("*") if path.is_file() and not ignored.intersection(path.relative_to(PACKAGE).parts) and path.name not in {"SHA256SUMS.json", ".flutter-plugins-dependencies"})
    hashes = {path.relative_to(PACKAGE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    checksum = PACKAGE / "SHA256SUMS.json"
    checksum.write_text(json.dumps(hashes, indent=2)+"\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for path in files + [checksum]:
            archive.write(path, "static_gesture_runtime/"+path.relative_to(PACKAGE).as_posix())
    result = {"archive": str(args.output.relative_to(ROOT)), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "bytes": args.output.stat().st_size, "files": len(files)+1, "physical_ios_tested": False}
    args.output.with_suffix(args.output.suffix+".sha256").write_text(result["sha256"]+"  "+args.output.name+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
