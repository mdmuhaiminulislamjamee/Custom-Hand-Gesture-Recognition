from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKSUMS = ROOT / "SHA256SUMS.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    checked = 0
    with CHECKSUMS.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            relative = Path(row["path"])
            candidate = (ROOT / relative).resolve()
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                failures.append(f"unsafe path: {relative}")
                continue
            if not candidate.is_file():
                failures.append(f"missing: {relative}")
                continue
            if candidate.stat().st_size != int(row["bytes"]):
                failures.append(f"size mismatch: {relative}")
                continue
            if sha256(candidate) != row["sha256"].lower():
                failures.append(f"SHA-256 mismatch: {relative}")
                continue
            checked += 1

    if failures:
        raise SystemExit("Package verification failed:\n" + "\n".join(failures))
    print(f"Package verification passed: {checked} files")


if __name__ == "__main__":
    main()
