"""Fetch only HaGRID's stop_inverted annotation members, not its image archives."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

from backend.geometry import landmarks_to_feature
from scripts.collect_v18_20_data import HAGRID_URL

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "stop_inverted"


class RangeArchive(io.RawIOBase):
    """Seekable HTTP range reader; downloads only the ZIP members requested."""

    def __init__(self, url: str):
        self.url, self.position = url, 0
        with self._request("bytes=-65536") as response:
            self.length = int(response.headers["Content-Range"].split("/")[-1])
            self.tail = response.read()
        self.tail_start = self.length - len(self.tail)

    def _request(self, byte_range: str):
        response = urllib.request.urlopen(
            urllib.request.Request(self.url, headers={"Range": byte_range}), timeout=90
        )
        if response.status != 206:
            response.close()
            raise ValueError("Annotation server must support byte ranges")
        return response

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.length if whence == 2 else 0)
        return self.position

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size <= 0:
            return b""
        if self.position >= self.tail_start:
            value = self.tail[self.position - self.tail_start:self.position - self.tail_start + size]
        else:
            with self._request(f"bytes={self.position}-{self.position + size - 1}") as response:
                value = response.read()
        if len(value) != size:
            raise ValueError("Truncated HTTP range response")
        self.position += len(value)
        return value


def collect():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if all((OUTPUT / f"hagrid_{split}.npz").exists() for split in ("train", "val", "test")):
        print("Using cached HaGRID Stop-inverted annotations", flush=True)
        return
    audit = {"source": "https://github.com/hukenovs/hagrid", "archive": HAGRID_URL,
             "source_class": "stop_inverted", "splits": {}}
    if (OUTPUT / "sources.json").exists():
        audit = json.loads((OUTPUT / "sources.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(RangeArchive(HAGRID_URL)) as archive:
        for split, limit in (("train", 3000), ("val", 600), ("test", 600)):
            destination = OUTPUT / f"hagrid_{split}.npz"
            member = f"annotations/{split}/stop_inverted.json"
            if destination.exists():
                print(f"Using cached {destination}", flush=True)
                continue
            print(f"Downloading {member}", flush=True)
            payload = archive.read(member)
            annotations = json.loads(payload)
            rows, per_person = [], {}
            for key in sorted(annotations, key=lambda value: hashlib.sha256(value.encode()).digest()):
                record = annotations[key]
                group = "hagrid:" + str(record.get("user_id", key))
                if per_person.get(group, 0) >= 2:
                    continue
                for hand, (label, landmarks) in enumerate(zip(record["labels"], record.get("hand_landmarks", []))):
                    if label != "stop_inverted" or landmarks is None:
                        continue
                    points = np.asarray(landmarks, dtype=np.float32)
                    if points.shape != (21, 2) or not np.isfinite(points).all():
                        continue
                    try:
                        features = landmarks_to_feature(points)
                    except ValueError:
                        continue
                    rows.append((points, features, group, f"{key}:{hand}"))
                    per_person[group] = per_person.get(group, 0) + 1
                    break
                if len(rows) >= limit:
                    break
            if not rows:
                raise ValueError(f"No valid source landmarks in {member}")
            np.savez_compressed(destination,
                                landmarks=np.asarray([row[0] for row in rows]),
                                X=np.asarray([row[1] for row in rows]),
                                group=np.asarray([row[2] for row in rows]),
                                source_id=np.asarray([row[3] for row in rows]),
                                source=np.full(len(rows), "hagrid_stop_inverted"))
            audit["splits"][split] = {"rows": len(rows), "participants": len(per_person),
                                      "member": member, "annotation_sha256": hashlib.sha256(payload).hexdigest()}
            print(f"{split}: {len(rows)} hands, {len(per_person)} people", flush=True)
            (OUTPUT / "sources.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            del annotations, payload
    if audit["splits"]:
        (OUTPUT / "sources.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    collect()
