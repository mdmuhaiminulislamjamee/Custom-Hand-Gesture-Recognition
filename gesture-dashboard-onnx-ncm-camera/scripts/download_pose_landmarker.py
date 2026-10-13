"""Install the pinned body-context model from Google's official model store."""
import hashlib
from pathlib import Path
import sys
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from backend.body_guard import POSE_MODEL_NAME, POSE_MODEL_SHA256

URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def main():
    path = PROJECT_ROOT / "models" / POSE_MODEL_NAME
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == POSE_MODEL_SHA256:
        print("Body pose model already verified.")
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read(8_000_000)
    if hashlib.sha256(data).hexdigest() != POSE_MODEL_SHA256:
        raise RuntimeError("Body pose model download checksum mismatch.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    temporary.write_bytes(data)
    temporary.replace(path)
    print(f"Installed {path.name} ({len(data)} bytes).")


if __name__ == "__main__":
    main()
