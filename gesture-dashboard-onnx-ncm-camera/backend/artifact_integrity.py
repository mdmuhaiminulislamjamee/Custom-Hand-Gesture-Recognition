from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

from .config import CLASS_NAMES, FEATURE_NAMES, MODELS_DIRECTORY


MANIFEST_NAME = "gesture_artifact_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
RUNTIME_PATTERNS = (
    "hand_landmarker.task",
    "blaze_face_short_range.tflite",
    "gesture_mobile_runtime_config.json",
    "gesture_mlp_production.onnx",
    "gesture_mlp_onnx_metadata.json",
    "gesture_online_replay_cache.npz",
    "gesture_online_validation_cache.npz",
    "gesture_online_untouched_test_cache.npz",
    "*.csv",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(destination: Path, payload: dict[str, Any]) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


class ArtifactRegistry:
    """Hashes the exact runtime artifacts before any trusted model is loaded."""

    def __init__(self, models_directory: Path = MODELS_DIRECTORY):
        self.models_directory = Path(models_directory)
        self.manifest_path = self.models_directory / MANIFEST_NAME
        self._lock = threading.RLock()

    def _runtime_files(self) -> list[Path]:
        paths: set[Path] = set()
        for pattern in RUNTIME_PATTERNS:
            paths.update(
                path for path in self.models_directory.glob(pattern) if path.is_file()
            )
        return sorted(paths, key=lambda item: item.name.lower())

    def build(self) -> dict[str, Any]:
        with self._lock:
            self.models_directory.mkdir(parents=True, exist_ok=True)
            artifacts = {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in self._runtime_files()
            }
            payload = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "class_names": CLASS_NAMES,
                "feature_count": len(FEATURE_NAMES),
                "artifacts": artifacts,
            }
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            payload["release_fingerprint"] = hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()
            _atomic_json(self.manifest_path, payload)
            return payload

    def load(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def verify(self) -> dict[str, Any]:
        with self._lock:
            manifest = self.load()
            if manifest is None:
                return {
                    "verified": False,
                    "status": "manifest_missing",
                    "manifest": str(self.manifest_path),
                    "release_fingerprint": None,
                    "checked_files": 0,
                    "errors": ["Trusted artifact manifest is missing or unreadable."],
                }
            errors: list[str] = []
            if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                errors.append("Unsupported artifact manifest schema version.")
            if manifest.get("class_names") != CLASS_NAMES:
                errors.append("Manifest class order does not match the ten-command contract.")
            if manifest.get("feature_count") != len(FEATURE_NAMES):
                errors.append("Manifest feature count does not match the 76-D contract.")
            entries = manifest.get("artifacts")
            if not isinstance(entries, dict):
                entries = {}
                errors.append("Manifest artifact table is invalid.")
            for name, expected in entries.items():
                path = self.models_directory / name
                if path.parent != self.models_directory.resolve():
                    errors.append(f"Unsafe artifact name in manifest: {name}")
                    continue
                if not path.is_file():
                    errors.append(f"Artifact is missing: {name}")
                    continue
                try:
                    size = path.stat().st_size
                    digest = sha256_file(path)
                except OSError as error:
                    errors.append(f"Could not verify {name}: {error}")
                    continue
                if size != int(expected.get("bytes", -1)):
                    errors.append(f"Artifact size changed: {name}")
                if digest != expected.get("sha256"):
                    errors.append(f"Artifact hash changed: {name}")

            names = set(entries)
            required_base = {
                "hand_landmarker.task",
                "gesture_mobile_runtime_config.json",
                "gesture_mlp_production.onnx",
                "gesture_mlp_onnx_metadata.json",
                "gesture_online_replay_cache.npz",
                "gesture_online_validation_cache.npz",
                "gesture_online_untouched_test_cache.npz",
            }
            for name in sorted(required_base - names):
                errors.append(f"Required runtime artifact is not registered: {name}")
            metadata_path = self.models_directory / "gesture_mlp_onnx_metadata.json"
            model_path = self.models_directory / "gesture_mlp_production.onnx"
            if metadata_path.is_file() and model_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata.get("format") != "ONNX":
                        errors.append("Classifier metadata does not declare ONNX format.")
                    if metadata.get("output_class_order") != CLASS_NAMES:
                        errors.append("ONNX class order differs from the ten-command contract.")
                    if metadata.get("feature_names") != FEATURE_NAMES:
                        errors.append("ONNX feature order differs from the 76-D contract.")
                    if metadata.get("onnx_file") != model_path.name:
                        errors.append("ONNX metadata points to a different model file.")
                    if metadata.get("onnx_sha256") != sha256_file(model_path):
                        errors.append("ONNX model hash differs from qualification metadata.")
                    if (metadata.get("parity") or {}).get("passed") is not True:
                        errors.append("ONNX parity qualification is not marked as passed.")
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    errors.append(f"ONNX metadata could not be validated: {error}")
            return {
                "verified": not errors,
                "status": "verified" if not errors else "verification_failed",
                "manifest": str(self.manifest_path),
                "release_fingerprint": manifest.get("release_fingerprint"),
                "created_utc": manifest.get("created_utc"),
                "checked_files": len(entries),
                "errors": errors,
            }
