from __future__ import annotations

import base64
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np
from PIL import Image

from .config import FEEDBACK_DIRECTORY, MODELS_DIRECTORY, RuntimeConfig


@dataclass(slots=True)
class FeedbackRecord:
    actual_label: str
    predicted_label: str
    runtime_action: str
    confidence: float
    model_name: str
    probabilities: dict[str, float]
    feature_vector: list[float] | None = None
    landmarks: list[list[float]] | None = None
    note: str = ""
    snapshot_data_url: str | None = None


class FeedbackStore:
    fields = [
        "feedback_id", "created_at_utc", "actual_label", "predicted_label",
        "runtime_action", "confidence", "model_name", "note", "image_path",
        "probabilities_json", "feature_vector_json", "landmarks_json",
    ]

    def __init__(self, config: RuntimeConfig, directory: Path = FEEDBACK_DIRECTORY):
        self.config = config
        self.directory = directory
        self.images_directory = directory / "images"
        self.csv_path = directory / "misclassification_log.csv"
        self._lock = threading.Lock()
        self.images_directory.mkdir(parents=True, exist_ok=True)

    def save(self, feedback: FeedbackRecord) -> dict[str, Any]:
        if feedback.actual_label not in self.config.feedback_labels:
            raise ValueError(f"Unknown actual label: {feedback.actual_label}")
        if feedback.predicted_label not in self.config.feedback_labels:
            raise ValueError(f"Unknown predicted label: {feedback.predicted_label}")
        if not np.isfinite(float(feedback.confidence)):
            raise ValueError("Feedback confidence must be finite.")
        unknown_probability_names = set(feedback.probabilities) - set(self.config.class_names)
        if unknown_probability_names:
            raise ValueError("Feedback contains unknown probability labels.")
        probability_values = np.asarray(list(feedback.probabilities.values()), dtype=np.float64)
        if len(probability_values) and (
            not np.isfinite(probability_values).all()
            or (probability_values < 0).any()
            or (probability_values > 1).any()
        ):
            raise ValueError("Feedback probabilities must be finite values from 0 to 1.")
        if feedback.feature_vector is not None:
            feature = np.asarray(feedback.feature_vector, dtype=np.float64)
            if feature.shape != (len(self.config.feature_names),) or not np.isfinite(feature).all():
                raise ValueError("Feedback features must follow the finite 76-D contract.")
        if feedback.landmarks is not None:
            landmarks = np.asarray(feedback.landmarks, dtype=np.float64)
            if landmarks.shape != (21, 2) or not np.isfinite(landmarks).all():
                raise ValueError("Feedback landmarks must be a finite 21x2 array.")
        timestamp = datetime.now(timezone.utc)
        fingerprint = hashlib.sha256(
            f"{timestamp.isoformat()}:{feedback.actual_label}:{feedback.predicted_label}".encode()
        ).hexdigest()[:16]
        image_path = ""
        if feedback.snapshot_data_url:
            prefix, separator, payload = feedback.snapshot_data_url.partition(",")
            if separator and "base64" in prefix:
                image_bytes = base64.b64decode(payload, validate=True)
                if len(image_bytes) > 5_000_000:
                    raise ValueError("Feedback image is larger than 5 MB.")
                try:
                    with Image.open(BytesIO(image_bytes)) as source:
                        source.verify()
                    with Image.open(BytesIO(image_bytes)) as source:
                        if source.width * source.height > 16_000_000:
                            raise ValueError("Feedback image dimensions are too large.")
                        normalized = source.convert("RGB")
                        output = BytesIO()
                        normalized.save(output, format="JPEG", quality=90, optimize=True)
                        image_bytes = output.getvalue()
                except (OSError, Image.DecompressionBombError) as error:
                    raise ValueError("Feedback snapshot is not a safe image.") from error
                saved_image = self.images_directory / f"{fingerprint}.jpg"
                saved_image.write_bytes(image_bytes)
                image_path = str(saved_image)
        row = {
            "feedback_id": fingerprint,
            "created_at_utc": timestamp.isoformat(),
            "actual_label": feedback.actual_label,
            "predicted_label": feedback.predicted_label,
            "runtime_action": feedback.runtime_action,
            "confidence": f"{float(feedback.confidence):.8f}",
            "model_name": feedback.model_name,
            "note": feedback.note.strip()[:500],
            "image_path": image_path,
            "probabilities_json": json.dumps(feedback.probabilities, separators=(",", ":")),
            "feature_vector_json": json.dumps(feedback.feature_vector or [], separators=(",", ":")),
            "landmarks_json": json.dumps(feedback.landmarks or [], separators=(",", ":")),
        }
        with self._lock:
            write_header = not self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        return {"status": "saved", "feedback_id": fingerprint, "image_saved": bool(image_path)}

    def summary(self) -> dict[str, Any]:
        if not self.csv_path.exists():
            return {"count": 0, "recent": [], "path": str(self.csv_path)}
        with self._lock, self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return {"count": len(rows), "recent": rows[-20:][::-1], "path": str(self.csv_path)}


def artifact_status(models_directory: Path = MODELS_DIRECTORY) -> list[dict[str, Any]]:
    expected = [
        ("MediaPipe hand detector", "hand_landmarker.task", True),
        ("Runtime configuration", "gesture_mobile_runtime_config.json", True),
        ("Qualified ONNX classifier", "gesture_mlp_production.onnx", True),
        ("ONNX parity metadata", "gesture_mlp_onnx_metadata.json", True),
        ("Safe Learn replay cache", "gesture_online_replay_cache.npz", True),
        ("Safe Learn validation cache", "gesture_online_validation_cache.npz", True),
        ("Untouched release test cache", "gesture_online_untouched_test_cache.npz", True),
        ("Artifact integrity manifest", "gesture_artifact_manifest.json", True),
        ("Evaluation evidence", "*.csv", False),
    ]
    rows = []
    for label, pattern, individually_required in expected:
        matches = list(models_directory.glob(pattern))
        rows.append({
            "label": label,
            "pattern": pattern,
            "present": bool(matches),
            "required": individually_required,
            "files": [item.name for item in matches],
            "bytes": sum(item.stat().st_size for item in matches),
        })
    return rows


def load_metric_files(models_directory: Path = MODELS_DIRECTORY) -> dict[str, Any]:
    result: dict[str, Any] = {"files": [], "rows": {}}
    for path in sorted(models_directory.glob("*.csv")):
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error):
            continue
        result["files"].append(path.name)
        result["rows"][path.name] = rows[:500]
    return result
