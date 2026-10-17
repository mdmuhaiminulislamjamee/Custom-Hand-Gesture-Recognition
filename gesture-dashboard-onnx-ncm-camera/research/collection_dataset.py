"""Read the dashboard's portable, human-reviewed image/JSON dataset.

Missing or contradictory landmarks remain raw image cases for re-extraction;
they are never converted into made-up feature rows or automatically relabeled.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from backend.config import CLASS_NAMES, DISABLED_GESTURES
from backend.data_collection import collection_plan, identifier
from backend.geometry import landmarks_to_feature
from research.multiview import _empty_capture, direction_from_landmarks, participant_split


def load_collection_dataset(dataset_root: Path):
    root = Path(dataset_root).resolve()
    descriptor = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if descriptor.get("schema_version") != 2 or descriptor.get("type") != "gesture_image_collection":
        raise ValueError("Unsupported dashboard collection schema")
    labels = [*CLASS_NAMES, "no_gesture"]
    prompts = {s["id"]: s for s in collection_plan()}
    rows, skipped, excluded, seen, relabeled = [], [], [], {}, []
    image_count = 0
    for sidecar in sorted((root / "participants").glob("*/*/*/*/*.json")):
        if not sidecar.resolve().is_relative_to(root):
            raise ValueError("Metadata path escapes dataset")
        row = json.loads(sidecar.read_text(encoding="utf-8"))
        image_count += 1
        if row.get("schema_version") != 2 or row.get("reviewed") is not True:
            raise ValueError(f"Unreviewed or invalid sample: {sidecar.name}")
        participant = identifier(row["participant_id"], "Participant")
        session = identifier(row["session_id"], "Session")
        prompt = prompts.get(row.get("step_id"))
        retired = row.get("label") in DISABLED_GESTURES
        if retired and row.get("step_id") == f"dorsal/{row.get('view')}":
            # Preserve the reviewed source image and its provenance. Apply the
            # current label policy only to the derived training row.
            prompt = prompts.get(f"no_gesture/dorsal_{row.get('view')}")
            if prompt is not None:
                prompt = {**prompt, "label": row["label"], "view": row["view"]}
        if prompt is None or row.get("label") != prompt["label"] or row.get("view") != prompt["view"]:
            raise ValueError(f"Label/prompt mismatch: {sidecar.name}")
        relative = Path(row["image_path"])
        image = (root / relative).resolve()
        if relative.is_absolute() or not image.is_relative_to(root) or image != sidecar.with_suffix(".jpg").resolve():
            raise ValueError("Image path escapes dataset or does not match its metadata")
        expected = root / "participants" / participant / row["label"] / row["view"] / session / sidecar.name
        if expected != sidecar:
            raise ValueError("Participant/session metadata does not match folders")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest != row.get("image_sha256") or row.get("sample_id") != digest[:24]:
            raise ValueError(f"Image checksum mismatch: {sidecar.name}")
        key = relative.as_posix()
        provenance = row.get("video_provenance") or {}
        clip_id = provenance.get("clip_id")
        if clip_id and re.fullmatch(r"[a-f0-9]{32}", str(clip_id)):
            clip_path = root / "videos" / "clips" / clip_id / "clip.json"
            if clip_path.is_file():
                clip = json.loads(clip_path.read_text(encoding="utf-8"))
                timestamp = provenance.get("timestamp_seconds")
                current_segment = next((segment for segment in clip.get("segments", [])
                    if isinstance(timestamp, (float, int))
                    and segment["start"] <= timestamp < segment["end"]), None)
                current_prompt = (None if not current_segment else
                    f"no_gesture/{current_segment.get('negative_reason', '')}"
                    if current_segment["label"] == "no_gesture" else
                    f"{current_segment['label']}/casual")
                expected_prompt = row["step_id"]
                if retired:
                    expected_prompt = f"no_gesture/dorsal_{row['view']}"
                    if current_prompt and current_prompt.startswith("dorsal/"):
                        current_prompt = f"no_gesture/dorsal_{current_prompt.split('/')[1]}"
                if not clip.get("reviewed") or current_prompt != expected_prompt:
                    excluded.append(dict(image_path=key, training_use="superseded_video_annotation",
                                         reason="video_label_changed_reextract_reviewed_spans"))
                    continue
        if digest in seen:
            previous = seen[digest]
            if previous != (participant, row["label"]):
                raise ValueError("Identical image assigned to different participants or labels; review leakage")
            skipped.append(dict(image_path=key, reason="duplicate_image"))
            continue
        seen[digest] = (participant, row["label"])
        if row["label"] == "fist" and row.get("finger_orientation") == "away_from_camera":
            excluded.append(dict(image_path=key, training_use="context_only",
                                 reason="reverse_facing_fist_requires_orientation_review"))
            continue
        if prompt["training_use"] != "hand_classifier":
            excluded.append(dict(image_path=key, training_use=prompt["training_use"]))
            continue
        if row.get("landmarks") is None or row.get("feature_vector") is None:
            skipped.append(dict(image_path=key, reason="needs_landmark_extraction"))
            continue
        points = np.asarray(row["landmarks"], dtype=np.float32)
        feature = np.asarray(row["feature_vector"], dtype=np.float32)
        if points.shape != (21, 2) or feature.shape != (76,) or not np.isfinite(points).all() or not np.isfinite(feature).all():
            raise ValueError(f"Invalid landmark/feature dimensions: {sidecar.name}")
        try:
            reconstructed = landmarks_to_feature(points)
        except ValueError:
            skipped.append(dict(image_path=key, reason="degenerate_landmarks"))
            continue
        # Runtime landmarks are rounded to six decimals before serialization.
        if not np.allclose(reconstructed, feature, atol=.003, rtol=.003):
            skipped.append(dict(image_path=key, reason="needs_feature_reextraction"))
            continue
        if row["label"] in CLASS_NAMES[:4]:
            direction, _ = direction_from_landmarks(points)
            if direction != row["label"]:
                skipped.append(dict(image_path=key, reason="review_direction_or_redetect_landmarks"))
                continue
        effective_label = "no_gesture" if retired else row["label"]
        if retired:
            relabeled.append(dict(image_path=key, original_label=row["label"], label=effective_label,
                                  reason="retired_dorsal_hard_negative"))
        rows.append(dict(X=reconstructed, landmarks=points, y=labels.index(effective_label),
                         sample_id=digest[:24], participant_id=participant, session_id=session,
                         split=participant_split(participant), label=effective_label,
                         view=row["view"], image_path=key))
    data = _empty_capture()
    if rows:
        data = {k: np.asarray([r[k] for r in rows], dtype=str if v.dtype.kind == 'U' else v.dtype)
                for k, v in data.items()}
    return data, dict(schema_version=2, raw_images=image_count, usable_feature_rows=len(rows),
                      raw_cases_for_review=skipped, excluded_by_design=excluded,
                      policy_relabels=relabeled,
                      split_policy="Deterministic participant-level 70/15/15; audit coverage before training",
                      note="Context and detector cases remain reviewed raw images; the single-hand landmark classifier trains only on hand-shape cases with valid landmarks.")
