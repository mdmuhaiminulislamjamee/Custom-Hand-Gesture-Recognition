from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from backend.config import CLASS_NAMES
from backend.geometry import landmarks_to_feature
from research.multiview import (
    MANIFEST_FIELDS,
    capture_subset,
    coverage_audit,
    direction_from_landmarks,
    load_capture_manifest,
    make_balanced_training_set,
    participant_split,
)


def directional_points(direction: str, offset: float = 0.0) -> np.ndarray:
    points = np.asarray(
        [
            [0.50, 0.72], [0.42, 0.64], [0.37, 0.57], [0.34, 0.52], [0.31, 0.48],
            [0.48, 0.58], [0.48, 0.45], [0.48, 0.31], [0.48, 0.16],
            [0.55, 0.57], [0.57, 0.55], [0.58, 0.58], [0.57, 0.62],
            [0.61, 0.59], [0.63, 0.58], [0.64, 0.62], [0.62, 0.66],
            [0.66, 0.62], [0.68, 0.62], [0.69, 0.66], [0.67, 0.70],
        ],
        dtype=np.float32,
    )
    vector = points[8] - points[5]
    target = {
        "up": np.asarray([0.0, -1.0]),
        "down": np.asarray([0.0, 1.0]),
        "left": np.asarray([1.0, 0.0]),
        "right": np.asarray([-1.0, 0.0]),
    }[direction]
    angle = np.arctan2(target[1], target[0]) - np.arctan2(vector[1], vector[0])
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    points = (points - points[0]) @ rotation.T + points[0]
    points[4, 0] += offset
    return points.astype(np.float32)


def write_manifest(root, rows):
    path = root / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def manifest_row(root, sample_id, participant, split, label, *, offset=0.0):
    points = directional_points(label, offset)
    feature = landmarks_to_feature(points)
    relative = f"images/{sample_id}.jpg"
    image = root / relative
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"jpeg-placeholder")
    return {
        "sample_id": sample_id,
        "image_path": relative,
        "created_at_utc": "2026-09-08T00:00:00+00:00",
        "participant_id": participant,
        "session_id": "session-1",
        "split": split,
        "label": label,
        "view": "front",
        "source": "ncm",
        "handedness": "Right",
        "width": "640",
        "height": "480",
        "input_luma_median": "100",
        "sharpness": "50",
        "pose_score": "0.9",
        "axis_dominance": "1.0",
        "captured_direction": label,
        "feature_vector_json": json.dumps(feature.tolist()),
        "landmarks_json": json.dumps(points.tolist()),
    }


def test_participant_split_is_stable_and_valid():
    assert participant_split("person-001") == participant_split("person-001")
    assert participant_split("person-001") in {"train", "val", "test"}
    with pytest.raises(ValueError, match="cannot be empty"):
        participant_split(" ")


@pytest.mark.parametrize("label", ["left", "right", "up", "down"])
def test_direction_from_landmarks_matches_ncm_semantics(label):
    actual, dominance = direction_from_landmarks(directional_points(label))
    assert actual == label
    assert dominance == pytest.approx(1.0)


def test_manifest_rejects_participant_split_leakage(tmp_path):
    rows = [
        manifest_row(tmp_path, "a", "person", "train", "up"),
        manifest_row(tmp_path, "b", "person", "test", "up", offset=0.001),
    ]
    write_manifest(tmp_path, rows)
    with pytest.raises(ValueError, match="participant leakage"):
        load_capture_manifest(tmp_path)


def test_manifest_loads_aligned_features_and_subsets(tmp_path):
    rows = [
        manifest_row(tmp_path, "a", "person-a", "train", "up"),
        manifest_row(tmp_path, "b", "person-b", "test", "left"),
    ]
    write_manifest(tmp_path, rows)
    loaded = load_capture_manifest(tmp_path)
    assert loaded["X"].shape == (2, 76)
    assert loaded["landmarks"].shape == (2, 21, 2)
    assert capture_subset(loaded, "test")["label"].tolist() == ["left"]


def test_coverage_requires_each_direction_in_each_split():
    data = {
        "label": np.asarray(["up", "up", "left"]),
        "split": np.asarray(["train", "train", "test"]),
        "participant_id": np.asarray(["a", "b", "c"]),
        "view": np.asarray(["front", "casual", "front"]),
    }
    audit = coverage_audit(
        data,
        requirements={"train": (1, 1), "val": (1, 1), "test": (1, 1)},
    )
    assert audit["passed"] is False
    assert audit["splits"]["train"]["up"]["passed"] is True
    assert audit["splits"]["val"]["up"]["passed"] is False


def test_coverage_enforces_viewpoint_diversity():
    labels = []
    splits = []
    participants = []
    views = []
    for split in ("train", "val", "test"):
        for label in ("left", "right", "up", "down"):
            labels.append(label)
            splits.append(split)
            participants.append(f"{split}-person")
            views.append("front")
    data = {
        "label": np.asarray(labels),
        "split": np.asarray(splits),
        "participant_id": np.asarray(participants),
        "view": np.asarray(views),
    }
    audit = coverage_audit(
        data,
        requirements={"train": (1, 1, 2), "val": (1, 1, 2), "test": (1, 1, 2)},
    )
    assert audit["passed"] is False
    assert audit["splits"]["test"]["down"]["minimum_views"] == 2


def test_balancing_uses_captured_rows_without_changing_class_counts():
    base_x = []
    base_y = []
    for label in range(9):
        for row in range(12):
            base_x.append(np.full(76, label + row / 1000, dtype=np.float32))
            base_y.append(label)
    base = {
        "X": np.asarray(base_x),
        "y": np.asarray(base_y),
        "parent_id": np.asarray([f"base-{i}" for i in range(len(base_y))]),
    }
    labels = ["left", "right", "up", "down"]
    captured_points = np.asarray([directional_points(label) for label in labels])
    captured = {
        "X": np.asarray([landmarks_to_feature(points) for points in captured_points]),
        "landmarks": captured_points,
        "y": np.asarray([CLASS_NAMES.index(label) for label in labels]),
        "sample_id": np.asarray([f"capture-{label}" for label in labels]),
    }
    balanced, audit = make_balanced_training_set(
        base,
        captured,
        per_class=8,
        capture_share=0.5,
        maximum_capture_repeats=2,
        seed=7,
    )
    assert balanced["X"].shape == (72, 76)
    assert np.bincount(balanced["y"], minlength=9).tolist() == [8] * 9
    assert int(np.sum(balanced["origin"] == "ncm_multiview")) == 8
    left = next(row for row in audit["classes"] if row["label"] == "left")
    assert left["captured_training_rows"] == 2
