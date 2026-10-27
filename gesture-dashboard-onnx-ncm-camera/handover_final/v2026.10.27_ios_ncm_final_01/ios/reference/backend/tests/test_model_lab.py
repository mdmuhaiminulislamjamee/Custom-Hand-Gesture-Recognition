from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from backend.config import CLASS_NAMES
from backend.model_lab import ModelLab
from research.lab_training import LABELS, audit_dataset, check_source_splits, mix_training


def test_mixing_preserves_active_commands_and_turns_dorsal_into_rejection():
    labels = np.repeat(np.arange(len(LABELS)), 6)
    base = {"X": np.tile(np.arange(76, dtype=np.float32), (len(labels), 1)), "y": labels}
    captured = {"X": np.ones((2, 76), dtype=np.float32), "y": np.array([8, len(CLASS_NAMES)])}
    training, counts = mix_training(base, captured, per_class=40)
    assert training["X"].shape == (40 * (len(LABELS) - 1), 76)
    expected = np.repeat(40, len(LABELS))
    expected[CLASS_NAMES.index("dorsal")] = 0
    assert np.array_equal(np.bincount(training["y"]), expected)
    assert "dorsal" not in counts
    assert counts["fist"]["captured_rows"] == 4
    assert counts["no_gesture"]["captured_rows"] == 4
    assert counts["peace"]["public_rows"] == 40
    assert counts["rock"]["public_rows"] == 40


def test_audit_empty_dataset_reports_missing_people_and_classes(tmp_path):
    from backend.data_collection import CollectionStore
    CollectionStore(tmp_path).initialize()
    data, report = audit_dataset(tmp_path)
    assert len(data["y"]) == 0
    assert report["can_train"] is False
    assert report["coverage_passed"] is False
    assert report["class_names"] == LABELS
    assert report["coverage"]["test"]["rows"] == 0


def test_copied_public_frames_cannot_leak_into_camera_evaluation():
    public = {split: {"X": np.full((1, 76), i, np.float32)}
              for i, split in enumerate(("train", "val", "test"))}
    captured = {"val": {"X": np.zeros((1, 76), np.float32)}}
    with pytest.raises(ValueError, match="Captured val features overlap public train"):
        check_source_splits(public, captured)
    check_source_splits(public, {"train": {"X": np.zeros((1, 76), np.float32)}})


@pytest.mark.parametrize("bad", [None, "../models", "not-a-job", "f" * 31])
def test_model_job_paths_are_bounded(tmp_path, bad):
    lab = ModelLab(tmp_path / "dataset", tmp_path / "jobs")
    with pytest.raises(ValueError, match="Invalid model job"):
        lab.get(bad)


def test_interrupted_jobs_fail_after_restart_and_cannot_be_downloaded(tmp_path):
    directory = tmp_path / "jobs" / ("a" * 32)
    directory.mkdir(parents=True)
    (directory / "job.json").write_text(json.dumps({"id": directory.name, "kind": "train",
        "status": "running", "logs": [], "result": None}), encoding="utf-8")
    (directory / "candidate.onnx").write_bytes(b"unfinished")
    lab = ModelLab(tmp_path / "dataset", directory.parent)
    assert lab.get(directory.name)["status"] == "failed"
    assert not lab.get(directory.name)["candidate_available"]
    with pytest.raises(ValueError, match="completed"):
        lab.candidate_path(directory.name)


def test_shutdown_cancels_queued_jobs_before_their_worker_can_start(tmp_path, monkeypatch):
    import backend.model_lab as module
    from types import SimpleNamespace
    monkeypatch.setattr(module.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    lab = ModelLab(tmp_path / "dataset", tmp_path / "jobs")
    job = lab.start("audit")
    assert job["status"] == "queued"
    lab.close()
    assert lab.get(job["id"])["status"] == "cancelled"
    lab._run(job["id"])
    assert not lab._processes
    with pytest.raises(ValueError, match="shutting down"):
        lab.start("audit")


def test_real_audit_worker_persists_report_and_never_changes_production(tmp_path):
    from backend.data_collection import CollectionStore
    root = tmp_path / "dataset"
    CollectionStore(root).initialize()
    lab = ModelLab(root, tmp_path / "jobs")
    job = lab.start("audit", epochs=1)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        current = lab.get(job["id"])
        if current["status"] not in {"queued", "running"}:
            break
        time.sleep(.05)
    try:
        assert current["status"] == "succeeded", current
        assert current["result"]["can_train"] is False
        assert (lab.root / job["id"] / "result.json").is_file()
        assert not current["candidate_available"]
    finally:
        lab.close()


def test_train_export_and_frozen_test_evaluation_use_current_output_contract(tmp_path, monkeypatch):
    from research import lab_training
    from scripts import train_stop_inverted as source
    from backend.config import PROJECT_ROOT
    from backend.model_runtime import OnnxClassifier
    from backend.config import RuntimeConfig

    rng = np.random.default_rng(123)
    y = np.repeat(np.arange(len(LABELS)), 5)
    x = rng.normal(size=(len(y), 76)).astype(np.float32)
    base = {"X": x, "y": y}
    captured = {"X": np.concatenate([x, x + .01, x + .02]), "y": np.tile(y, 3),
                "split": np.repeat(["train", "val", "test"], len(y))}
    audit = {"can_train": True, "coverage_passed": False}
    monkeypatch.setattr(lab_training, "audit_dataset", lambda _: (captured, audit))
    monkeypatch.setattr(source, "prepare", lambda **_: (
        {"train": base, "val": {"X": x + 10, "y": y}, "test": {"X": x + 20, "y": y}}, {}))
    original_mix = lab_training.mix_training
    monkeypatch.setattr(lab_training, "mix_training", lambda a, b: original_mix(a, b, per_class=12))
    output = tmp_path / "candidate"
    output.mkdir()
    result = lab_training.train(tmp_path, output, PROJECT_ROOT, epochs=1)
    assert result["test_evaluated"] is False
    assert result["production_model_changed"] is False
    assert result["parity"]["label_agreement"] == 1
    classifier = OnnxClassifier(output / "candidate.onnx", output / "candidate_metadata.json", RuntimeConfig())
    probabilities, mass = classifier.predict_with_quality(x[:2])
    assert probabilities.shape == (2, len(CLASS_NAMES))
    assert mass.shape == (2,)
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    report = lab_training.evaluate(output, evaluation_dir, PROJECT_ROOT)
    assert report["camera_test"]["candidate"]["rows"] == len(y)
    assert report["public_test"]["production"]["rows"] == len(y)
    assert report["live_camera_qualification_pending"] is True


def test_model_lab_api_rejects_bad_origin_and_invalid_candidate_paths(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.app as module
    lab = ModelLab(tmp_path / "dataset", tmp_path / "jobs")
    monkeypatch.setattr(module, "model_lab", lab)
    client = TestClient(module.app)
    assert client.get("/api/model-lab/status").status_code == 200
    assert client.get("/api/model-lab/status", headers={"Origin": "https://untrusted.example"}).status_code == 403
    assert client.post("/api/model-lab/jobs", json={"kind": "evaluate", "candidate_job_id": "../production"}).status_code == 422
    assert client.post("/api/model-lab/jobs", json={"kind": "train", "epochs": 0}).status_code == 422
    assert client.get("/api/model-lab/jobs/invalid/candidate").status_code == 409
    assert client.get("/api/model-lab/status").json()["jobs"] == []
