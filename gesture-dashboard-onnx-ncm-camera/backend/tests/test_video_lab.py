from __future__ import annotations

import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.config import RuntimeConfig
from backend.data_collection import CollectionStore
from backend.video_lab import (VideoLabStore, create_video_router, probe_video,
                               score_replay, validate_segments)


def video_bytes(tmp_path, suffix=".avi"):
    path = tmp_path / ("fixture" + suffix)
    fourcc = "VP80" if suffix == ".webm" else "MJPG"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 10, (80, 60))
    if not writer.isOpened():
        pytest.skip(f"Local OpenCV does not encode {suffix}")
    for i in range(20):
        frame = np.full((60, 80, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


@pytest.fixture
def lab(tmp_path):
    store = VideoLabStore(tmp_path / "videos")
    collection = CollectionStore(tmp_path / "images")
    app = FastAPI()
    def origin(request):
        if request.headers.get("origin") == "https://untrusted.example":
            raise HTTPException(403)
    app.include_router(create_video_router(store, collection,
        SimpleNamespace(model_manager=None), RuntimeConfig(), origin))
    with TestClient(app) as client:
        yield store, collection, client
    store.close()


def upload(client, content, filename="test.avi"):
    return client.post("/api/video-lab/upload", files={"video": (filename, content)},
                       data={"participant_id": "person-1", "session_id": "session-1",
                             "source": "webcam", "scenario": "standing_arm_hanging"})


def test_upload_review_media_range_and_restart(lab, tmp_path):
    store, _, client = lab
    content = video_bytes(tmp_path)
    response = upload(client, content)
    assert response.status_code == 200, response.text
    clip = response.json()
    assert clip["duration_seconds"] == pytest.approx(2)
    assert clip["frame_count"] == 20
    assert store.media_path(clip["id"]).read_bytes() == content
    assert client.get(clip["media_url"], headers={"Range": "bytes=0-15"}).status_code == 206
    assert client.get(clip["media_url"], headers={"Range": "bytes=0-15"}).content == content[:16]
    assert client.post(f'/api/video-lab/clips/{clip["id"]}/extract', json={}).status_code == 400
    spans = [{"start": 0, "end": 2, "label": "no_gesture", "negative_reason": "standing_arm_hanging"}]
    draft = client.post(f'/api/video-lab/clips/{clip["id"]}/annotations',
                        json={"segments": spans, "reviewed": False})
    assert draft.status_code == 200
    assert draft.json()["reviewed"] is False
    assert client.post(f'/api/video-lab/clips/{clip["id"]}/evaluate', json={}).status_code == 400
    response = client.post(f'/api/video-lab/clips/{clip["id"]}/annotations',
                           json={"segments": spans, "reviewed": True})
    assert response.status_code == 200, response.text
    assert response.json()["segments"][0]["training_use"] == "context_only"
    assert client.post(f'/api/video-lab/clips/{clip["id"]}/extract',
                       json={"interval_seconds": -1}).status_code == 400
    restarted = VideoLabStore(store.root)
    try:
        assert restarted.get_clip(clip["id"])["reviewed"] is True
    finally:
        restarted.close()


def test_reject_invalid_annotations_and_paths(lab, tmp_path):
    store, _, client = lab
    clip = upload(client, video_bytes(tmp_path)).json()
    base = f'/api/video-lab/clips/{clip["id"]}'
    assert client.post(base + "/annotations", json={"segments": [], "reviewed": True}).status_code == 400
    spans = [{"start": 0, "end": 2, "label": "no_gesture", "negative_reason": "standing_arm_hanging",
              "training_use": "hand_classifier"}]
    assert client.post(base + "/annotations", json={"segments": spans, "reviewed": True}).status_code == 400
    assert upload(client, b"invalid data").status_code == 400
    assert upload(client, b"invalid data", "test.exe").status_code == 400
    assert client.get("/api/video-lab", headers={"Origin": "https://untrusted.example"}).status_code == 403
    with pytest.raises(ValueError, match="Invalid video"):
        store.get_clip("../../other")
    with pytest.raises(ValueError, match="overlap"):
        validate_segments([dict(start=0, end=1.5, label="dorsal"), dict(start=1, end=2, label="left")], 2)
    with pytest.raises(ValueError, match="reverse-facing"):
        validate_segments([dict(start=0, end=1, label="fist", finger_orientation="away_from_camera")], 2)


def test_webm_upload_roundtrip(lab, tmp_path):
    store, _, client = lab
    content = video_bytes(tmp_path, ".webm")
    response = upload(client, content, "capture.webm")
    assert response.status_code == 200, response.text
    clip = response.json()
    assert clip["duration_seconds"] == pytest.approx(2, abs=.11)
    assert len(list(store._frames(clip, .1))) == 20


class FakeRuntime:
    def __init__(self, failure=False):
        self.closed = False
        self.times = []
        self.failure = failure

    def process_frame(self, jpeg, session, **kwargs):
        assert "center_crop_ratio" not in kwargs, "Body verification needs the full frame"
        self.times.append(kwargs["now"])
        if self.failure:
            return {"status": "dependency_missing", "message": "Detector unavailable"}
        timestamp = kwargs["now"] - 1000
        emitted = .39 < timestamp < .41
        return dict(status="no_hand", runtime_prediction="left" if emitted else "no_gesture",
                    runtime_action="Move Left" if emitted else "Wait / No Action",
                    confidence=.9 if emitted else 0, quality={})

    def close(self):
        self.closed = True


def test_context_export_keeps_raw_failures_and_portable_schema(lab, tmp_path):
    store, collection, client = lab
    clip = upload(client, video_bytes(tmp_path)).json()
    clip = store.annotate(clip["id"], [dict(start=0, end=2, label="no_gesture",
        negative_reason="standing_arm_hanging")], True)
    runtime = FakeRuntime(failure=True)
    report = store.extract(clip, collection, lambda _: runtime, RuntimeConfig(), 1, lambda _: None)
    assert report["saved"] == 2
    assert runtime.closed
    records = [json.loads(p.read_text()) for p in collection.root.glob("participants/*/*/*/*/*.json")]
    assert len(records) == 2
    assert all(r["reviewed"] and r["schema_version"] == 2 for r in records)
    assert all(r["training_use"] == "context_only" and r["feature_vector"] is None for r in records)
    assert all(r["feature_space"] == "smoothed_landmarks_in_full_frame" for r in records)
    assert all(r["video_provenance"]["clip_id"] == clip["id"] for r in records)
    assert store.get_clip(clip["id"])["extraction"]["saved"] == 2


def test_causal_replay_scores_actions_persists_and_errors_on_runtime_failure(lab, tmp_path):
    store, _, client = lab
    clip = upload(client, video_bytes(tmp_path)).json()
    clip = store.annotate(clip["id"], [dict(start=0, end=1, label="left"),
        dict(start=1, end=2, label="no_gesture", negative_reason="standing_arm_hanging")], True)
    runtime = FakeRuntime()
    report = store.evaluate(clip, lambda _: runtime, RuntimeConfig(), None, lambda _: None)
    assert report["metrics"]["command_recall"] == 1
    assert report["metrics"]["false_actions_per_hour"] == 0
    assert report["metrics"]["mean_latency_seconds"] == pytest.approx(.4)
    assert runtime.times == sorted(runtime.times)
    assert runtime.closed
    assert len(store.get_clip(clip["id"])["reports"]) == 1
    broken = FakeRuntime(failure=True)
    with pytest.raises(ValueError, match="unavailable"):
        store.evaluate(clip, lambda _: broken, RuntimeConfig(), None, lambda _: None)
    assert broken.closed


def test_event_metrics_distinguish_false_wrong_duplicate_and_unscored():
    spans = [dict(start=0, end=1, label="no_gesture"), dict(start=1, end=3, label="left"),
             dict(start=3, end=4, label="uncertain")]
    events = [dict(time=.5, label="dorsal"), dict(time=1.2, label="right"),
              dict(time=1.5, label="left"), dict(time=2, label="left"), dict(time=3.5, label="dorsal")]
    metrics, tagged = score_replay(spans, events, 4, 40, 30)
    assert metrics["false_actions_per_hour"] == 3600
    assert metrics["wrong_commands"] == metrics["duplicate_actions"] == 1
    assert metrics["command_recall"] == 1
    assert metrics["mean_latency_seconds"] == .5
    assert metrics["annotation_coverage"] == .75
    assert [event["outcome"] for event in tagged] == ["false_action", "wrong_command", "correct", "duplicate", "unscored"]


def test_evaluate_api_uses_selected_candidate_job_and_polls(lab, tmp_path):
    store, collection, client = lab
    clip = upload(client, video_bytes(tmp_path)).json()
    store.annotate(clip["id"], [dict(start=0, end=2, label="left")], True)
    selected = []
    candidate = "a" * 32
    app = FastAPI()
    def factory(job_id):
        selected.append(job_id)
        return FakeRuntime()
    app.include_router(create_video_router(store, collection, None, RuntimeConfig(),
                                          lambda _: None, candidate_engine_factory=factory))
    with TestClient(app) as candidate_client:
        response = candidate_client.post(f'/api/video-lab/clips/{clip["id"]}/evaluate',
                                         json={"candidate_job_id": candidate})
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = candidate_client.get(f'/api/video-lab/jobs/{job_id}').json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(.01)
        assert job["status"] == "completed", job
        assert job["result"]["model"] == candidate
        assert job["result"]["metrics"]["command_recall"] == 1
        assert selected == [candidate]


def test_jobs_poll_failure_and_close_cancels_decode(lab, tmp_path):
    store, _, client = lab
    clip = upload(client, video_bytes(tmp_path)).json()
    job = store.start_job("test", clip["id"], lambda progress: {"done": True})
    deadline = time.monotonic() + 3
    while store.get_job(job["id"])["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
        time.sleep(.01)
    assert client.get(f'/api/video-lab/jobs/{job["id"]}').json()["result"] == {"done": True}
    store.close()
    with pytest.raises(ValueError, match="shutting down"):
        next(store._frames(clip, .1))


def test_overlong_video_rejected_without_processing_unbounded_frames(monkeypatch, tmp_path):
    class Capture:
        def isOpened(self): return True
        def get(self, key):
            return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_WIDTH: 80,
                    cv2.CAP_PROP_FRAME_HEIGHT: 60, cv2.CAP_PROP_POS_MSEC: 301000}.get(key, 0)
        def read(self): return True, np.zeros((60, 80, 3), dtype=np.uint8)
        def release(self): pass
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: Capture())
    with pytest.raises(ValueError, match="300 seconds"):
        probe_video(tmp_path / "overlong.avi")
