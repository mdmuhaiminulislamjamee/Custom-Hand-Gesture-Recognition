"""Reviewed video collection, portable still exports, and causal runtime replay.

Raw uploads never change. Human time-span labels are authoritative; predictions
are diagnostics. Context and detector negatives retain their collection policy.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import mimetypes
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .config import ACTIVE_CLASS_NAMES, CLASS_NAMES, DISABLED_GESTURES
from .data_collection import atomic_json, collection_plan, identifier


MAX_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_DURATION_SECONDS = 300.0
VIDEO_SUFFIXES = {".mp4", ".webm", ".avi", ".mov"}
TRAINING_USES = {"hand_classifier", "context_only", "detector_only"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def finite_number(value, name, minimum=0.0, maximum=MAX_DURATION_SECONDS):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}.")
    return number


def probe_video(path: Path):
    """Decode bounded input, including WebM files without duration headers."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("This file cannot be decoded as a supported video.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or not 1 <= fps <= 240:
            raise ValueError("The video must declare a frame rate between 1 and 240 FPS.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not 1 <= width <= 4096 or not 1 <= height <= 4096 or width * height > 12_000_000:
            raise ValueError("Use a video no larger than 4096 pixels per side and 12 megapixels.")
        count, last_time, timestamp_frames = 0, 0.0, 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            count += 1
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError("The video contains invalid or changing frame dimensions.")
            position = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if math.isfinite(position) and position > last_time:
                last_time = position
                timestamp_frames += 1
            duration = max(count / fps, last_time + 1 / fps)
            if duration > MAX_DURATION_SECONDS + .001 or count > 72_000:
                raise ValueError("Videos must be no longer than 300 seconds. Split this recording into clips.")
        if count < 2:
            raise ValueError("The video must contain at least two decodable frames.")
        return dict(width=width, height=height, fps=fps, frame_count=count,
                    duration_seconds=round(max(count / fps, last_time + 1 / fps), 6),
                    timestamp_source="decoder" if timestamp_frames > count // 2 else "frame_rate")
    finally:
        capture.release()


def reviewed_scenarios():
    rows = [dict(id="valid_command", title="Valid command", training_use="hand_classifier",
                 instruction="Record neutral, present a valid command, hold, then release. Review its time span."),
            dict(id="ordinary_activity", title="Everyday activity", training_use="context_only",
                 instruction="Record ordinary activity, then label each relevant span and negative reason.")]
    rows.extend(dict(id=s["view"], title=f'{s["group"]}: {s["view"].replace("_", " ")}',
                     instruction=s["instruction"], training_use=s["training_use"])
                for s in collection_plan() if s["label"] == "no_gesture")
    return rows


def validate_segments(segments, duration):
    if not isinstance(segments, list) or not 1 <= len(segments) <= 500:
        raise ValueError("Review between 1 and 500 annotation spans.")
    prompts = {s["view"]: s for s in collection_plan() if s["label"] == "no_gesture"}
    output = []
    for row in segments:
        if not isinstance(row, dict):
            raise ValueError("Each annotation must be an object.")
        start = finite_number(row.get("start"), "Start", maximum=duration)
        end = finite_number(row.get("end"), "End", maximum=duration)
        if end <= start:
            raise ValueError("Annotation end must be after its start.")
        label = row.get("label")
        if label not in {*CLASS_NAMES, "no_gesture", "uncertain"}:
            raise ValueError("Unknown gesture label.")
        reason = str(row.get("negative_reason") or "")
        if label in DISABLED_GESTURES:
            label, reason = "no_gesture", "dorsal_casual"
        if label == "no_gesture":
            if reason not in prompts:
                raise ValueError("Select a known negative reason for every no_gesture span.")
            training_use = prompts[reason]["training_use"]
            if row.get("training_use", training_use) != training_use:
                raise ValueError("The negative reason determines whether a sample can train the hand classifier.")
        elif label == "uncertain":
            training_use, reason = "excluded", "uncertain_transition"
        else:
            training_use, reason = "hand_classifier", ""
            if row.get("training_use", training_use) != training_use:
                raise ValueError("Valid commands use hand_classifier training policy.")
        hand = row.get("handedness", "unspecified")
        orientation = row.get("finger_orientation", "unspecified")
        if hand not in {"left", "right", "both", "none", "unspecified"}:
            raise ValueError("Unknown handedness.")
        if orientation not in {"toward_camera", "away_from_camera", "none", "unspecified"}:
            raise ValueError("Unknown finger orientation.")
        if label == "fist" and orientation == "away_from_camera":
            raise ValueError("A reverse-facing fist must be reviewed as no_gesture, not Fist.")
        output.append(dict(start=start, end=end, label=label, negative_reason=reason,
                           training_use=training_use, handedness=hand,
                           finger_orientation=orientation))
    output.sort(key=lambda row: row["start"])
    if any(b["start"] < a["end"] - 1e-7 for a, b in zip(output, output[1:])):
        raise ValueError("Annotation spans must not overlap.")
    return output


def score_replay(segments, events, duration, frame_count, detected_frames):
    """Count emitted action edges, never frame-level predictions as actions."""
    negative_seconds = sum(s["end"] - s["start"] for s in segments if s["label"] == "no_gesture")
    reviewed_seconds = sum(s["end"] - s["start"] for s in segments if s["label"] != "uncertain")
    positives = [(i, s) for i, s in enumerate(segments) if s["label"] in CLASS_NAMES]
    correct, latencies, wrong, false, duplicates, unscored = set(), [], 0, 0, 0, 0
    output_events = []
    for event in events:
        event = dict(event)
        match = next(((i, s) for i, s in enumerate(segments)
                      if s["start"] <= event["time"] < s["end"]), None)
        if match is None or match[1]["label"] == "uncertain":
            event["outcome"] = "unscored"
            unscored += 1
        else:
            index, segment = match
            event.update(segment_index=index, expected=segment["label"])
            if segment["label"] == "no_gesture":
                event["outcome"] = "false_action"
                false += 1
            elif event["label"] != segment["label"]:
                event["outcome"] = "wrong_command"
                wrong += 1
            elif index in correct:
                event["outcome"] = "duplicate"
                duplicates += 1
            else:
                event["outcome"] = "correct"
                correct.add(index)
                latencies.append(event["time"] - segment["start"])
        output_events.append(event)
    per_class = {}
    for label in CLASS_NAMES:
        indices = [i for i, s in positives if s["label"] == label]
        if indices:
            matched = sum(i in correct for i in indices)
            per_class[label] = dict(segments=len(indices), matched=matched,
                                   recall=matched / len(indices))
    metrics = dict(false_actions=false,
                   false_actions_per_hour=false * 3600 / negative_seconds if negative_seconds else None,
                   command_recall=len(correct) / len(positives) if positives else None,
                   wrong_commands=wrong, duplicate_actions=duplicates,
                   mean_latency_seconds=sum(latencies) / len(latencies) if latencies else None,
                   negative_seconds=negative_seconds, positive_segments=len(positives),
                   matched_segments=len(correct), missed_segments=len(positives) - len(correct),
                   reviewed_seconds=reviewed_seconds,
                   annotation_coverage=reviewed_seconds / duration if duration else 0,
                   processed_frames=frame_count, detected_frames=detected_frames,
                   detection_coverage=detected_frames / frame_count if frame_count else 0,
                   unscored_actions=unscored, per_class=per_class)
    return metrics, output_events


class VideoLabStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-lab")
        self._jobs = {}
        self._closing = threading.Event()

    def close(self):
        self._closing.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _inside(self, *parts):
        path = self.root.joinpath(*parts).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Video path escapes the configured storage folder.")
        return path

    def _clip_dir(self, clip_id):
        if not isinstance(clip_id, str) or not re.fullmatch(r"[a-f0-9]{32}", clip_id):
            raise ValueError("Invalid video ID.")
        return self._inside("clips", clip_id)

    def new_upload(self, filename):
        suffix = Path(str(filename or "")).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise ValueError("Upload an MP4, WebM, AVI, or MOV video.")
        clip_id = uuid.uuid4().hex
        directory = self._clip_dir(clip_id)
        directory.mkdir(parents=True, exist_ok=False)
        return clip_id, directory / ("upload" + suffix)

    @staticmethod
    def upload_metadata(participant_id, session_id, source, scenario, lighting, distance, note):
        if source not in {"ncm", "webcam"}:
            raise ValueError("Choose NCM or webcam as the recording source.")
        if scenario not in {s["id"] for s in reviewed_scenarios()}:
            raise ValueError("Choose a known recording scenario.")
        if lighting not in {"normal", "dim", "backlit"} or distance not in {"near", "medium", "far"}:
            raise ValueError("Unknown recording conditions.")
        return dict(participant_id=identifier(participant_id, "Participant ID"),
                    session_id=identifier(session_id, "Session ID"), source=source,
                    scenario=scenario, lighting=lighting, distance=distance,
                    note=str(note).strip()[:2000])

    def finish_upload(self, clip_id, staged_path, filename, metadata):
        expected = self._clip_dir(clip_id) / ("upload" + Path(filename).suffix.lower())
        if Path(staged_path).resolve() != expected or not expected.is_file():
            raise ValueError("Invalid staged upload.")
        size = expected.stat().st_size
        if not 1 <= size <= MAX_UPLOAD_BYTES:
            raise ValueError("Video uploads must be at most 256 MB.")
        info = probe_video(expected)
        digest = hashlib.sha256()
        with expected.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        original = expected.with_name("original" + expected.suffix)
        expected.replace(original)
        record = dict(schema_version=1, id=clip_id, original_filename=Path(filename).name[:255],
                      created_at_utc=utc_now(), bytes=size, sha256=digest.hexdigest(),
                      media_filename=original.name, **metadata, **info,
                      reviewed=False, annotations_revision=0, segments=[], extraction=None,
                      reports=[], media_url=f"/api/video-lab/clips/{clip_id}/media")
        atomic_json(self._clip_dir(clip_id) / "clip.json", record)
        return record

    def get_clip(self, clip_id):
        with self._lock:
            self._clip_dir(clip_id)
            path = self._inside("clips", clip_id, "clip.json")
            if not path.is_file():
                raise ValueError("The selected video does not exist.")
            clip = json.loads(path.read_text(encoding="utf-8"))
            if clip.get("segments"):
                clip["segments"] = validate_segments(clip["segments"], clip["duration_seconds"])
            return clip

    def media_path(self, clip_id):
        clip = self.get_clip(clip_id)
        name = clip.get("media_filename", "")
        if name not in {"original" + suffix for suffix in VIDEO_SUFFIXES}:
            raise ValueError("Invalid stored video filename.")
        path = self._inside("clips", clip_id, name)
        if not path.is_file():
            raise ValueError("The original video is missing.")
        return path

    def summary(self):
        with self._lock:
            clips = []
            base = self._inside("clips")
            if base.exists():
                for metadata in base.glob("*/clip.json"):
                    if not metadata.resolve().is_relative_to(base):
                        continue
                    try:
                        clip = self.get_clip(metadata.parent.name)
                        # Full timelines are available through clip details.
                        clip["reports"] = [{k: v for k, v in r.items() if k not in {"timeline", "events"}}
                                           for r in clip.get("reports", [])]
                        clips.append(clip)
                    except (ValueError, OSError):
                        continue
            return dict(clips=sorted(clips, key=lambda c: c["created_at_utc"], reverse=True),
                        classes=[*ACTIVE_CLASS_NAMES, "no_gesture", "uncertain"],
                        scenarios=reviewed_scenarios(), max_upload_bytes=MAX_UPLOAD_BYTES,
                        max_duration_seconds=MAX_DURATION_SECONDS)

    def annotate(self, clip_id, segments, reviewed):
        if not isinstance(reviewed, bool):
            raise ValueError("Specify whether the annotation spans were reviewed by a person.")
        with self._lock:
            clip = self.get_clip(clip_id)
            clip["segments"] = validate_segments(segments, clip["duration_seconds"])
            clip["reviewed"] = reviewed
            clip["annotations_revision"] += 1
            clip["reviewed_at_utc"] = utc_now() if reviewed else None
            atomic_json(self._clip_dir(clip_id) / "clip.json", clip)
            return clip

    def reviewed_clip(self, clip_id):
        clip = self.get_clip(clip_id)
        if not clip.get("reviewed") or not clip.get("segments"):
            raise ValueError("Review and save annotation spans before extracting or evaluating.")
        return clip

    def start_job(self, kind, clip_id, operation: Callable):
        with self._lock:
            if self._closing.is_set():
                raise ValueError("Video processing is shutting down.")
            running = [j for j in self._jobs.values() if j["status"] in {"queued", "running"}]
            if len(running) >= 4:
                raise ValueError("Four video jobs are already queued. Wait for a job to finish.")
            if any(j["clip_id"] == clip_id and j["kind"] == kind for j in running):
                raise ValueError("This video operation is already running.")
            job_id = uuid.uuid4().hex
            job = dict(id=job_id, job_id=job_id, clip_id=clip_id, kind=kind,
                       status="queued", progress=0.0, result=None, error=None,
                       created_at_utc=utc_now())
            self._jobs[job_id] = job
            def execute():
                with self._lock:
                    job["status"] = "running"
                def progress(value):
                    with self._lock:
                        job["progress"] = round(min(1.0, max(0.0, value)), 4)
                try:
                    result = operation(progress)
                    with self._lock:
                        job.update(status="completed", progress=1.0, result=result,
                                   finished_at_utc=utc_now())
                except Exception as error:
                    with self._lock:
                        job.update(status="failed", error=str(error), finished_at_utc=utc_now())
            self._executor.submit(execute)
            return copy.deepcopy(job)

    def get_job(self, job_id):
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError("Video job not found. Job status resets when the server restarts.")
            return copy.deepcopy(self._jobs[job_id])

    def _frames(self, clip, interval):
        import cv2

        capture = cv2.VideoCapture(str(self.media_path(clip["id"])))
        try:
            index, next_time, last_time = 0, 0.0, -1.0
            while capture.isOpened():
                if self._closing.is_set():
                    raise ValueError("Video processing stopped because the server is shutting down.")
                success, frame = capture.read()
                if not success:
                    break
                nominal = index / clip["fps"]
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if clip["timestamp_source"] != "decoder" or not math.isfinite(timestamp) or timestamp < last_time:
                    timestamp = nominal
                timestamp = max(timestamp, last_time + 1e-6)
                last_time = timestamp
                index += 1
                if timestamp + 1e-6 < next_time:
                    continue
                next_time = timestamp + interval
                success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not success:
                    raise ValueError("A video frame could not be encoded.")
                yield index - 1, timestamp, encoded.tobytes()
            if index != clip["frame_count"]:
                raise ValueError("The recording changed or decoding ended before all validated frames.")
        finally:
            capture.release()

    @staticmethod
    def _predict(runtime, jpeg, session, timestamp, config):
        # Keep the full frame for hand search and consistent preview coordinates.
        result = runtime.process_frame(jpeg, session, now=1000.0 + timestamp)
        if result.get("status") not in {"predicted", "no_hand"}:
            raise ValueError(result.get("message") or "Runtime is unavailable; replay cannot be scored.")
        return result

    def extract(self, clip, collection_store, runtime_factory, config, interval, progress):
        from .model_runtime import RuntimeSession

        interval = finite_number(interval, "Sampling interval", minimum=.2, maximum=10)
        collection_store.add_participant(clip["participant_id"])
        prompts = {s["id"]: s for s in collection_plan()}
        session, runtime = RuntimeSession(config), runtime_factory(None)
        saved, duplicates, excluded, samples = 0, 0, 0, []
        try:
            for index, timestamp, jpeg in self._frames(clip, interval):
                progress(timestamp / clip["duration_seconds"])
                segment = next((s for s in clip["segments"] if s["start"] <= timestamp < s["end"]), None)
                if segment is None or segment["label"] == "uncertain":
                    excluded += 1
                    continue
                step_id = (f'no_gesture/{segment["negative_reason"]}' if segment["label"] == "no_gesture"
                           else f'{segment["label"]}/casual')
                if step_id not in prompts:
                    raise ValueError("An annotation no longer matches the current collection plan; review it again.")
                # Raw reviewed images remain useful when the current detector
                # or classifier is unavailable; never use predictions as a
                # condition for saving a human-reviewed example.
                try:
                    result = self._predict(runtime, jpeg, session, timestamp, config)
                except (ValueError, RuntimeError) as error:
                    result = dict(status="diagnostics_unavailable", runtime_prediction="no_gesture",
                                  action_reason=str(error), feature_vector=None, landmarks=None)
                frozen = collection_store.freeze_bytes(index, jpeg, result, source=clip["source"])
                try:
                    row = collection_store.save(token=frozen["token"], participant_id=clip["participant_id"],
                                                session_id=clip["session_id"], step_id=step_id,
                                                handedness=segment["handedness"],
                                                finger_orientation=segment["finger_orientation"],
                                                lighting=clip["lighting"], distance=clip["distance"], note=clip["note"])
                except ValueError as error:
                    if "already saved" in str(error):
                        duplicates += 1
                        continue
                    raise
                image = Path(row["image_path"])
                metadata_path = image.with_suffix(".json")
                with collection_store._lock:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata["feature_space"] = "smoothed_landmarks_in_full_frame"
                    metadata["world_landmarks"] = result.get("world_landmarks")
                    metadata["video_provenance"] = dict(clip_id=clip["id"], sha256=clip["sha256"],
                                                        frame_index=index, timestamp_seconds=timestamp,
                                                        annotations_revision=clip["annotations_revision"],
                                                        negative_reason=segment["negative_reason"],
                                                        training_use=segment["training_use"])
                    atomic_json(metadata_path, metadata)
                saved += 1
                samples.append(dict(sample_id=row["sample_id"], timestamp_seconds=timestamp,
                                    label=segment["label"], training_use=segment["training_use"],
                                    has_features=row["has_features"], image_path=metadata["image_path"]))
        finally:
            runtime.close()
        report = dict(saved=saved, duplicates=duplicates, excluded=excluded, samples=samples,
                      created_at_utc=utc_now(), annotations_revision=clip["annotations_revision"],
                      interval_seconds=interval,
                      note="Review selected frames for pose diversity before training. Context/detector negatives remain excluded from the 76-feature hand classifier.")
        with self._lock:
            current = self.get_clip(clip["id"])
            current["extraction"] = report
            atomic_json(self._clip_dir(clip["id"]) / "clip.json", current)
        return report

    def evaluate(self, clip, runtime_factory, config, candidate_job_id, progress):
        from .model_runtime import RuntimeSession

        runtime, session = runtime_factory(candidate_job_id), RuntimeSession(config)
        manager = getattr(runtime, "model_manager", None)
        model_metadata = copy.deepcopy(getattr(manager, "bundle_metadata", {}))
        events, timeline, detected_frames = [], [], 0
        interval = 1.0 / min(10.0, float(config.target_fps))
        try:
            for _, timestamp, jpeg in self._frames(clip, interval):
                result = self._predict(runtime, jpeg, session, timestamp, config)
                detected = result.get("status") == "predicted"
                detected_frames += int(detected)
                label = result.get("runtime_prediction", "no_gesture")
                action = result.get("runtime_action", "Wait / No Action")
                quality = result.get("quality") or {}
                timeline.append(dict(time=round(timestamp, 6), prediction=label, action=action,
                                     reason=result.get("action_reason", ""), confidence=result.get("confidence", 0),
                                     detected=detected, issues=quality.get("issues", [])))
                if action not in {"Wait / No Action", "No Action", "", None}:
                    events.append(dict(time=timestamp, label=label, action=action,
                                       confidence=result.get("confidence", 0)))
                progress(timestamp / clip["duration_seconds"])
        finally:
            runtime.close()
        metrics, events = score_replay(clip["segments"], events, clip["duration_seconds"], len(timeline), detected_frames)
        report = dict(id=uuid.uuid4().hex, clip_id=clip["id"], created_at_utc=utc_now(),
                      model=candidate_job_id or "production", annotations_revision=clip["annotations_revision"],
                      model_metadata=model_metadata, source_fps=clip["fps"],
                      replay_fps=min(10.0, config.target_fps), metrics=metrics, events=events, timeline=timeline,
                      policy="Causal hand-only runtime replay including detection, geometry, tracking, holds, action cooldown, and release. Uncertain spans and unlabeled gaps are replayed but unscored. Each positive span represents one intended action.")
        with self._lock:
            current = self.get_clip(clip["id"])
            current["reports"].append(report)
            atomic_json(self._clip_dir(clip["id"]) / "clip.json", current)
        return report


def create_video_router(store, collection_store, engine, config, check_origin,
                        *, candidate_engine_factory=None):
    router = APIRouter(prefix="/api/video-lab", tags=["video-lab"])

    def runtime_factory(candidate_job_id):
        if candidate_job_id:
            if candidate_engine_factory is None:
                raise ValueError("Candidate replay is unavailable; choose the production model.")
            return candidate_engine_factory(candidate_job_id)
        from .model_runtime import InferenceEngine
        isolated = InferenceEngine(config, engine.model_manager.models_directory, load_models=False)
        # A replay is a fixed ONNX comparison: snapshot immutable classifiers,
        # not the mutable feedback adapter used by the live session.
        with engine.model_manager._lock:
            isolated.model_manager.models = dict(engine.model_manager.models)
            isolated.model_manager.selected_model_name = engine.model_manager.selected_model_name
            isolated.model_manager.errors = list(engine.model_manager.errors)
            isolated.model_manager.bundle_metadata = copy.deepcopy(engine.model_manager.bundle_metadata)
        isolated.model_manager.bundle_metadata["online_adapter"] = "disabled_for_fixed_replay"
        return isolated

    async def call(request, method, *args, **kwargs):
        accepted = check_origin(request)
        if inspect.isawaitable(accepted):
            await accepted
        try:
            return await asyncio.to_thread(method, *args, **kwargs)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except OSError as error:
            raise HTTPException(status_code=503, detail=f"Video storage unavailable: {error}") from error

    @router.get("")
    async def summary(request: Request):
        return await call(request, store.summary)

    @router.post("/upload")
    async def upload(request: Request, video: UploadFile = File(...), participant_id: str = Form(...),
                     session_id: str = Form(...), source: str = Form("webcam"),
                     scenario: str = Form("ordinary_activity"), lighting: str = Form("normal"),
                     distance: str = Form("medium"), note: str = Form("")):
        metadata = await call(request, store.upload_metadata, participant_id, session_id, source,
                              scenario, lighting, distance, note)
        clip_id, staged = await call(request, store.new_upload, video.filename)
        complete = False
        try:
            total = 0
            with staged.open("xb") as handle:
                while block := await video.read(1024 * 1024):
                    total += len(block)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Video uploads must be at most 256 MB.")
                    handle.write(block)
            result = await call(request, store.finish_upload, clip_id, staged, video.filename, metadata)
            complete = True
            return result
        finally:
            await video.close()
            if not complete:
                staged.unlink(missing_ok=True)

    @router.get("/jobs/{job_id}")
    async def job(request: Request, job_id: str):
        return await call(request, store.get_job, job_id)

    @router.get("/clips/{clip_id}")
    async def details(request: Request, clip_id: str):
        return await call(request, store.get_clip, clip_id)

    @router.get("/clips/{clip_id}/media")
    async def media(request: Request, clip_id: str):
        path = await call(request, store.media_path, clip_id)
        return FileResponse(path, media_type=mimetypes.guess_type(str(path))[0] or "video/mp4",
                            headers={"Cache-Control": "private, max-age=3600"})

    @router.post("/clips/{clip_id}/annotations")
    async def annotations(request: Request, clip_id: str, payload: dict[str, Any]):
        return await call(request, store.annotate, clip_id, payload.get("segments"), payload.get("reviewed"))

    @router.post("/clips/{clip_id}/extract")
    async def extract(request: Request, clip_id: str, payload: dict[str, Any]):
        clip = await call(request, store.reviewed_clip, clip_id)
        interval = await call(request, finite_number, payload.get("interval_seconds", 1),
                              "Sampling interval", minimum=.2, maximum=10)
        return await call(request, store.start_job, "extract", clip_id,
                          lambda progress: store.extract(clip, collection_store, runtime_factory, config, interval, progress))

    @router.post("/clips/{clip_id}/evaluate")
    async def evaluate(request: Request, clip_id: str, payload: dict[str, Any]):
        clip = await call(request, store.reviewed_clip, clip_id)
        candidate = payload.get("candidate_job_id")
        if candidate is not None and (not isinstance(candidate, str) or not re.fullmatch(r"[a-f0-9]{32}", candidate)):
            raise HTTPException(status_code=400, detail="Invalid candidate job ID.")
        return await call(request, store.start_job, "evaluate", clip_id,
                          lambda progress: store.evaluate(clip, runtime_factory, config, candidate, progress))

    return router
