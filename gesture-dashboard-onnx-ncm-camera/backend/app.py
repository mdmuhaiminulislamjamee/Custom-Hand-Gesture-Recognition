from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import secrets
import time
from typing import Any, Literal

from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .artifact_integrity import ArtifactRegistry
from .config import (
    MODELS_DIRECTORY,
    ProductionSettings,
    RuntimeConfig,
    load_production_settings,
    load_runtime_config,
)
from .model_runtime import InferenceEngine, RuntimeSession
from .ncm_camera import NcmCameraClient, NcmCameraConfig
from .observability import RuntimeMetrics, configure_logging
from .online_learning import OnlineLearningService
from .runtime import InferenceStartLimiter
from .storage import FeedbackRecord, FeedbackStore, artifact_status, load_metric_files
from .data_collection import CollectionStore


ALLOWED_ORIGINS = {
    "http://127.0.0.1:3200",
    "http://localhost:3200",
    "http://127.0.0.1:5175",
    "http://localhost:5175",
}


class FeedbackPayload(BaseModel):
    actual_label: str
    predicted_label: str
    runtime_action: str = "Wait / No Action"
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str = "unknown"
    probabilities: dict[str, float] = Field(default_factory=dict)
    base_probabilities: dict[str, float] | None = None
    feature_vector: list[float] | None = None
    landmarks: list[list[float]] | None = None
    note: str = Field(default="", max_length=500)
    snapshot_data_url: str | None = None
    learning_mode: Literal["audit", "safe", "force"] = "audit"


class OnlineRollbackPayload(BaseModel):
    backup_name: str = Field(min_length=1, max_length=255)


production_settings: ProductionSettings = load_production_settings()
logger = configure_logging(production_settings.log_directory)
artifact_registry = ArtifactRegistry()
runtime_metrics = RuntimeMetrics(production_settings.metrics_window_size)
service_lock = asyncio.Lock()
configuration_error: str | None = None


def _build_services() -> tuple[
    RuntimeConfig,
    InferenceEngine,
    FeedbackStore,
    OnlineLearningService,
]:
    global configuration_error
    errors: list[str] = []
    try:
        config = load_runtime_config()
    except (OSError, ValueError) as error:
        errors.append(str(error))
        config = RuntimeConfig()
    integrity = artifact_registry.verify()
    trusted = bool(
        integrity["verified"] or not production_settings.require_artifact_manifest
    )
    if not trusted:
        errors.extend(integrity["errors"])
    inference_engine = InferenceEngine(config, load_models=trusted)
    if errors:
        inference_engine.model_manager.errors[:0] = errors
    configuration_error = " · ".join(errors) if errors else None
    store = FeedbackStore(config)
    learning = OnlineLearningService(
        config,
        inference_engine.model_manager,
        MODELS_DIRECTORY,
        require_holdout_for_safe=True,
    )
    inference_engine.model_manager.set_online_adapter(learning)
    return config, inference_engine, store, learning


runtime_config, engine, feedback_store, online_learning = _build_services()
inference_start_limiter = InferenceStartLimiter(runtime_config.frame_interval_ms)
ncm_camera = NcmCameraClient(NcmCameraConfig.from_environment())
action_history: deque[dict[str, Any]] = deque(maxlen=500)
active_sessions: dict[str, RuntimeSession] = {}
collection_store = CollectionStore()


async def _process_frame_with_limit(
    frame: bytes,
    session: RuntimeSession,
    model_name: str | None,
    **kwargs: Any,
) -> dict[str, Any]:
    # The reservation is committed before model execution, so a failed inference
    # cannot let the next request bypass the backend-wide rate limit.
    await inference_start_limiter.wait_for_start()
    return await asyncio.to_thread(
        engine.process_frame,
        frame,
        session,
        model_name,
        **kwargs,
    )


@asynccontextmanager
async def application_lifespan(_: FastAPI):
    if ncm_camera.config.auto_connect:
        ncm_camera.start()
    yield
    ncm_camera.stop()
    engine.close()


app = FastAPI(
    title="Gesture Control Lab ONNX + NCM Camera API",
    version="2.0.0",
    description=(
        "Independent MediaPipe + ONNX Runtime deployment for the JLIP camera "
        "stream delivered by the USB-NCM development board."
    ),
    lifespan=application_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Token", "X-Request-ID"],
)


@app.middleware("http")
async def operational_middleware(request: Request, call_next: Any) -> Any:
    request_id = request.headers.get("x-request-id") or secrets.token_hex(8)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        runtime_metrics.record_error("unhandled_http_error")
        logger.exception(
            "Unhandled HTTP request error",
            extra={"request_id": request_id, "method": request.method, "path": request.url.path},
        )
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    logger.info(
        "HTTP request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )
    return response


def _admin_allowed(enabled: bool, supplied_token: str | None) -> None:
    if not enabled:
        raise HTTPException(status_code=403, detail="This administrative operation is disabled.")
    expected = production_settings.admin_token
    if production_settings.production_mode and not expected:
        raise HTTPException(
            status_code=503,
            detail="Set GESTURE_ADMIN_TOKEN before enabling administrative operations.",
        )
    if expected and not secrets.compare_digest(expected, supplied_token or ""):
        raise HTTPException(status_code=401, detail="Invalid administrative token.")


def _selectable_models() -> list[str]:
    status = engine.model_manager.status()
    selectable = list(status["selectable_models"])
    if production_settings.production_mode:
        selected = status["selected_model_name"]
        return [selected] if selected in selectable else []
    return selectable


def _onnx_qualification() -> dict[str, Any]:
    metadata_path = MODELS_DIRECTORY / "gesture_mlp_onnx_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {
            "status": "failed",
            "current": False,
            "pc_release_ready": False,
            "selected_model": "ONNX",
            "deferred_classes": [],
            "failing_gated_classes": [],
            "message": f"ONNX metadata is unavailable: {error}",
        }
    parity = metadata.get("parity") or {}
    quality = metadata.get("quality") or {}
    qualification = metadata.get("qualification") or {}
    release_checks = qualification.get("checks") or {}
    confirmation = qualification.get("independent_confirmation") or {}
    confirmation_checks = confirmation.get("checks") or {}
    selection = qualification.get("selection") or {}

    required_release_checks = {
        "runtime_macro_f1_at_least_0_97",
        "accepted_precision_at_least_0_99",
        "unknown_false_accept_rate_at_most_0_02",
        "fist_f1_at_least_0_95",
        "fist_recall_at_least_0_95",
        "thumb_down_f1_at_least_0_95",
        "thumb_down_recall_at_least_0_95",
        "thumb_down_within_0_02_of_like_f1",
        "thumb_down_like_confusion_at_most_0_02",
        "thumb_down_fist_confusion_at_most_0_02",
        "classifier_p95_below_5_ms",
    }
    required_confirmation_checks = {
        "new_subject_present_class_macro_f1_at_least_0_95",
        "new_subject_accepted_precision_at_least_0_99",
        "new_subject_unknown_false_accept_rate_at_most_0_02",
        "new_subject_fist_recall_at_least_0_95",
        "new_subject_thumb_down_recall_at_least_0_95",
        "new_subject_thumb_down_within_0_02_of_like_f1",
        "new_subject_upward_open_palm_recall_at_least_0_98",
    }

    def threshold_matches(name: str, expected: float) -> bool:
        try:
            return abs(float(selection[name]) - float(expected)) <= 1e-9
        except (KeyError, TypeError, ValueError):
            return False

    gates = {
        "schema_version_is_current": metadata.get("schema_version") == 3,
        "exact_ten_class_contract": (
            metadata.get("output_class_order") == runtime_config.class_names
        ),
        "onnx_parity": bool(parity.get("passed")),
        "qualification_passed": qualification.get("passed") is True,
        "complete_release_checks": required_release_checks <= set(release_checks),
        "complete_confirmation_checks": (
            required_confirmation_checks <= set(confirmation_checks)
        ),
        "selected_thresholds_match_runtime": (
            threshold_matches("known_mass_floor", runtime_config.known_mass_floor)
            and threshold_matches("confidence_floor", runtime_config.confidence_floor)
            and threshold_matches(
                "probability_margin_floor",
                runtime_config.probability_margin_floor,
            )
        ),
        "selection_matches_model": (
            selection.get("model_sha256") == metadata.get("onnx_sha256")
        ),
        **{name: release_checks.get(name) is True for name in sorted(required_release_checks)},
        **{
            name: confirmation_checks.get(name) is True
            for name in sorted(required_confirmation_checks)
        },
    }
    open_set = {
        "known_acceptance_rate": quality.get("known_acceptance_rate"),
        "unknown_false_acceptance_rate": quality.get(
            "unknown_false_acceptance_rate"
        ),
    }
    hard_cases = {
        "public_confusion_rates": qualification.get("confusion_rates") or {},
        "confirmation_confusion_rates": confirmation.get("confusion_rates") or {},
    }
    passed = all(gates.values())
    return {
        "status": "passed" if passed else "failed",
        "current": passed,
        "pc_release_ready": passed,
        "selected_model": "ONNX",
        "format": "ONNX",
        "created_utc": metadata.get("created_utc"),
        "source_model": metadata.get("model_name"),
        "class_names": metadata.get("output_class_order"),
        "deferred_classes": [],
        "failing_gates": [name for name, value in gates.items() if not value],
        "failing_gated_classes": [],
        "gates": gates,
        "quality": quality,
        "open_set_rejection": open_set,
        "hard_case_metrics": hard_cases,
        "gated_macro_f1": quality.get("macro_f1"),
        "gated_minimum_per_class_f1": quality.get("minimum_per_class_f1"),
        "onnx_parity": parity,
        "prediction_agreement": parity.get("prediction_agreement"),
        "maximum_absolute_probability_error": parity.get(
            "maximum_absolute_probability_error"
        ),
        "message": (
            "The exact ten-command ONNX graph passed parity, public-test, "
            "new-participant confirmation, threshold, and new-command gates."
            if passed
            else "One or more ONNX release qualification gates did not pass."
        ),
    }


def current_health() -> dict[str, Any]:
    engine_status = engine.status()
    engine_status["model"]["selectable_models"] = _selectable_models()
    engine_status["model"]["production_selection_locked"] = production_settings.production_mode
    integrity = artifact_registry.verify()
    operational_ready = bool(
        engine_status["ready"]
        and (integrity["verified"] or not production_settings.require_artifact_manifest)
    )
    release = _onnx_qualification()
    learning_status = online_learning.status()
    return {
        "status": "ready" if operational_ready else "setup_required",
        "operational_ready": operational_ready,
        "engine": engine_status,
        "config": runtime_config.public_dict(),
        "artifacts": artifact_status(),
        "artifact_integrity": integrity,
        "online_learning": {
            **learning_status,
            "tflite_policy": learning_status["policy"],
        },
        "production": {
            **production_settings.public_dict(),
            "release_qualification": release,
        },
        "runtime_metrics": runtime_metrics.snapshot(),
        "ncm_camera": ncm_camera.status(),
        "active_sessions": len(active_sessions),
        "configuration_error": configuration_error,
    }


@app.get("/livez")
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
def readiness() -> JSONResponse:
    health = current_health()
    return JSONResponse(health, status_code=200 if health["operational_ready"] else 503)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return current_health()


@app.get("/api/config")
def config() -> dict[str, Any]:
    return runtime_config.public_dict()


@app.get("/api/ncm/status")
def ncm_status() -> dict[str, Any]:
    return ncm_camera.status()


@app.post("/api/ncm/connect")
def ncm_connect() -> dict[str, Any]:
    ncm_camera.start()
    return {
        "ok": True,
        "message": (
            f"Connecting from {ncm_camera.config.host_ip} to "
            f"{ncm_camera.config.device_ip}:{ncm_camera.config.tcp_port}."
        ),
        "camera": ncm_camera.status(),
    }


@app.post("/api/ncm/disconnect")
def ncm_disconnect() -> dict[str, Any]:
    ncm_camera.stop()
    return {"ok": True, "camera": ncm_camera.status()}


@app.post("/api/ncm/discover")
async def ncm_discover() -> dict[str, Any]:
    return await asyncio.to_thread(ncm_camera.discover)


@app.get("/api/ncm/frame.jpg")
def ncm_frame() -> Response:
    frame_id, jpeg = ncm_camera.latest_frame()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="No JLIP JPEG frame has arrived yet.")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"X-NCM-Frame-ID": str(frame_id), "Cache-Control": "no-store"},
    )


@app.get("/api/ncm/stream.mjpg")
def ncm_stream() -> StreamingResponse:
    async def generate_stream():
        last_frame_id = -1
        while True:
            frame_id, jpeg = await asyncio.to_thread(
                ncm_camera.wait_for_frame, last_frame_id, 1.0
            )
            if jpeg is None or frame_id == last_frame_id:
                await asyncio.sleep(0.1)
                continue
            last_frame_id = frame_id
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"X-NCM-Frame-ID: {frame_id}\r\n".encode("ascii")
                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                + jpeg
                + b"\r\n"
            )

    return StreamingResponse(
        generate_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/models")
def models() -> dict[str, Any]:
    status = engine.model_manager.status()
    status["selectable_models"] = _selectable_models()
    status["production_selection_locked"] = production_settings.production_mode
    return status


@app.get("/api/artifacts")
def artifacts() -> dict[str, Any]:
    return {
        "models_directory": str(MODELS_DIRECTORY),
        "artifacts": artifact_status(),
        "integrity": artifact_registry.verify(),
    }


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    return load_metric_files()


@app.get("/api/runtime-metrics")
def runtime_metric_snapshot() -> dict[str, Any]:
    return runtime_metrics.snapshot()


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    return runtime_metrics.prometheus()


@app.get("/api/production-qualification")
def production_qualification() -> dict[str, Any]:
    return _onnx_qualification()


@app.post("/api/production-qualification/run")
async def run_production_qualification(
    x_admin_token: str | None = Header(default=None),
) -> dict[str, Any]:
    del x_admin_token
    return _onnx_qualification()


@app.get("/api/actions")
def actions() -> dict[str, Any]:
    return {"count": len(action_history), "recent": list(action_history)[::-1]}


@app.get("/api/feedback")
def feedback_summary() -> dict[str, Any]:
    return feedback_store.summary()


class CollectionParticipantPayload(BaseModel):
    participant_id: str = Field(min_length=1, max_length=48)


class CollectionSamplePayload(CollectionParticipantPayload):
    token: str = Field(min_length=32, max_length=32)
    session_id: str = Field(min_length=1, max_length=48)
    step_id: str = Field(max_length=80)
    handedness: Literal["left", "right", "none", "unspecified"] = "unspecified"
    finger_orientation: Literal["toward_camera", "away_from_camera", "none", "unspecified"] = "unspecified"
    lighting: Literal["normal", "dim", "backlit"] = "normal"
    distance: Literal["near", "medium", "far"] = "medium"
    note: str = Field(default="", max_length=500)


class CollectionDeleteSelectedPayload(CollectionParticipantPayload):
    sample_refs: list[str] = Field(min_length=1, max_length=100)


def _collection_call(request: Request, method, **kwargs):
    origin = request.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        raise HTTPException(status_code=403, detail="Origin is not allowed.")
    try:
        return method(**kwargs)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=503, detail=f"Collection storage unavailable: {error}") from error


@app.get("/api/collection")
def collection_summary(request: Request, participant_id: str | None = None):
    return _collection_call(request, collection_store.summary, participant_id=participant_id)


@app.post("/api/collection/participants")
def collection_participant(request: Request, payload: CollectionParticipantPayload):
    return _collection_call(request, collection_store.add_participant, **payload.model_dump())


@app.post("/api/collection/preview")
async def collection_preview(
    request: Request,
    source: Literal["ncm", "webcam"] = "ncm",
    image: UploadFile | None = File(default=None),
):
    if source == "ncm":
        if not ncm_camera.status()["connected"]:
            raise HTTPException(status_code=409, detail="Connect the board camera before capturing.")
        return _collection_call(request, collection_store.freeze)
    if image is None:
        raise HTTPException(status_code=400, detail="A webcam JPEG frame is required.")
    if image.content_type and not (
        image.content_type.startswith("image/")
        or image.content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=415, detail="Only image uploads are accepted.")
    content = await image.read(production_settings.max_image_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded webcam frame is empty.")
    if len(content) > production_settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="Webcam frame exceeds the configured size limit.")
    session = RuntimeSession(runtime_config)
    result = await _process_frame_with_limit(content, session, None)
    runtime_metrics.record_frame(result)
    return _collection_call(
        request,
        collection_store.freeze_bytes,
        frame_id=time.time_ns(),
        jpeg=content,
        result=result,
        source="webcam",
    )


@app.post("/api/collection/samples")
def collection_sample(request: Request, payload: CollectionSamplePayload):
    return _collection_call(request, collection_store.save, **payload.model_dump())


@app.get("/api/collection/samples")
def collection_samples(request: Request, participant_id: str, step_id: str | None = None):
    return _collection_call(
        request,
        collection_store.list_samples,
        participant_id=participant_id,
        step_id=step_id,
    )


@app.get("/api/collection/image")
def collection_image(request: Request, participant_id: str, sample_ref: str):
    image = _collection_call(
        request,
        collection_store.sample_image,
        participant_id=participant_id,
        sample_ref=sample_ref,
    )
    return Response(content=image, media_type="image/jpeg")


@app.post("/api/collection/samples/delete-selected")
def collection_delete_selected(request: Request, payload: CollectionDeleteSelectedPayload):
    return _collection_call(request, collection_store.delete_selected, **payload.model_dump())


@app.post("/api/collection/samples/delete-last")
def collection_delete_last(request: Request, payload: CollectionParticipantPayload):
    return _collection_call(request, collection_store.delete_last, **payload.model_dump())


@app.post("/api/feedback")
def save_feedback(payload: FeedbackPayload) -> dict[str, Any]:
    try:
        values = payload.model_dump()
        learning_mode = values.pop("learning_mode")
        base_probabilities = values.pop("base_probabilities")
        if learning_mode == "force":
            raise PermissionError(
                "Force learning is disabled; only validation-gated safe updates are allowed."
            )
        if learning_mode == "safe" and values.get("feature_vector") is None:
            raise ValueError("Safe learning requires a captured 76-D feature vector.")
        saved = feedback_store.save(FeedbackRecord(**values))
        if learning_mode == "audit":
            update: dict[str, Any] = {
                "status": "audit_only",
                "message": "Reviewed feedback saved without changing the live adapter.",
            }
        else:
            supplied_probabilities = base_probabilities or values["probabilities"] or None
            try:
                update = online_learning.learn(
                    values["feature_vector"],
                    values["actual_label"],
                    probabilities=supplied_probabilities,
                    mode="safe",
                    reviewed=True,
                )
            except (ValueError, TypeError, OSError, RuntimeError) as error:
                # The reviewed record was already committed. Report the adapter
                # failure in-band so clients do not retry and duplicate feedback.
                update = {
                    "status": "error",
                    "message": "Feedback was saved, but the safe update failed: " + str(error),
                }
        return {**saved, "learning_mode": learning_mode, "online_update": update}
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/online-learning/backups")
def online_backups() -> dict[str, Any]:
    backups = online_learning.list_backups()
    return {"count": len(backups), "backups": backups}


@app.post("/api/online-learning/rollback")
def rollback_online_adapter(
    payload: OnlineRollbackPayload,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _admin_allowed(production_settings.allow_artifact_reload, x_admin_token)
    try:
        return online_learning.rollback(payload.backup_name)
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/predict-image")
async def predict_image(
    image: UploadFile = File(...),
    model_name: str | None = None,
) -> dict[str, Any]:
    if model_name is not None and model_name not in _selectable_models():
        raise HTTPException(status_code=400, detail="Unknown model selection.")
    if image.content_type and not (
        image.content_type.startswith("image/") or image.content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=415, detail="Only image uploads are accepted.")
    content = await image.read(production_settings.max_image_bytes + 1)
    if len(content) > production_settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the configured size limit.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    session = RuntimeSession(runtime_config)
    result: dict[str, Any] = {}
    for _ in range(runtime_config.stable_frames_required):
        result = await _process_frame_with_limit(content, session, model_name)
        runtime_metrics.record_frame(result)
    return result


@app.post("/api/reload")
async def reload_artifacts(
    x_admin_token: str | None = Header(default=None),
) -> dict[str, Any]:
    global runtime_config, engine, feedback_store, online_learning
    _admin_allowed(production_settings.allow_artifact_reload, x_admin_token)
    async with service_lock:
        old_engine = engine
        runtime_config, engine, feedback_store, online_learning = _build_services()
        await inference_start_limiter.configure(runtime_config.frame_interval_ms)
        for session in active_sessions.values():
            session.reset()
        old_engine.close()
    return current_health()


@app.websocket("/ws/live")
async def live_socket(websocket: WebSocket) -> None:
    origin = websocket.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        await websocket.close(code=1008, reason="Origin is not allowed.")
        return
    if len(active_sessions) >= production_settings.max_active_sessions:
        await websocket.close(code=1013, reason="The local inference service is at capacity.")
        return
    await websocket.accept()
    session_id = f"session-{secrets.token_hex(8)}"
    session = RuntimeSession(runtime_config)
    active_sessions[session_id] = session
    runtime_metrics.session_opened()
    selected_model: str | None = None
    await websocket.send_json({
        "type": "hello",
        "session_id": session_id,
        "health": current_health(),
    })
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            frame = message.get("bytes")
            if frame is not None:
                if len(frame) > production_settings.max_websocket_frame_bytes:
                    runtime_metrics.record_error("frame_too_large")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Camera frame exceeds the configured size limit.",
                    })
                    continue
                try:
                    result = await _process_frame_with_limit(
                        frame, session, selected_model
                    )
                except Exception:
                    runtime_metrics.record_error("inference_exception")
                    logger.exception("Frame inference failed")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Frame inference failed; the session remains available.",
                    })
                    continue
                runtime_metrics.record_frame(result)
                if result.get("runtime_action") not in {None, "Wait / No Action"}:
                    action_history.append({
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "session_id": session_id,
                        "model": result.get("model"),
                        "action": result["runtime_action"],
                        "prediction": result.get("runtime_prediction"),
                        "confidence": result.get("confidence", 0.0),
                    })
                await websocket.send_json({"type": "prediction", **result})
            elif message.get("text"):
                try:
                    command = json.loads(message["text"])
                except json.JSONDecodeError:
                    command = {"type": message["text"]}
                command_type = command.get("type")
                if command_type == "reset":
                    session.reset()
                    await websocket.send_json({"type": "reset_complete"})
                elif command_type == "select_model":
                    requested = str(command.get("model", ""))
                    selectable = _selectable_models()
                    if requested not in selectable:
                        await websocket.send_json({"type": "error", "message": "Unknown model selection."})
                    else:
                        selected_model = requested
                        session.reset()
                        await websocket.send_json({"type": "model_selected", "model": requested})
                elif command_type == "start_follow_object":
                    session.temporal_gate.reset()
                    session.diagnostic_gates.clear()
                    update = session.follow_object.start()
                    await websocket.send_json({
                        "type": "follow_object_started",
                        "follow_object": {
                            name: getattr(update, name) for name in update.__dataclass_fields__
                        },
                    })
                elif command_type == "stop_follow_object":
                    update = session.follow_object.stop()
                    await websocket.send_json({
                        "type": "follow_object_stopped",
                        "follow_object": {
                            name: getattr(update, name) for name in update.__dataclass_fields__
                        },
                    })
                else:
                    await websocket.send_json({"type": "error", "message": "Unknown command."})
    except WebSocketDisconnect:
        pass
    finally:
        active_sessions.pop(session_id, None)
        runtime_metrics.session_closed()


@app.websocket("/ws/ncm-live")
async def ncm_live_socket(websocket: WebSocket) -> None:
    """Push predictions from the board camera; the browser sends no image data."""
    origin = websocket.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        await websocket.close(code=1008, reason="Origin is not allowed.")
        return
    if len(active_sessions) >= production_settings.max_active_sessions:
        await websocket.close(code=1013, reason="The local inference service is at capacity.")
        return

    await websocket.accept()
    ncm_camera.start()
    session_id = f"ncm-session-{secrets.token_hex(8)}"
    session = RuntimeSession(runtime_config)
    active_sessions[session_id] = session
    runtime_metrics.session_opened()
    send_lock = asyncio.Lock()
    selected_model: str | None = None

    async def send_json(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def prediction_producer() -> None:
        last_frame_id = -1
        last_status_sent_at = 0.0
        while True:
            frame_id, frame = await asyncio.to_thread(
                ncm_camera.wait_for_frame, last_frame_id, 1.0
            )
            if frame is None or frame_id == last_frame_id:
                now = time.monotonic()
                if now - last_status_sent_at >= 1.0:
                    await send_json({"type": "ncm_status", "camera": ncm_camera.status()})
                    last_status_sent_at = now
                await asyncio.sleep(0.1)
                continue
            last_frame_id = frame_id
            try:
                result = await _process_frame_with_limit(
                    frame,
                    session,
                    selected_model,
                    center_crop_ratio=runtime_config.roi_size_ratio,
                )
            except Exception:
                runtime_metrics.record_error("ncm_inference_exception")
                logger.exception("NCM camera frame inference failed")
                await send_json({
                    "type": "error",
                    "message": "NCM frame inference failed; reconnect is still active.",
                })
                continue
            camera_status = ncm_camera.status()
            last_status_sent_at = time.monotonic()
            result["camera_fps"] = camera_status["camera_fps"]
            result["ncm_camera"] = camera_status
            result["ncm_frame_id"] = frame_id
            collection_store.observe(frame_id, frame, result)
            runtime_metrics.record_frame(result)
            if result.get("runtime_action") not in {None, "Wait / No Action"}:
                action_history.append({
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "session_id": session_id,
                    "model": result.get("model"),
                    "action": result["runtime_action"],
                    "prediction": result.get("runtime_prediction"),
                    "confidence": result.get("confidence", 0.0),
                })
            await send_json({"type": "prediction", **result})

    async def command_consumer() -> None:
        nonlocal selected_model
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect
            text = message.get("text")
            if not text:
                continue
            try:
                command = json.loads(text)
            except json.JSONDecodeError:
                command = {"type": text}
            command_type = command.get("type")
            if command_type == "reset":
                session.reset()
                await send_json({"type": "reset_complete"})
            elif command_type == "select_model":
                requested = str(command.get("model", ""))
                if requested not in _selectable_models():
                    await send_json({"type": "error", "message": "Unknown model selection."})
                else:
                    selected_model = requested
                    session.reset()
                    await send_json({"type": "model_selected", "model": requested})
            elif command_type == "start_follow_object":
                session.temporal_gate.reset()
                session.diagnostic_gates.clear()
                update = session.follow_object.start()
                await send_json({
                    "type": "follow_object_started",
                    "follow_object": {
                        name: getattr(update, name) for name in update.__dataclass_fields__
                    },
                })
            elif command_type == "stop_follow_object":
                update = session.follow_object.stop()
                await send_json({
                    "type": "follow_object_stopped",
                    "follow_object": {
                        name: getattr(update, name) for name in update.__dataclass_fields__
                    },
                })
            else:
                await send_json({"type": "error", "message": "Unknown command."})

    await send_json({
        "type": "hello",
        "session_id": session_id,
        "health": current_health(),
        "camera": ncm_camera.status(),
    })
    producer = asyncio.create_task(prediction_producer())
    consumer = asyncio.create_task(command_consumer())
    try:
        done, pending = await asyncio.wait(
            {producer, consumer}, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            exception = task.exception()
            if exception is not None:
                raise exception
        for task in pending:
            task.cancel()
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        producer.cancel()
        consumer.cancel()
        await asyncio.gather(producer, consumer, return_exceptions=True)
        active_sessions.pop(session_id, None)
        runtime_metrics.session_closed()
        if not any(name.startswith("ncm-session-") for name in active_sessions):
            await asyncio.to_thread(ncm_camera.stop)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Gesture Control Lab ONNX + NCM Camera API",
        "version": app.version,
        "dashboard": "Start the frontend and open http://127.0.0.1:3200",
        "docs": "http://127.0.0.1:8200/docs",
    }
