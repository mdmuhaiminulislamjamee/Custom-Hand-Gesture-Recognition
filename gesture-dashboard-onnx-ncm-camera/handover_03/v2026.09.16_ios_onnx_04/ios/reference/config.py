from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIRECTORY = Path(os.getenv("GESTURE_MODELS_DIR", PROJECT_ROOT / "models"))
FEEDBACK_DIRECTORY = Path(os.getenv("GESTURE_FEEDBACK_DIR", PROJECT_ROOT / "feedback"))


def _environment_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, 1/0, yes/no, or on/off.")


def _environment_integer(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value

CLASS_NAMES = [
    "left", "right", "up", "down", "open_palm", "like", "dorsal", "ok",
    "fist", "thumb_down",
]
REJECT_LABEL = "no_gesture"
FEEDBACK_LABELS = [*CLASS_NAMES, REJECT_LABEL]

GESTURE_TO_ACTION = {
    "left": "Move Left",
    "right": "Move Right",
    "up": "Move Up",
    "down": "Move Down",
    "open_palm": "Enable / Disable Object Tracking",
    "like": "Play / Pause",
    "dorsal": "Return to Default Position",
    "ok": "Start / Stop Recording",
    "fist": "Mute",
    "thumb_down": "Volume Down",
}

FEATURE_NAMES = (
    [f"landmark_{index}_{axis}" for index in range(21) for axis in ("x", "y")]
    + [f"landmark_{index}_radius" for index in range(21)]
    + ["index_segment_dx", "index_segment_dy", "wrist_index_dx", "wrist_index_dy"]
    + ["thumb_index_gap_ratio"]
    + [
        "thumb_extension", "index_extension", "middle_extension",
        "ring_extension", "pinky_extension", "other_fingers_folded",
        "thumb_tip_index_mcp_ratio", "thumb_reach_ratio",
    ]
)
assert len(FEATURE_NAMES) == 76


@dataclass(frozen=True, slots=True)
class ProductionSettings:
    """Operational controls that do not alter classifier behavior."""

    environment: str
    require_artifact_manifest: bool
    allow_force_learning: bool
    allow_artifact_reload: bool
    admin_token: str
    max_image_bytes: int
    max_websocket_frame_bytes: int
    max_active_sessions: int
    metrics_window_size: int
    minimum_macro_f1: float
    minimum_per_class_f1: float
    deferred_quality_classes: tuple[str, ...]
    log_directory: Path

    @property
    def production_mode(self) -> bool:
        return self.environment == "production"

    def public_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "production_mode": self.production_mode,
            "require_artifact_manifest": self.require_artifact_manifest,
            "allow_force_learning": self.allow_force_learning,
            "allow_artifact_reload": self.allow_artifact_reload,
            "admin_token_configured": bool(self.admin_token),
            "max_image_bytes": self.max_image_bytes,
            "max_websocket_frame_bytes": self.max_websocket_frame_bytes,
            "max_active_sessions": self.max_active_sessions,
            "metrics_window_size": self.metrics_window_size,
            "quality_gate": {
                "minimum_macro_f1": self.minimum_macro_f1,
                "minimum_per_class_f1": self.minimum_per_class_f1,
                "deferred_classes": list(self.deferred_quality_classes),
            },
        }


def load_production_settings() -> ProductionSettings:
    environment = os.getenv("GESTURE_ENVIRONMENT", "production").strip().lower()
    if environment not in {"development", "test", "production"}:
        raise ValueError("GESTURE_ENVIRONMENT must be development, test, or production.")
    default_manifest = environment == "production"
    log_directory = Path(
        os.getenv("GESTURE_LOG_DIR", str(PROJECT_ROOT / ".runtime" / "logs"))
    ).resolve()
    return ProductionSettings(
        environment=environment,
        require_artifact_manifest=_environment_flag(
            "GESTURE_REQUIRE_ARTIFACT_MANIFEST", default_manifest
        ),
        allow_force_learning=_environment_flag("GESTURE_ALLOW_FORCE_LEARNING", False),
        allow_artifact_reload=_environment_flag(
            "GESTURE_ALLOW_ARTIFACT_RELOAD", environment != "production"
        ),
        admin_token=os.getenv("GESTURE_ADMIN_TOKEN", "").strip(),
        max_image_bytes=_environment_integer(
            "GESTURE_MAX_IMAGE_BYTES", 8_000_000, 100_000, 25_000_000
        ),
        max_websocket_frame_bytes=_environment_integer(
            "GESTURE_MAX_WEBSOCKET_FRAME_BYTES", 4_000_000, 100_000, 15_000_000
        ),
        max_active_sessions=_environment_integer(
            "GESTURE_MAX_ACTIVE_SESSIONS", 4, 1, 32
        ),
        metrics_window_size=_environment_integer(
            "GESTURE_METRICS_WINDOW_SIZE", 1000, 50, 100_000
        ),
        minimum_macro_f1=float(os.getenv("GESTURE_MINIMUM_MACRO_F1", "0.90")),
        minimum_per_class_f1=float(
            os.getenv("GESTURE_MINIMUM_PER_CLASS_F1", "0.80")
        ),
        deferred_quality_classes=(),
        log_directory=log_directory,
    )


@dataclass(slots=True)
class RuntimeConfig:
    class_names: list[str] = field(default_factory=lambda: CLASS_NAMES.copy())
    gesture_to_action: dict[str, str] = field(
        default_factory=lambda: GESTURE_TO_ACTION.copy()
    )
    feature_names: list[str] = field(default_factory=lambda: FEATURE_NAMES.copy())
    feedback_labels: list[str] = field(default_factory=lambda: FEEDBACK_LABELS.copy())
    reject_label: str = REJECT_LABEL
    target_fps: float = 10.0
    ema_alpha: float = 0.45
    confidence_floor: float = 0.70
    probability_margin_floor: float = 0.08
    known_mass_floor: float = 0.70
    geometry_recovery_mass_floor: float = 0.15
    stable_frames_required: int = 3
    release_frames_required: int = 3
    minimum_hold_seconds: float = 0.18
    action_cooldown_seconds: float = 0.80
    follow_timeout_seconds: float = 20.0
    follow_hold_seconds: float = 2.5
    follow_session_timeout_seconds: float = 120.0
    follow_success_display_seconds: float = 8.0
    follow_min_confidence: float = 0.70
    roi_size_ratio: float = 0.92
    model_file: str = "gesture_mlp_production.onnx"
    model_format: str = "ONNX float32 via ONNX Runtime CPUExecutionProvider"
    selected_model_name: str = "ONNX"
    zoom_gap_calibration: dict[str, float] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.class_names)}

    @property
    def frame_interval_ms(self) -> int:
        return max(1, round(1000 / max(self.target_fps, 0.1)))

    @property
    def frame_budget_ms(self) -> float:
        return 1000.0 / max(self.target_fps, 0.1)

    def public_dict(self) -> dict[str, Any]:
        return {
            "class_names": self.class_names,
            "feedback_labels": self.feedback_labels,
            "reject_label": self.reject_label,
            "gesture_to_action": self.gesture_to_action,
            "feature_count": len(self.feature_names),
            "target_fps": self.target_fps,
            "frame_interval_ms": self.frame_interval_ms,
            "frame_budget_ms": self.frame_budget_ms,
            "ema_alpha": self.ema_alpha,
            "confidence_floor": self.confidence_floor,
            "probability_margin_floor": self.probability_margin_floor,
            "known_mass_floor": self.known_mass_floor,
            "geometry_recovery_mass_floor": self.geometry_recovery_mass_floor,
            "stable_frames_required": self.stable_frames_required,
            "release_frames_required": self.release_frames_required,
            "minimum_hold_seconds": self.minimum_hold_seconds,
            "action_cooldown_seconds": self.action_cooldown_seconds,
            "follow_timeout_seconds": self.follow_timeout_seconds,
            "follow_hold_seconds": self.follow_hold_seconds,
            "follow_session_timeout_seconds": self.follow_session_timeout_seconds,
            "follow_success_display_seconds": self.follow_success_display_seconds,
            "follow_min_confidence": self.follow_min_confidence,
            "roi_size_ratio": self.roi_size_ratio,
            "model_file": self.model_file,
            "model_format": self.model_format,
            "selected_model_name": self.selected_model_name,
            "zoom_gap_calibration": self.zoom_gap_calibration,
            "follow_object_sequence": ["open_palm"],
            "follow_object_base_models": ["ONNX"],
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_runtime_config(models_directory: Path = MODELS_DIRECTORY) -> RuntimeConfig:
    path = models_directory / "gesture_mobile_runtime_config.json"
    data = _read_json(path)
    smoothing = data.get("temporal_smoothing", {})
    class_names = list(data.get("class_names") or CLASS_NAMES)
    if class_names != CLASS_NAMES:
        raise ValueError(
            "Runtime configuration class order does not match the ten-command "
            f"taxonomy: {class_names}"
        )
    feature_names = list(data.get("feature_names") or FEATURE_NAMES)
    if feature_names != FEATURE_NAMES:
        raise ValueError("Runtime configuration does not contain the exact 76-D feature order.")
    feedback_labels = list(data.get("feedback_labels") or FEEDBACK_LABELS)
    if feedback_labels != FEEDBACK_LABELS:
        raise ValueError("Feedback labels must contain the ten commands plus no_gesture.")
    reject_label = str(data.get("reject_label", REJECT_LABEL))
    if reject_label != REJECT_LABEL:
        raise ValueError("The runtime rejection label must be no_gesture.")
    gesture_to_action = dict(data.get("gesture_to_action") or GESTURE_TO_ACTION)
    if gesture_to_action != GESTURE_TO_ACTION:
        raise ValueError("Gesture actions do not match the ten-command release contract.")
    return RuntimeConfig(
        class_names=class_names,
        gesture_to_action=gesture_to_action,
        feature_names=feature_names,
        feedback_labels=feedback_labels,
        reject_label=reject_label,
        target_fps=float(data.get("target_fps", 10.0)),
        ema_alpha=float(smoothing.get("alpha", 0.45)),
        confidence_floor=float(smoothing.get("confidence_floor", 0.70)),
        probability_margin_floor=float(
            smoothing.get("probability_margin_floor", 0.08)
        ),
        known_mass_floor=float(smoothing.get("known_mass_floor", 0.70)),
        geometry_recovery_mass_floor=float(
            smoothing.get("geometry_recovery_mass_floor", 0.15)
        ),
        stable_frames_required=int(smoothing.get("stable_frames_required", 3)),
        release_frames_required=int(smoothing.get("release_frames_required", 3)),
        minimum_hold_seconds=float(smoothing.get("minimum_hold_seconds", 0.18)),
        action_cooldown_seconds=float(data.get("action_cooldown_seconds", 0.80)),
        follow_timeout_seconds=float(
            data.get("follow_object_step_timeout_seconds", 20.0)
        ),
        follow_hold_seconds=float(data.get("follow_object_hold_seconds", 2.5)),
        follow_session_timeout_seconds=float(
            data.get("follow_object_session_timeout_seconds", 120.0)
        ),
        follow_success_display_seconds=float(
            data.get("follow_object_success_display_seconds", 8.0)
        ),
        follow_min_confidence=float(
            data.get("follow_object_base_consensus_min_confidence", 0.70)
        ),
        roi_size_ratio=float(data.get("roi_size_ratio", 0.92)),
        model_file="gesture_mlp_production.onnx",
        model_format="ONNX float32 via ONNX Runtime CPUExecutionProvider",
        selected_model_name="ONNX",
        zoom_gap_calibration=data.get("zoom_gap_calibration"),
        raw=data,
    )
