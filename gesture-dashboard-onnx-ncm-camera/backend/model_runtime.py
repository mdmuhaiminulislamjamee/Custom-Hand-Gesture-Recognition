from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import MODELS_DIRECTORY, RuntimeConfig
from .follow_object import FollowObjectStateMachine
from .geometry import GeometryResolver, landmarks_to_feature
from .runtime import TemporalGate
from .hand_detection import HandDetector, DETECTOR_SETTINGS


MIN_HAND_AREA_RATIO = 0.0001
LANDMARK_SLOW_ALPHA = 0.18
LANDMARK_FAST_ALPHA = 0.72
LANDMARK_MOTION_GAIN = 1.35
MAX_LANDMARK_SHAPE_JUMP = 0.62
MAX_LANDMARK_CENTROID_JUMP = 1.20


@dataclass(slots=True)
class RuntimeSession:
    config: RuntimeConfig
    temporal_gate: TemporalGate = field(init=False)
    follow_object: FollowObjectStateMachine = field(init=False)
    diagnostic_gates: dict[str, TemporalGate] = field(default_factory=dict)
    frame_times: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    last_action_gesture: str | None = None
    last_action_at: float = 0.0
    _action_release_frames: int = field(default=0, init=False, repr=False)
    _smoothed_landmarks: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _last_raw_landmarks: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _pending_jump_landmarks: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _pending_jump_frames: int = field(default=0, init=False, repr=False)
    _last_landmark_at: float | None = field(default=None, init=False, repr=False)
    _missing_landmark_frames: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.temporal_gate = TemporalGate(self.config)
        self.follow_object = FollowObjectStateMachine(
            timeout_seconds=self.config.follow_timeout_seconds,
            hold_seconds=self.config.follow_hold_seconds,
            session_timeout_seconds=self.config.follow_session_timeout_seconds,
        )

    def reset(self) -> None:
        self.temporal_gate.reset()
        self.follow_object.reset()
        self.diagnostic_gates.clear()
        self.frame_times.clear()
        self.last_action_gesture = None
        self.last_action_at = 0.0
        self._action_release_frames = 0
        self.reset_landmark_tracking()

    def record_frame(self, now: float) -> float:
        self.frame_times.append(now)
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / elapsed if elapsed > 1e-9 else 0.0

    def diagnostic_gate(self, model_name: str) -> TemporalGate:
        if model_name not in self.diagnostic_gates:
            self.diagnostic_gates[model_name] = TemporalGate(self.config)
        return self.diagnostic_gates[model_name]

    def observe_action_release(self, gesture: str | None, *, tracking_gap: bool = False) -> bool:
        """Re-arm a latched toggle only after a sustained gesture release."""
        if self.last_action_gesture is None:
            self._action_release_frames = 0
            return False
        if gesture == self.last_action_gesture:
            self._action_release_frames = 0
            return False
        self._action_release_frames += 1
        required = max(1, int(self.config.release_frames_required))
        if tracking_gap:
            # Brief low-resolution detector dropouts must not re-trigger a held
            # toggle when the same hand returns. No action is emitted in gaps.
            required = max(required, int(np.ceil(self.config.target_fps * .6)))
        if self._action_release_frames < required:
            return False
        self.last_action_gesture = None
        self._action_release_frames = 0
        return True

    def reset_landmark_tracking(self) -> None:
        """Clear only the camera/landmark filter state for this client session."""
        self._smoothed_landmarks = None
        self._last_raw_landmarks = None
        self._pending_jump_landmarks = None
        self._pending_jump_frames = 0
        self._last_landmark_at = None
        self._missing_landmark_frames = 0

    @staticmethod
    def _landmark_scale(points: np.ndarray) -> float:
        width = float(np.ptp(points[:, 0]))
        height = float(np.ptp(points[:, 1]))
        return max(float(np.hypot(width, height)), 0.04)

    @classmethod
    def _landmark_motion(
        cls, previous: np.ndarray, current: np.ndarray
    ) -> tuple[float, float, float]:
        previous_scale = cls._landmark_scale(previous)
        current_scale = cls._landmark_scale(current)
        reference_scale = max((previous_scale + current_scale) * 0.5, 0.04)
        point_motion = float(
            np.median(np.linalg.norm(current - previous, axis=1)) / reference_scale
        )
        centroid_shift = float(
            np.linalg.norm(current.mean(axis=0) - previous.mean(axis=0))
            / reference_scale
        )
        previous_shape = (previous - previous.mean(axis=0)) / previous_scale
        current_shape = (current - current.mean(axis=0)) / current_scale
        shape_motion = float(
            np.median(np.linalg.norm(current_shape - previous_shape, axis=1))
        )
        return point_motion, centroid_shift, shape_motion

    def observe_missing_landmarks(self, now: float) -> dict[str, Any]:
        """Age the session filter and reset it after a short tracking outage."""
        self._missing_landmark_frames += 1
        elapsed = (
            None
            if self._last_landmark_at is None
            else max(0.0, float(now) - self._last_landmark_at)
        )
        reset = bool(
            self._smoothed_landmarks is not None
            and (
                self._missing_landmark_frames >= 3
                or (elapsed is not None and elapsed >= 0.35)
            )
        )
        missing_frames = self._missing_landmark_frames
        if reset:
            self.reset_landmark_tracking()
        return {
            "accepted": False,
            "reason": "no_landmarks",
            "filter_initialized": self._smoothed_landmarks is not None,
            "filter_reset": reset,
            "missing_frames": missing_frames,
            "seconds_since_landmarks": elapsed,
            "jump_rejected": False,
            "smoothing_alpha": None,
            "point_motion_ratio": None,
            "centroid_shift_ratio": None,
            "shape_motion_ratio": None,
        }

    def stabilize_landmarks(
        self, landmarks: np.ndarray, now: float
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Validate and adaptively smooth one MediaPipe landmark observation.

        The filter is deliberately session-local: browser and board-camera clients do
        not share smoothing history. An isolated implausible jump is ignored, while a
        consistent jump on two consecutive frames is treated as a genuine re-acquire.
        """
        points = np.asarray(landmarks, dtype=np.float32)
        base_diagnostics: dict[str, Any] = {
            "accepted": False,
            "reason": "invalid_landmarks",
            "filter_initialized": self._smoothed_landmarks is not None,
            "filter_reset": False,
            "missing_frames": 0,
            "seconds_since_landmarks": 0.0,
            "jump_rejected": False,
            "smoothing_alpha": None,
            "point_motion_ratio": None,
            "centroid_shift_ratio": None,
            "shape_motion_ratio": None,
        }
        if points.shape != (21, 2) or not np.isfinite(points).all():
            return None, base_diagnostics

        width = float(np.ptp(points[:, 0]))
        height = float(np.ptp(points[:, 1]))
        diagonal = float(np.hypot(width, height))
        area = float(width * height)
        inside_fraction = float(
            np.mean(
                (points[:, 0] >= -0.08)
                & (points[:, 0] <= 1.08)
                & (points[:, 1] >= -0.08)
                & (points[:, 1] <= 1.08)
            )
        )
        base_diagnostics.update({
            "bbox_width": width,
            "bbox_height": height,
            "bbox_area": area,
            "bbox_diagonal": diagonal,
            "inside_frame_fraction": inside_fraction,
        })
        # Tiny or substantially clipped landmark clouds are not reliable enough for
        # the classifier and are a common source of background false positives.
        if diagonal < 0.025 or area < MIN_HAND_AREA_RATIO or inside_fraction < 0.86:
            base_diagnostics["reason"] = "implausible_landmark_geometry"
            return None, base_diagnostics

        now = float(now)
        stale_reset = bool(
            self._last_landmark_at is not None
            and now - self._last_landmark_at >= 0.50
        )
        if stale_reset:
            self.reset_landmark_tracking()
            base_diagnostics["filter_reset"] = True

        if self._smoothed_landmarks is None or self._last_raw_landmarks is None:
            self._smoothed_landmarks = points.copy()
            self._last_raw_landmarks = points.copy()
            self._last_landmark_at = now
            self._missing_landmark_frames = 0
            base_diagnostics.update({
                "accepted": True,
                "reason": "filter_initialized",
                "filter_initialized": True,
                "smoothing_alpha": 1.0,
            })
            return self._smoothed_landmarks.copy(), base_diagnostics

        point_motion, centroid_shift, shape_motion = self._landmark_motion(
            self._last_raw_landmarks, points
        )
        base_diagnostics.update({
            "point_motion_ratio": point_motion,
            "centroid_shift_ratio": centroid_shift,
            "shape_motion_ratio": shape_motion,
        })
        is_jump = bool(
            shape_motion > MAX_LANDMARK_SHAPE_JUMP
            or centroid_shift > MAX_LANDMARK_CENTROID_JUMP
        )
        if is_jump:
            pending_is_consistent = False
            if self._pending_jump_landmarks is not None:
                pending_point, pending_centroid, pending_shape = self._landmark_motion(
                    self._pending_jump_landmarks, points
                )
                pending_is_consistent = bool(
                    pending_point < 0.30
                    and pending_centroid < 0.38
                    and pending_shape < 0.28
                )
            if pending_is_consistent:
                self._pending_jump_frames += 1
            else:
                self._pending_jump_landmarks = points.copy()
                self._pending_jump_frames = 1
            if self._pending_jump_frames < 2:
                base_diagnostics.update({
                    "reason": "isolated_landmark_jump",
                    "jump_rejected": True,
                    "pending_jump_frames": self._pending_jump_frames,
                })
                return None, base_diagnostics

            # The new observation persisted, so start a clean track at its position.
            self._smoothed_landmarks = points.copy()
            self._last_raw_landmarks = points.copy()
            self._pending_jump_landmarks = None
            self._pending_jump_frames = 0
            self._last_landmark_at = now
            self._missing_landmark_frames = 0
            base_diagnostics.update({
                "accepted": True,
                "reason": "persistent_jump_reacquired",
                "filter_initialized": True,
                "filter_reset": True,
                "smoothing_alpha": 1.0,
                "pending_jump_frames": 0,
            })
            return self._smoothed_landmarks.copy(), base_diagnostics

        self._pending_jump_landmarks = None
        self._pending_jump_frames = 0
        # Strong smoothing at rest removes landmark shimmer. The filter becomes more
        # responsive as real hand motion increases, avoiding a sluggish control feel.
        alpha = float(
            np.clip(
                LANDMARK_SLOW_ALPHA + LANDMARK_MOTION_GAIN * point_motion,
                LANDMARK_SLOW_ALPHA,
                LANDMARK_FAST_ALPHA,
            )
        )
        # Keep the same smoothing duration when the incoming frame rate varies.
        elapsed = max(0.001, min(0.5, now - self._last_landmark_at))
        alpha = 1.0 - (1.0 - alpha) ** (elapsed * self.config.target_fps)
        self._smoothed_landmarks = (
            alpha * points + (1.0 - alpha) * self._smoothed_landmarks
        ).astype(np.float32)
        self._last_raw_landmarks = points.copy()
        self._last_landmark_at = now
        self._missing_landmark_frames = 0
        base_diagnostics.update({
            "accepted": True,
            "reason": "adaptively_smoothed",
            "filter_initialized": True,
            "smoothing_alpha": alpha,
        })
        return self._smoothed_landmarks.copy(), base_diagnostics


_DLL_DIRECTORY_HANDLES: list[Any] = []


def _prepare_windows_dll_search() -> None:
    """Make optional native-runtime directories visible before importing ORT."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidates = [
        os.getenv("GESTURE_MSVC_RUNTIME_DIR", ""),
        str(Path(sys.base_prefix)),
        str(Path(sys.base_prefix) / "Library" / "bin"),
        str(Path(sys.executable).resolve().parent),
    ]
    for value in candidates:
        if not value:
            continue
        directory = Path(value).expanduser()
        if not directory.is_dir():
            continue
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory.resolve())))
        except OSError:
            continue


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class OnnxClassifier:
    """Strict ONNX Runtime wrapper for the qualified 76-D gesture MLP."""

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        config: RuntimeConfig,
    ):
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing ONNX classifier: {model_path.name}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing ONNX metadata: {metadata_path.name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "ONNX":
            raise ValueError("Classifier metadata does not declare ONNX format.")
        if list(metadata.get("output_class_order", [])) != config.class_names:
            raise ValueError("ONNX output class order differs from the eight-command contract.")
        if list(metadata.get("feature_names", [])) != config.feature_names:
            raise ValueError("ONNX feature order differs from the 76-D contract.")
        expected_digest = str(metadata.get("onnx_sha256", "")).lower()
        actual_digest = _sha256(model_path)
        if not expected_digest or actual_digest != expected_digest:
            raise ValueError("ONNX model hash does not match its qualification metadata.")
        parity = metadata.get("parity") or {}
        if parity.get("passed") is not True:
            raise ValueError("ONNX parity qualification is not marked as passed.")

        _prepare_windows_dll_search()
        try:
            import onnxruntime as ort
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "ONNX Runtime could not load. Install requirements.txt and the "
                "Microsoft Visual C++ 2015-2022 x64 Redistributable."
            ) from error

        options = ort.SessionOptions()
        # A thread pool costs more than it saves for this small single-hand MLP.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected one ONNX input, found {len(inputs)}.")
        model_input = inputs[0]
        if model_input.name != metadata["input"]["name"]:
            raise ValueError("ONNX input name differs from metadata.")
        if model_input.type != "tensor(float)":
            raise ValueError(f"Expected float32 ONNX input, found {model_input.type}.")
        if len(model_input.shape) != 2 or model_input.shape[1] != len(config.feature_names):
            raise ValueError(f"Expected ONNX input shape [N, 76], found {model_input.shape}.")
        self.input_name = model_input.name
        self.classes_ = np.arange(len(config.class_names), dtype=int)
        self.metadata = metadata
        self.model_path = model_path

        smoke_outputs = self.session.run(
            None,
            {self.input_name: np.zeros((1, len(config.feature_names)), dtype=np.float32)},
        )
        output_descriptors = self.session.get_outputs()
        probability_indexes = [
            index
            for index, value in enumerate(smoke_outputs)
            if isinstance(value, np.ndarray)
            and value.shape == (1, len(config.class_names))
            and np.issubdtype(value.dtype, np.floating)
        ]
        if len(probability_indexes) != 1:
            shapes = [getattr(value, "shape", None) for value in smoke_outputs]
            raise ValueError(
                "Could not identify the unique eight-class ONNX probability output; "
                f"observed shapes: {shapes}."
            )
        self.probability_output_name = output_descriptors[probability_indexes[0]].name
        configured_mass_name = str(metadata.get("known_mass_output", ""))
        mass_outputs = [
            descriptor.name
            for descriptor, value in zip(output_descriptors, smoke_outputs)
            if descriptor.name == configured_mass_name
            and isinstance(value, np.ndarray)
            and value.shape == (1, 1)
            and np.issubdtype(value.dtype, np.floating)
        ]
        if len(mass_outputs) != 1:
            raise ValueError("The ONNX graph is missing its open-set known-mass output.")
        self.known_mass_output_name = mass_outputs[0]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities, _ = self.predict_with_quality(features)
        return probabilities

    def predict_with_quality(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(features, dtype=np.float32).reshape(-1, 76)
        result, known_mass = self.session.run(
            [self.probability_output_name, self.known_mass_output_name],
            {self.input_name: rows},
        )
        probabilities = np.asarray(result, dtype=np.float64)
        if probabilities.shape != (len(rows), len(self.classes_)):
            raise RuntimeError(f"Unexpected ONNX probability shape: {probabilities.shape}")
        if not np.isfinite(probabilities).all():
            raise RuntimeError("ONNX Runtime returned non-finite probabilities.")
        probabilities = np.clip(probabilities, 0.0, None)
        totals = probabilities.sum(axis=1, keepdims=True)
        if (totals <= 1e-12).any():
            raise RuntimeError("ONNX Runtime returned an empty probability distribution.")
        mass = np.asarray(known_mass, dtype=np.float64).reshape(-1)
        if mass.shape != (len(rows),) or not np.isfinite(mass).all():
            raise RuntimeError(f"Unexpected ONNX known-mass shape: {mass.shape}")
        return probabilities / totals, np.clip(mass, 0.0, 1.0)


class ModelManager:
    def __init__(
        self,
        config: RuntimeConfig,
        models_directory: Path = MODELS_DIRECTORY,
        *,
        load_models: bool = True,
    ):
        self.config = config
        self.models_directory = models_directory
        self.models: dict[str, Any] = {}
        self.selected_model_name = config.selected_model_name
        self.errors: list[str] = []
        self.bundle_metadata: dict[str, Any] = {}
        self.online_adapter: Any | None = None
        self._lock = threading.RLock()
        if load_models:
            self._load()
        else:
            self.errors.append(
                "Classifier loading was blocked because artifact integrity verification failed."
            )

    def _load(self) -> None:
        model_path = self.models_directory / "gesture_mlp_production.onnx"
        metadata_path = self.models_directory / "gesture_mlp_onnx_metadata.json"
        try:
            classifier = OnnxClassifier(model_path, metadata_path, self.config)
            self.models["ONNX"] = classifier
            self.selected_model_name = "ONNX"
            self.bundle_metadata = {
                "format": "ONNX",
                "runtime": "ONNX Runtime CPUExecutionProvider",
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "model_sha256": classifier.metadata.get("onnx_sha256"),
                "source_model": classifier.metadata.get("model_name"),
                "parity": classifier.metadata.get("parity"),
                "opsets": classifier.metadata.get("opsets"),
            }
        except Exception as error:
            self.errors.append(f"Could not load ONNX classifier: {error}")

    @property
    def ready(self) -> bool:
        return bool(self.models)

    def status(self) -> dict[str, Any]:
        selectable = list(self.models)
        return {
            "ready": self.ready,
            "available_models": list(self.models),
            "selectable_models": selectable,
            "selected_model_name": self.selected_model_name,
            "follow_object_ensemble_ready": False,
            "follow_object_runtime": "ONNX" if self.ready else None,
            "errors": self.errors,
            "bundle": self.bundle_metadata,
        }

    def resolve_name(self, model_name: str | None = None) -> str:
        if not self.models:
            raise RuntimeError("No trained classifier artifact is available.")
        if model_name is not None and model_name not in self.models:
            raise ValueError(f"Unknown model selection: {model_name}")
        return model_name if model_name is not None else self.selected_model_name

    def set_online_adapter(self, adapter: Any | None) -> None:
        with self._lock:
            self.online_adapter = adapter

    def predict_detailed(
        self, feature: np.ndarray, model_name: str | None = None
    ) -> tuple[np.ndarray, str, float, dict[str, Any]]:
        feature_row = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        with self._lock:
            name = self.resolve_name(model_name)
            model = self.models[name]
            started = time.perf_counter()
            if hasattr(model, "predict_with_quality"):
                probability_rows, mass_rows = model.predict_with_quality(feature_row)
                known_mass = float(np.asarray(mass_rows).reshape(-1)[0])
            else:
                probability_rows = model.predict_proba(feature_row)
                known_mass = 1.0
            probabilities = np.asarray(probability_rows[0], dtype=np.float64)
            base_probabilities = probabilities.copy()
            base_known_mass = known_mass
            adapter_applied = False
            adapter_quality: dict[str, Any] = {}
            if self.online_adapter is not None:
                try:
                    adapted, adapter_quality = self.online_adapter.apply_with_quality(
                        probabilities,
                        feature_row[0],
                        known_gesture_mass=known_mass,
                    )
                    probabilities = np.asarray(adapted, dtype=np.float64).reshape(-1)
                    known_mass = float(adapter_quality["known_gesture_mass"])
                    adapter_applied = bool(
                        not np.allclose(probabilities, base_probabilities, atol=1e-12)
                        or not np.isclose(
                            known_mass,
                            base_known_mass,
                            atol=1e-12,
                        )
                    )
                except (KeyError, TypeError, ValueError, RuntimeError) as error:
                    # The immutable ONNX graph remains available if the optional
                    # adapter state is invalid or an update is being replaced.
                    probabilities = base_probabilities.copy()
                    known_mass = base_known_mass
                    adapter_quality = {"state_error": str(error)}
            elapsed_ms = (time.perf_counter() - started) * 1000
        classes = np.asarray(getattr(model, "classes_", np.arange(len(probabilities))), dtype=int)
        if not np.array_equal(classes, np.arange(len(self.config.class_names))):
            reordered = np.zeros(len(self.config.class_names), dtype=np.float64)
            reordered[classes] = probabilities
            probabilities = reordered
        probabilities /= max(float(probabilities.sum()), 1e-12)
        return probabilities, name, elapsed_ms, {
            "known_gesture_mass": known_mass,
            "online_adapter_applied": adapter_applied,
            "base_probabilities": base_probabilities,
            "online_adapter": adapter_quality,
        }

    def predict(
        self, feature: np.ndarray, model_name: str | None = None
    ) -> tuple[np.ndarray, str, float]:
        probabilities, name, elapsed_ms, _ = self.predict_detailed(feature, model_name)
        return probabilities, name, elapsed_ms

    def predict_many(
        self, feature: np.ndarray, model_names: list[str]
    ) -> dict[str, tuple[np.ndarray, float, dict[str, Any]]]:
        results: dict[str, tuple[np.ndarray, float, dict[str, Any]]] = {}
        for name in dict.fromkeys(model_names):
            if name not in self.models:
                continue
            probabilities, _, elapsed_ms, quality = self.predict_detailed(feature, name)
            results[name] = (probabilities, elapsed_ms, quality)
        return results


def adaptive_low_light_preprocess(
    bgr_image: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Enhance only under-exposed frames and return measurable image quality.

    The input orientation is intentionally preserved. In particular, this function
    never flips the NCM frame, so directional gestures retain camera coordinates.
    """
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - process_frame reports this first
        raise RuntimeError("OpenCV is required for camera preprocessing.") from error

    image = np.asarray(bgr_image)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Camera preprocessing requires a non-empty BGR image.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(image)

    input_luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_mean = float(np.mean(input_luma))
    input_median = float(np.median(input_luma))
    input_std = float(np.std(input_luma))
    p05, p95 = (float(value) for value in np.percentile(input_luma, [5, 95]))
    input_range = p95 - p05
    underexposed_fraction = float(np.mean(input_luma < 35))
    low_light = bool(input_median < 92.0 or underexposed_fraction > 0.38)

    enhancement_strength = float(
        np.clip(
            max(
                (92.0 - input_median) / 72.0,
                (underexposed_fraction - 0.25) / 0.60,
            ),
            0.0,
            1.0,
        )
    )
    gamma = 1.0
    output = image
    if low_light and enhancement_strength > 0.01:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=1.6 + 0.8 * enhancement_strength,
            tileGridSize=(8, 8),
        )
        contrast_lightness = clahe.apply(lightness)
        normalized_median = max(input_median / 255.0, 1.0 / 255.0)
        gamma = float(
            np.clip(
                np.log(112.0 / 255.0) / np.log(normalized_median),
                0.55,
                1.0,
            )
        )
        lookup = np.clip(
            255.0 * np.power(np.arange(256, dtype=np.float32) / 255.0, gamma),
            0.0,
            255.0,
        ).astype(np.uint8)
        corrected_lightness = cv2.LUT(contrast_lightness, lookup)
        enhanced = cv2.cvtColor(
            cv2.merge((corrected_lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )
        # A graded blend avoids amplifying sensor noise around the threshold.
        blend = 0.35 + 0.55 * enhancement_strength
        output = cv2.addWeighted(image, 1.0 - blend, enhanced, blend, 0.0)

    output = np.ascontiguousarray(output, dtype=np.uint8)
    output_luma = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    output_mean = float(np.mean(output_luma))
    output_std = float(np.std(output_luma))
    output_p05, output_p95 = (
        float(value) for value in np.percentile(output_luma, [5, 95])
    )
    sharpness = float(cv2.Laplacian(output_luma, cv2.CV_64F).var())
    near_blank = bool(input_mean < 6.0 and p95 < 16.0 and input_std < 4.5)
    low_contrast = bool(input_range < 24.0)
    blur_risk = bool(sharpness < 18.0)
    diagnostics: dict[str, Any] = {
        "input_luma_mean": input_mean,
        "input_luma_median": input_median,
        "input_luma_std": input_std,
        "input_dynamic_range_p90": input_range,
        "output_luma_mean": output_mean,
        "output_luma_std": output_std,
        "output_dynamic_range_p90": output_p95 - output_p05,
        "underexposed_fraction": underexposed_fraction,
        "sharpness_laplacian_variance": sharpness,
        "low_light": low_light,
        "low_contrast": low_contrast,
        "blur_risk": blur_risk,
        "near_blank": near_blank,
        "usable_for_detection": not near_blank,
        "enhancement_applied": output is not image,
        "enhancement_strength": enhancement_strength,
        "gamma": gamma,
        "input_mirrored": False,
    }
    return output, diagnostics




class InferenceEngine:
    def __init__(
        self,
        config: RuntimeConfig,
        models_directory: Path = MODELS_DIRECTORY,
        *,
        load_models: bool = True,
    ):
        self.config = config
        self.model_manager = ModelManager(
            config, models_directory, load_models=load_models
        )
        self.detector = HandDetector(models_directory / "hand_landmarker.task")
        self.resolver = GeometryResolver(config)
        self._lock = threading.Lock()

    def close(self) -> None:
        self.detector.close()

    def status(self) -> dict[str, Any]:
        return {
            "model": self.model_manager.status(),
            "mediapipe": {
                "ready": self.detector.ready,
                "error": self.detector.error,
                "running_mode": getattr(self.detector, "running_mode", "VIDEO"),
                "input_mirrored": False,
                "adaptive_low_light_preprocessing": True,
                "per_session_landmark_smoothing": True,
                "detection_policy": DETECTOR_SETTINGS,
            },
            "ready": self.model_manager.ready and self.detector.ready,
        }

    @staticmethod
    def _quality_payload(
        image: dict[str, Any],
        detector: dict[str, Any],
        landmarks: dict[str, Any],
        *,
        classification_allowed: bool,
    ) -> dict[str, Any]:
        issues: list[str] = []
        if image.get("near_blank"):
            issues.append("near_blank_frame")
        elif image.get("low_light"):
            issues.append("low_light_enhanced")
        if image.get("low_contrast"):
            issues.append("low_contrast")
        if image.get("blur_risk"):
            issues.append("blur_risk")
        if landmarks.get("jump_rejected"):
            issues.append("landmark_jump_rejected")
        if detector.get("rejection_reason"):
            issues.append(str(detector["rejection_reason"]))
        reason = landmarks.get("reason")
        if reason in {"invalid_landmarks", "implausible_landmark_geometry"}:
            issues.append(str(reason))
        if not detector.get("candidate_count", 0) and not image.get("near_blank"):
            issues.append("no_hand_candidate")
        return {
            "classification_allowed": bool(classification_allowed),
            "issues": list(dict.fromkeys(issues)),
            "image": image,
            "detector": detector,
            "landmarks": landmarks,
        }

    def _no_hand_result(
        self,
        *,
        session: RuntimeSession,
        now: float,
        total_started: float,
        actual_fps: float,
        preprocess_ms: float,
        mediapipe_ms: float,
        image_quality: dict[str, Any],
        detector_quality: dict[str, Any],
        landmark_quality: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        session.temporal_gate.reset()
        session.diagnostic_gates.clear()
        reject_label = getattr(self.config, "reject_label", "no_gesture")
        session.observe_action_release(None, tracking_gap=True)
        follow = session.follow_object.observe(
            reject_label,
            stable=False,
            confidence=0.0,
            source="no hand",
            now=now,
        )
        total_ms = (time.perf_counter() - total_started) * 1000
        return {
            "status": "no_hand",
            "message": message,
            "runtime_prediction": reject_label,
            "runtime_action": "Wait / No Action",
            "confidence": 0.0,
            "stable_frames": 0,
            "follow_object": asdict(follow),
            "actual_fps": actual_fps,
            "quality": self._quality_payload(
                image_quality,
                detector_quality,
                landmark_quality,
                classification_allowed=False,
            ),
            "timing": self._timing(
                mediapipe_ms=mediapipe_ms,
                feature_ms=0.0,
                classifier_ms=0.0,
                diagnostics_ms=0.0,
                total_ms=total_ms,
                effective_fps=actual_fps,
                preprocess_ms=preprocess_ms,
            ),
        }

    def process_frame(
        self,
        image_bytes: bytes,
        session: RuntimeSession,
        model_name: str | None = None,
        *,
        center_crop_ratio: float | None = None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        now = time.monotonic()
        actual_fps = session.record_frame(now)
        try:
            import cv2
        except ImportError as error:
            return {"status": "dependency_missing", "message": str(error)}
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"status": "invalid_image", "message": "The frame could not be decoded."}
        source_height, source_width = image.shape[:2]
        if center_crop_ratio is not None:
            ratio = min(1.0, max(0.1, float(center_crop_ratio)))
            height, width = image.shape[:2]
            side = max(1, int(round(min(width, height) * ratio)))
            left = max(0, (width - side) // 2)
            top = max(0, (height - side) // 2)
            image = image[top : top + side, left : left + side]
        # Keep native detail for distant fingers, with a bounded workload.
        if max(image.shape[:2]) > 960:
            scale = 960 / max(image.shape[:2])
            image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        preprocess_started = time.perf_counter()
        detection_image, image_quality = adaptive_low_light_preprocess(image)
        image_quality.update(source_width=source_width, source_height=source_height,
                             analysis_width=image.shape[1], analysis_height=image.shape[0])
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        rgb = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)

        if not self.model_manager.ready:
            return {
                "status": "model_missing",
                "message": "Export the trained notebook artifacts into the models directory.",
                "actual_fps": actual_fps,
                "quality": {"image": image_quality},
            }
        if not self.detector.ready:
            return {
                "status": "mediapipe_missing",
                "message": self.detector.error,
                "actual_fps": actual_fps,
                "quality": {"image": image_quality},
            }

        if not image_quality["usable_for_detection"]:
            landmark_quality = session.observe_missing_landmarks(now)
            landmark_quality["reason"] = "near_blank_frame"
            detector_quality = {
                "running_mode": getattr(self.detector, "running_mode", "VIDEO"),
                "skipped": True,
                "skip_reason": "near_blank_frame",
                "candidate_count": 0,
            }
            return self._no_hand_result(
                session=session,
                now=now,
                total_started=total_started,
                actual_fps=actual_fps,
                preprocess_ms=preprocess_ms,
                mediapipe_ms=0.0,
                image_quality=image_quality,
                detector_quality=detector_quality,
                landmark_quality=landmark_quality,
                message="The frame is too dark or blank for reliable hand detection.",
            )

        with self._lock:
            landmark_started = time.perf_counter()
            # Track the observed hand even if the gesture classifier rejects it.
            # Searching for a high classifier score on every rejected frame can
            # replace a real palm with a hallucinated hand on the background.
            def candidate_score(points):
                # Use a pose contradiction to recover drifted finger identities.
                # Low vocabulary mass alone must NOT restart a real hand track.
                try:
                    probabilities, _, _, _ = self.model_manager.predict_detailed(
                        landmarks_to_feature(points), model_name)
                    _, details = self.resolver.resolve(probabilities, points)
                    return 1.0 if details["pose_validation"]["valid"] else 0.0
                except ValueError:
                    return -0.5
            raw_landmarks = (self.detector.detect(rgb, candidate_score=candidate_score)
                             if isinstance(self.detector, HandDetector) else self.detector.detect(rgb))
            mediapipe_ms = (time.perf_counter() - landmark_started) * 1000
            detector_quality = (
                self.detector.diagnostics()
                if hasattr(self.detector, "diagnostics")
                else {"candidate_count": int(raw_landmarks is not None)}
            )
            if raw_landmarks is None:
                landmark_quality = session.observe_missing_landmarks(now)
                return self._no_hand_result(
                    session=session,
                    now=now,
                    total_started=total_started,
                    actual_fps=actual_fps,
                    preprocess_ms=preprocess_ms,
                    mediapipe_ms=mediapipe_ms,
                    image_quality=image_quality,
                    detector_quality=detector_quality,
                    landmark_quality=landmark_quality,
                    message="No usable hand was found inside the guide.",
                )
            if detector_quality.get("reacquired"):
                session.reset_landmark_tracking()
                session.temporal_gate.reset()
            landmarks, landmark_quality = session.stabilize_landmarks(
                raw_landmarks, now
            )
            if landmarks is None:
                if landmark_quality.get("reason") != "isolated_landmark_jump":
                    missing_state = session.observe_missing_landmarks(now)
                    landmark_quality["filter_reset"] = missing_state["filter_reset"]
                    landmark_quality["missing_frames"] = missing_state["missing_frames"]
                    landmark_quality["seconds_since_landmarks"] = missing_state[
                        "seconds_since_landmarks"
                    ]
                return self._no_hand_result(
                    session=session,
                    now=now,
                    total_started=total_started,
                    actual_fps=actual_fps,
                    preprocess_ms=preprocess_ms,
                    mediapipe_ms=mediapipe_ms,
                    image_quality=image_quality,
                    detector_quality=detector_quality,
                    landmark_quality=landmark_quality,
                    message="Hand landmarks were rejected as unstable or incomplete.",
                )
            feature_started = time.perf_counter()
            feature = landmarks_to_feature(landmarks)
            feature_ms = (time.perf_counter() - feature_started) * 1000
            follow_was_active = session.follow_object.active
            selected_name = self.model_manager.resolve_name(model_name)
            diagnostic_names = [
                name for name in ("ONNX", selected_name)
                if name in self.model_manager.models
            ]
            model_rows = self.model_manager.predict_many(feature, diagnostic_names)
            resolved_rows: dict[str, np.ndarray] = {}
            resolver_rows: dict[str, dict[str, dict | None]] = {}
            for name, (probabilities, _, _) in model_rows.items():
                resolved_rows[name], resolver_rows[name] = self.resolver.resolve(
                    probabilities,
                    landmarks,
                    handedness=detector_quality.get("selected_handedness"),
                    handedness_confidence=detector_quality.get(
                        "selected_handedness_confidence"
                    ),
                )

        raw_probabilities, classifier_ms, selected_quality = model_rows[selected_name]
        resolved_probabilities = resolved_rows[selected_name]
        resolver_details = resolver_rows[selected_name]
        pose_validation = resolver_details.get("pose_validation") or {}
        direction_details = resolver_details.get("directional") or {}
        negative_support = float(
            (selected_quality.get("online_adapter") or {}).get("negative_support", 0)
        )
        geometry_supported = bool(pose_validation.get("geometry_supported", False)) and negative_support <= 0
        directional_recovery = bool(
            geometry_supported
            and direction_details.get("strong_geometry", False)
            and direction_details.get("model_supported_pose", False)
        )
        if not pose_validation.get("valid", True) and hasattr(self.detector, "request_pose_recovery"):
            self.detector.request_pose_recovery()
        decision = session.temporal_gate.update(
            resolved_probabilities,
            now=now,
            known_gesture_mass=float(
                selected_quality.get("known_gesture_mass", 1.0)
            ),
            pose_valid=bool(pose_validation.get("valid", True)),
            rejection_reason=str(
                pose_validation.get("reason", "pose geometry rejected")
            ),
            geometry_supported=geometry_supported,
            directional_recovery=directional_recovery,
        )

        diagnostic_results: dict[str, dict[str, Any]] = {}
        for name, (probabilities, elapsed_ms, model_quality) in model_rows.items():
            model_pose = resolver_rows[name].get("pose_validation") or {}
            model_direction = resolver_rows[name].get("directional") or {}
            model_negative_support = float(
                (model_quality.get("online_adapter") or {}).get("negative_support", 0)
            )
            model_geometry_supported = bool(model_pose.get("geometry_supported", False)) and model_negative_support <= 0
            model_decision = (
                decision if name == selected_name
                else session.diagnostic_gate(name).update(
                    resolved_rows[name],
                    now=now,
                    known_gesture_mass=float(
                        model_quality.get("known_gesture_mass", 1.0)
                    ),
                    pose_valid=bool(model_pose.get("valid", True)),
                    rejection_reason=str(
                        model_pose.get("reason", "pose geometry rejected")
                    ),
                    geometry_supported=model_geometry_supported,
                    directional_recovery=bool(
                        model_geometry_supported
                        and model_direction.get("strong_geometry", False)
                        and model_direction.get("model_supported_pose", False)
                    ),
                )
            )
            diagnostic_results[name] = {
                "used_for_runtime": name == selected_name,
                "raw_prediction": self.config.class_names[int(np.argmax(probabilities))],
                "prediction": model_decision.predicted_gesture,
                "confidence": model_decision.confidence,
                "stable_frames": model_decision.stable_frames,
                "execute": model_decision.execute,
                "reason": model_decision.reason,
                "classifier_ms": elapsed_ms,
                "known_gesture_mass": model_decision.known_gesture_mass,
                "probability_margin": model_decision.probability_margin,
                "online_adapter_applied": bool(
                    model_quality.get("online_adapter_applied", False)
                ),
            }

        follow = session.follow_object.observe(
            decision.predicted_gesture,
            stable=bool(
                decision.execute
                and decision.confidence >= self.config.follow_min_confidence
            ),
            confidence=decision.confidence,
            source=selected_name,
            now=now,
        )
        mapped_action = self.config.gesture_to_action.get(
            decision.predicted_gesture, "No Gesture"
        )
        runtime_action = "Wait / No Action"
        action_reason = decision.reason
        if follow.completed:
            runtime_action = "Follow Object"
            action_reason = "Open Palm enabled object tracking"
        elif follow_was_active:
            action_reason = follow.message
        elif decision.execute:
            new_action_edge = decision.predicted_gesture != session.last_action_gesture
            cooldown_ready = now - session.last_action_at >= self.config.action_cooldown_seconds
            if new_action_edge and cooldown_ready:
                runtime_action = mapped_action
                action_reason = "stable command emitted"
                session.last_action_gesture = decision.predicted_gesture
                session.last_action_at = now
                session._action_release_frames = 0
            else:
                action_reason = "duplicate command suppressed"
                session.observe_action_release(decision.predicted_gesture)
        else:
            current_gesture = (
                None
                if decision.predicted_gesture == self.config.reject_label
                else decision.predicted_gesture
            )
            session.observe_action_release(current_gesture)

        total_ms = (time.perf_counter() - total_started) * 1000
        diagnostics_ms = float(sum(row[1] for row in model_rows.values()))
        base_probability_values = np.asarray(
            selected_quality.get("base_probabilities", raw_probabilities),
            dtype=np.float64,
        ).reshape(-1)
        return {
            "status": "predicted",
            "model": selected_name,
            "raw_prediction": self.config.class_names[int(np.argmax(raw_probabilities))],
            "runtime_prediction": decision.predicted_gesture,
            "mapped_action": mapped_action,
            "runtime_action": runtime_action,
            "action_reason": action_reason,
            "confidence": decision.confidence,
            "stable_frames": decision.stable_frames,
            "frames_used": decision.frames_used,
            "held_seconds": decision.held_seconds,
            "known_gesture_mass": decision.known_gesture_mass,
            "probability_margin": decision.probability_margin,
            "probabilities": decision.probabilities,
            "raw_probabilities": {
                name: float(raw_probabilities[index])
                for index, name in enumerate(self.config.class_names)
            },
            "base_probabilities": {
                name: float(base_probability_values[index])
                for index, name in enumerate(self.config.class_names)
            },
            "online_adapter_applied": bool(
                selected_quality.get("online_adapter_applied", False)
            ),
            "follow_object": asdict(follow),
            "diagnostics": diagnostic_results,
            "resolvers": resolver_details,
            "landmarks": landmarks.round(6).tolist(),
            "display_landmarks": (landmarks.round(6).tolist()
                                  if decision.predicted_gesture != self.config.reject_label else []),
            "feature_vector": feature.round(7).tolist(),
            "actual_fps": actual_fps,
            "quality": self._quality_payload(
                image_quality,
                detector_quality,
                landmark_quality,
                classification_allowed=True,
            ),
            "timing": self._timing(
                mediapipe_ms=mediapipe_ms,
                feature_ms=feature_ms,
                classifier_ms=classifier_ms,
                diagnostics_ms=diagnostics_ms,
                total_ms=total_ms,
                effective_fps=actual_fps,
                preprocess_ms=preprocess_ms,
            ),
        }

    def _timing(
        self,
        *,
        mediapipe_ms: float,
        feature_ms: float,
        classifier_ms: float,
        diagnostics_ms: float,
        total_ms: float,
        effective_fps: float,
        preprocess_ms: float = 0.0,
    ) -> dict[str, float | bool | str]:
        frame_budget = self.config.frame_budget_ms
        uncapped = 1000.0 / total_ms if total_ms > 0 else 0.0
        budget_used = 100.0 * total_ms / frame_budget
        capacity_pass = total_ms <= frame_budget
        return {
            "preprocess_ms": float(preprocess_ms),
            "mediapipe_ms": float(mediapipe_ms),
            "feature_ms": float(feature_ms),
            "classifier_ms": float(classifier_ms),
            "diagnostics_ms": float(diagnostics_ms),
            "total_ms": float(total_ms),
            "uncapped_fps": float(uncapped),
            "configured_fps": float(self.config.target_fps),
            "effective_application_fps": float(
                min(self.config.target_fps, max(0.0, effective_fps))
            ),
            "frame_budget_ms": float(frame_budget),
            "budget_used_percent": float(budget_used),
            "headroom_ms": float(frame_budget - total_ms),
            "target_fps_capacity_pass": bool(capacity_pass),
            # Kept for older dashboard clients; this now means the configured
            # target FPS budget (10 FPS in the backend-safe release).
            "ten_fps_capacity_pass": bool(capacity_pass),
            "verdict": "PASS" if capacity_pass else "FAIL",
        }
