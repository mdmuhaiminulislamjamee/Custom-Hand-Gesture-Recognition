from __future__ import annotations

"""Small, guarded online-learning layer for the immutable ONNX classifier.

The ONNX graph remains the source of the *base* probabilities.  This module
stores reviewed feature vectors as local calibration prototypes and blends a
bounded amount of their label evidence into the base distribution.  A proposed
prototype is committed only when it improves the reviewed sample and stays
inside the configured validation tolerances.

Only NumPy and the Python standard library are used.  In particular, persisted
files are non-pickled ``.npz`` archives, so loading this state never executes
Python objects.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import tempfile
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import CLASS_NAMES


FEATURE_COUNT = 76
CLASS_COUNT = len(CLASS_NAMES)
SCHEMA_VERSION = 2
DIRECTION_SEMANTIC_VERSION = "ncm_unmirrored_horizontal_swap_v1"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODELS_DIRECTORY = _PROJECT_ROOT / "models"


class _IncompatibleAdapterState(ValueError):
    """A valid older state that must not be applied to this model contract."""


@dataclass(slots=True)
class _AdapterState:
    feature_center: np.ndarray
    feature_scale: np.ndarray
    prototypes: np.ndarray
    labels: np.ndarray
    base_probabilities: np.ndarray
    influences: np.ndarray
    attempted_updates: int
    accepted_updates: int
    rejected_updates: int
    rollback_count: int
    created_utc: str
    last_updated_utc: str | None
    last_validation: dict[str, Any] | None

    def clone(self) -> "_AdapterState":
        return _AdapterState(
            feature_center=self.feature_center.copy(),
            feature_scale=self.feature_scale.copy(),
            prototypes=self.prototypes.copy(),
            labels=self.labels.copy(),
            base_probabilities=self.base_probabilities.copy(),
            influences=self.influences.copy(),
            attempted_updates=self.attempted_updates,
            accepted_updates=self.accepted_updates,
            rejected_updates=self.rejected_updates,
            rollback_count=self.rollback_count,
            created_utc=self.created_utc,
            last_updated_utc=self.last_updated_utc,
            last_validation=(
                None
                if self.last_validation is None
                else json.loads(json.dumps(self.last_validation))
            ),
        )


@dataclass(frozen=True, slots=True)
class _Cache:
    features: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray | None
    source: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _soft_normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Probabilities must be finite and non-negative.")
    totals = values.sum(axis=-1, keepdims=True)
    if (totals <= 1e-12).any():
        raise ValueError("Each probability row must have positive mass.")
    return values / totals


def _metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    acceptance_scale: np.ndarray | None = None,
    acceptance_floor: float = 0.80,
) -> dict[str, Any]:
    rows = _soft_normalize(probabilities)
    truth = np.asarray(labels, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[0] != len(truth):
        raise ValueError("Validation probabilities and labels are not aligned.")
    predicted = rows.argmax(axis=1)
    scales = (
        np.ones(len(truth), dtype=np.float64)
        if acceptance_scale is None
        else np.asarray(acceptance_scale, dtype=np.float64).reshape(-1)
    )
    if scales.shape != (len(truth),) or not np.isfinite(scales).all():
        raise ValueError("Validation acceptance scales are not aligned and finite.")
    scales = np.clip(scales, 0.0, 1.0)
    accepted = scales >= float(acceptance_floor)
    predicted = np.where(accepted, predicted, -1)
    known = truth >= 0
    unknown = ~known
    per_class: list[float] = []
    present: list[bool] = []
    for class_index in range(class_count):
        actual = known & (truth == class_index)
        guessed = predicted == class_index
        true_positive = int(np.count_nonzero(actual & guessed))
        false_positive = int(np.count_nonzero(~actual & guessed))
        false_negative = int(np.count_nonzero(actual & ~guessed))
        denominator = 2 * true_positive + false_positive + false_negative
        per_class.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
        present.append(bool(np.any(actual)))
    present_values = [score for score, exists in zip(per_class, present) if exists]
    if np.any(known):
        known_indexes = np.flatnonzero(known)
        # A local rejection is a confidence loss for a known validation row.
        correct_mass = rows[known_indexes, truth[known_indexes]] * scales[known_indexes]
        clipped = np.clip(correct_mass, 1e-12, 1.0)
        accuracy = float(np.mean(predicted[known] == truth[known]))
        log_loss = float(-np.mean(np.log(clipped)))
    else:
        accuracy = 0.0
        log_loss = 0.0
    return {
        "rows": int(len(truth)),
        "known_rows": int(np.count_nonzero(known)),
        "unknown_rows": int(np.count_nonzero(unknown)),
        "covered_classes": int(sum(present)),
        "accuracy": accuracy,
        "macro_f1": float(np.mean(present_values)) if present_values else 0.0,
        "per_class_f1": per_class,
        "class_present": present,
        "log_loss": log_loss,
        "unknown_false_acceptance_rate": (
            float(np.mean(accepted[unknown])) if np.any(unknown) else 0.0
        ),
        "overall_rejection_accuracy": float(
            np.mean(np.where(known, predicted == truth, ~accepted))
        ),
    }


class OnlineLearningAdapter:
    """Validation-gated, persisted local probability adapter.

    ``config`` may be a ``RuntimeConfig``-like object (with ``class_names`` and
    ``feature_names``), or it may be the configured ten-class sequence itself.
    ``model_manager`` is optional and is used only to obtain raw ONNX
    probabilities for caches that contain ``X``/``y`` but no probability rows.

    The adapter is intentionally conservative:

    * only an explicit, reviewed 76-D sample can create a candidate;
    * force/unsafe updates are not supported;
    * the correction has local RBF support and a hard blend ceiling;
    * cached holdout metrics can veto the candidate; and
    * every accepted state has an atomic pre-update backup.
    """

    def __init__(
        self,
        config: Any,
        model_manager: Any | None = None,
        models_directory: Path | str = _DEFAULT_MODELS_DIRECTORY,
        *,
        state_path: Path | str | None = None,
        replay_path: Path | str | None = None,
        validation_path: Path | str | None = None,
        backup_directory: Path | str | None = None,
        base_predictor: Callable[[np.ndarray], np.ndarray] | None = None,
        require_holdout_for_safe: bool = False,
        adaptation_strength: float = 0.55,
        kernel_bandwidth: float = 0.35,
        minimum_similarity: float = 0.02,
        minimum_probability_gain: float = 0.002,
        minimum_rejection_gain: float = 0.05,
        rejection_strength: float = 0.90,
        adapter_acceptance_floor: float = 0.80,
        allowed_macro_f1_drop: float = 0.003,
        allowed_per_class_f1_drop: float = 0.02,
        allowed_accuracy_drop: float = 0.0,
        allowed_log_loss_increase: float = 0.005,
        allowed_unknown_probability_shift: float = 0.05,
        max_examples_per_class: int = 64,
        max_backups: int = 50,
        validation_rows_per_class: int = 128,
    ):
        if hasattr(config, "class_names"):
            class_names = tuple(str(value) for value in config.class_names)
            feature_names = tuple(str(value) for value in config.feature_names)
            reject_label = str(getattr(config, "reject_label", "no_gesture"))
        else:
            class_names = tuple(str(value) for value in config)
            feature_names = tuple(f"feature_{index}" for index in range(FEATURE_COUNT))
            reject_label = "no_gesture"
        if tuple(class_names) != tuple(CLASS_NAMES):
            raise ValueError(
                "Online learning requires the exact configured ten-command class order."
            )
        if len(feature_names) != FEATURE_COUNT or len(set(feature_names)) != FEATURE_COUNT:
            raise ValueError("Online learning requires the exact unique 76-D feature contract.")
        if not 0.0 < float(adaptation_strength) < 1.0:
            raise ValueError("adaptation_strength must be between zero and one.")
        if float(kernel_bandwidth) <= 0.0:
            raise ValueError("kernel_bandwidth must be positive.")
        if not 0.0 <= float(minimum_similarity) < 1.0:
            raise ValueError("minimum_similarity must be in [0, 1).")
        if reject_label in class_names:
            raise ValueError("The reject label must not be one of the ten output classes.")
        if not 0.0 < float(rejection_strength) <= 1.0:
            raise ValueError("rejection_strength must be in (0, 1].")
        if not 0.0 < float(adapter_acceptance_floor) < 1.0:
            raise ValueError("adapter_acceptance_floor must be between zero and one.")
        if int(max_examples_per_class) < 1 or int(max_backups) < 1:
            raise ValueError("Retention limits must be positive.")
        if int(validation_rows_per_class) < 1:
            raise ValueError("validation_rows_per_class must be positive.")
        bounded_gains = {
            "minimum_probability_gain": minimum_probability_gain,
            "minimum_rejection_gain": minimum_rejection_gain,
        }
        for name, value in bounded_gains.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        tolerances = {
            "allowed_macro_f1_drop": allowed_macro_f1_drop,
            "allowed_per_class_f1_drop": allowed_per_class_f1_drop,
            "allowed_accuracy_drop": allowed_accuracy_drop,
            "allowed_log_loss_increase": allowed_log_loss_increase,
            "allowed_unknown_probability_shift": allowed_unknown_probability_shift,
        }
        for name, value in tolerances.items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

        directory = Path(models_directory)
        self.class_names = class_names
        self.feature_names = feature_names
        self.reject_label = reject_label
        self.direction_semantic_version = DIRECTION_SEMANTIC_VERSION
        bundle = getattr(model_manager, "bundle_metadata", {})
        self.base_model_sha256 = (
            str(bundle.get("model_sha256"))
            if isinstance(bundle, Mapping) and bundle.get("model_sha256")
            else None
        )
        self.class_to_idx = {name: index for index, name in enumerate(class_names)}
        self.state_path = Path(state_path) if state_path is not None else directory / "gesture_online_adapter_state.npz"
        self.replay_path = Path(replay_path) if replay_path is not None else directory / "gesture_online_replay_cache.npz"
        self.validation_path = (
            Path(validation_path)
            if validation_path is not None
            else directory / "gesture_online_validation_cache.npz"
        )
        self.backup_directory = (
            Path(backup_directory)
            if backup_directory is not None
            else directory / "online_adapter_backups"
        )
        self.require_holdout_for_safe = bool(require_holdout_for_safe)
        self.adaptation_strength = float(adaptation_strength)
        self.kernel_bandwidth = float(kernel_bandwidth)
        self.minimum_similarity = float(minimum_similarity)
        self.minimum_probability_gain = float(minimum_probability_gain)
        self.minimum_rejection_gain = float(minimum_rejection_gain)
        self.rejection_strength = float(rejection_strength)
        self.adapter_acceptance_floor = float(adapter_acceptance_floor)
        self.allowed_macro_f1_drop = float(allowed_macro_f1_drop)
        self.allowed_per_class_f1_drop = float(allowed_per_class_f1_drop)
        self.allowed_accuracy_drop = float(allowed_accuracy_drop)
        self.allowed_log_loss_increase = float(allowed_log_loss_increase)
        self.allowed_unknown_probability_shift = float(allowed_unknown_probability_shift)
        self.max_examples_per_class = int(max_examples_per_class)
        self.max_backups = int(max_backups)
        self.validation_rows_per_class = int(validation_rows_per_class)
        self._lock = threading.RLock()
        self._base_predictor = base_predictor or self._predictor_from_manager(model_manager)
        self._load_error: str | None = None
        self._state_notice: str | None = None
        self._quarantined_state_path: Path | None = None
        self._state = self._new_state()
        if self.state_path.is_file():
            try:
                self._state = self._load_state(self.state_path)
            except _IncompatibleAdapterState as error:
                try:
                    self._quarantined_state_path = self._quarantine_incompatible_state()
                except OSError as quarantine_error:
                    self._load_error = (
                        f"Online adapter state is incompatible ({error}) and could not "
                        f"be preserved: {quarantine_error}"
                    )
                else:
                    self._state_notice = (
                        f"Incompatible online adapter state was preserved at "
                        f"{self._quarantined_state_path}; Safe Learn started fresh."
                    )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                # Runtime inference stays available with unmodified ONNX output,
                # but learning is fail-closed until a valid state is restored.
                self._load_error = f"Online adapter state is invalid: {error}"

    @staticmethod
    def _predictor_from_manager(model_manager: Any | None) -> Callable[[np.ndarray], np.ndarray] | None:
        if model_manager is None:
            return None
        models = getattr(model_manager, "models", {})
        classifier = models.get("ONNX") if isinstance(models, Mapping) else None
        predictor = getattr(classifier, "predict_proba", None)
        return predictor if callable(predictor) else None

    def _quarantine_incompatible_state(self) -> Path:
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        token = secrets.token_hex(3)
        destination = self.backup_directory / (
            f"incompatible_{self.state_path.stem}_{timestamp}_{token}.npz"
        )
        os.replace(self.state_path, destination)
        return destination

    def _new_state(self) -> _AdapterState:
        center, scale = self._normalization_from_available_caches()
        return _AdapterState(
            feature_center=center,
            feature_scale=scale,
            prototypes=np.empty((0, FEATURE_COUNT), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
            base_probabilities=np.empty((0, CLASS_COUNT), dtype=np.float64),
            influences=np.empty((0,), dtype=np.float64),
            attempted_updates=0,
            accepted_updates=0,
            rejected_updates=0,
            rollback_count=0,
            created_utc=_utc_now(),
            last_updated_utc=None,
            last_validation=None,
        )

    def _normalization_from_available_caches(self) -> tuple[np.ndarray, np.ndarray]:
        references: list[np.ndarray] = []
        for path in (getattr(self, "replay_path", None), getattr(self, "validation_path", None)):
            if path is None or not Path(path).is_file():
                continue
            try:
                with np.load(path, allow_pickle=False) as archive:
                    key = self._feature_key(archive.files)
                    rows = np.asarray(archive[key], dtype=np.float64)
                if rows.ndim == 2 and rows.shape[1] == FEATURE_COUNT and np.isfinite(rows).all():
                    references.append(rows)
            except (OSError, ValueError, KeyError):
                continue
        if not references:
            return (
                np.zeros(FEATURE_COUNT, dtype=np.float64),
                np.full(FEATURE_COUNT, 0.25, dtype=np.float64),
            )
        rows = np.vstack(references)
        center = np.median(rows, axis=0)
        lower, upper = np.quantile(rows, [0.25, 0.75], axis=0)
        robust_scale = (upper - lower) / 1.349
        standard_scale = np.std(rows, axis=0)
        scale = np.maximum.reduce(
            [robust_scale, 0.5 * standard_scale, np.full(FEATURE_COUNT, 0.03)]
        )
        return center.astype(np.float64), scale.astype(np.float64)

    @staticmethod
    def _feature_key(files: Sequence[str]) -> str:
        for key in ("X", "x", "features", "feature_vectors"):
            if key in files:
                return key
        raise KeyError("Cache does not contain X/features rows.")

    @staticmethod
    def _label_key(files: Sequence[str]) -> str:
        for key in ("y", "labels", "class_indices"):
            if key in files:
                return key
        raise KeyError("Cache does not contain y/labels rows.")

    @staticmethod
    def _probability_key(files: Sequence[str]) -> str | None:
        for key in ("base_probabilities", "probabilities", "proba", "P"):
            if key in files:
                return key
        return None

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, (bytes, np.bytes_)):
            return bytes(value).decode("utf-8")
        return str(value)

    def _coerce_public_probabilities(self, probabilities: Any) -> tuple[np.ndarray, bool]:
        if isinstance(probabilities, Mapping):
            if set(probabilities) != set(self.class_names):
                raise ValueError("Probability mapping must contain exactly the configured classes.")
            values = np.asarray([probabilities[name] for name in self.class_names], dtype=np.float64)
        else:
            values = np.asarray(probabilities, dtype=np.float64)
        was_vector = values.ndim == 1
        if was_vector:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != CLASS_COUNT:
            raise ValueError("Expected one ten-class probability row per feature row.")
        return _soft_normalize(values), was_vector

    @staticmethod
    def _coerce_features(features: Any) -> tuple[np.ndarray, bool]:
        values = np.asarray(features, dtype=np.float64)
        was_vector = values.ndim == 1
        if was_vector:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
            raise ValueError("A finite 76-D feature vector is required.")
        if not np.isfinite(values).all():
            raise ValueError("Feature vectors must contain only finite values.")
        return values, was_vector

    def _apply_state_with_quality(
        self,
        probabilities: np.ndarray,
        features: np.ndarray,
        state: _AdapterState,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        base = _soft_normalize(probabilities)
        if len(state.prototypes) == 0:
            ones = np.ones(len(features), dtype=np.float64)
            zeros = np.zeros(len(features), dtype=np.float64)
            return base.copy(), ones, {
                "positive_support": zeros,
                "negative_support": zeros,
                "rejection_score": zeros,
                "nearest_similarity": zeros,
            }
        differences = (
            features[:, None, :] - state.prototypes.astype(np.float64)[None, :, :]
        ) / state.feature_scale[None, None, :]
        squared_distance = np.mean(np.square(differences), axis=2)
        similarities = np.exp(
            -0.5 * squared_distance / (self.kernel_bandwidth * self.kernel_bandwidth)
        )
        similarities[similarities < self.minimum_similarity] = 0.0
        evidence_weights = similarities * state.influences[None, :]
        positive_mask = state.labels >= 0
        negative_mask = ~positive_mask
        positive_total = (
            evidence_weights[:, positive_mask].sum(axis=1)
            if np.any(positive_mask)
            else np.zeros(len(features), dtype=np.float64)
        )
        negative_total = (
            evidence_weights[:, negative_mask].sum(axis=1)
            if np.any(negative_mask)
            else np.zeros(len(features), dtype=np.float64)
        )
        evidence_total = positive_total + negative_total
        label_evidence = np.zeros((len(features), CLASS_COUNT), dtype=np.float64)
        for class_index in range(CLASS_COUNT):
            mask = state.labels == class_index
            if np.any(mask):
                label_evidence[:, class_index] = evidence_weights[:, mask].sum(axis=1)
        positive_supported = positive_total > 1e-12
        label_evidence[positive_supported] /= positive_total[positive_supported, None]
        # Nearby negative feedback attenuates positive corrections. This avoids a
        # contradictory reviewed pair turning into a confident command.
        positive_fraction = positive_total / np.maximum(evidence_total, 1e-12)
        blend = (
            self.adaptation_strength
            * np.minimum(1.0, positive_total)
            * positive_fraction
        )
        adjusted = (1.0 - blend[:, None]) * base + blend[:, None] * label_evidence
        adjusted[~positive_supported] = base[~positive_supported]

        negative_fraction = negative_total / np.maximum(evidence_total, 1e-12)
        negative_support = np.minimum(1.0, negative_total) * negative_fraction
        rejection_score = np.clip(
            self.rejection_strength * negative_support, 0.0, 1.0
        )
        acceptance_scale = 1.0 - rejection_score
        return _soft_normalize(adjusted), acceptance_scale, {
            "positive_support": np.minimum(1.0, positive_total),
            "negative_support": negative_support,
            "rejection_score": rejection_score,
            "nearest_similarity": similarities.max(axis=1),
        }

    def _apply_state(
        self,
        probabilities: np.ndarray,
        features: np.ndarray,
        state: _AdapterState,
    ) -> np.ndarray:
        return self._apply_state_with_quality(probabilities, features, state)[0]

    def apply(self, probabilities: Any, feature: Any) -> np.ndarray:
        """Apply the live adapter to raw ONNX probabilities.

        Both inputs may be a single row or matching batches.  A single row
        returns shape ``(10,)``; batches return ``(N, 10)``.  If persisted state
        failed validation, this method deliberately returns normalized base
        probabilities so camera inference remains available.
        """

        rows, probability_was_vector = self._coerce_public_probabilities(probabilities)
        features, feature_was_vector = self._coerce_features(feature)
        if probability_was_vector != feature_was_vector or len(rows) != len(features):
            raise ValueError("Probabilities and features must have matching row counts.")
        with self._lock:
            result = (
                rows.copy()
                if self._load_error is not None
                else self._apply_state(rows, features, self._state)
            )
        return result[0] if probability_was_vector else result

    def apply_with_quality(
        self,
        probabilities: Any,
        feature: Any,
        *,
        known_gesture_mass: float | Sequence[float] | np.ndarray = 1.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply command calibration and return local rejection evidence.

        ``quality["known_gesture_mass"]`` is the supplied ONNX known-gesture
        mass multiplied by the adapter's acceptance scale.  The runtime should
        pass this adjusted mass to its existing open-set/temporal gate.  This is
        how a reviewed ``no_gesture`` sample suppresses a false command without
        exposing it as an eleventh public command output.
        """

        rows, probability_was_vector = self._coerce_public_probabilities(probabilities)
        features, feature_was_vector = self._coerce_features(feature)
        if probability_was_vector != feature_was_vector or len(rows) != len(features):
            raise ValueError("Probabilities and features must have matching row counts.")
        masses = np.asarray(known_gesture_mass, dtype=np.float64)
        if masses.ndim == 0:
            masses = np.full(len(rows), float(masses), dtype=np.float64)
        else:
            masses = masses.reshape(-1)
        if masses.shape != (len(rows),) or not np.isfinite(masses).all():
            raise ValueError("known_gesture_mass must be one finite value per row.")
        if (masses < 0.0).any() or (masses > 1.0).any():
            raise ValueError("known_gesture_mass must be in [0, 1].")
        with self._lock:
            if self._load_error is None:
                result, acceptance, details = self._apply_state_with_quality(
                    rows, features, self._state
                )
            else:
                result = rows.copy()
                acceptance = np.ones(len(rows), dtype=np.float64)
                zeros = np.zeros(len(rows), dtype=np.float64)
                details = {
                    "positive_support": zeros,
                    "negative_support": zeros,
                    "rejection_score": zeros,
                    "nearest_similarity": zeros,
                }
        adjusted_mass = masses * acceptance
        quality: dict[str, Any] = {
            **details,
            "acceptance_scale": acceptance,
            "base_known_gesture_mass": masses,
            "known_gesture_mass": adjusted_mass,
            "state_error": self._load_error,
        }
        if probability_was_vector:
            quality = {
                key: (float(value[0]) if isinstance(value, np.ndarray) else value)
                for key, value in quality.items()
            }
            return result[0], quality
        return result, quality

    def _predict_base(self, features: np.ndarray) -> np.ndarray:
        if self._base_predictor is None:
            raise ValueError(
                "Raw base probabilities are absent; provide them in the NPZ cache "
                "or configure base_predictor/the ONNX model manager."
            )
        predicted = np.asarray(self._base_predictor(features.astype(np.float32)), dtype=np.float64)
        if predicted.shape != (len(features), CLASS_COUNT):
            raise ValueError(
                f"Base predictor returned {predicted.shape}; expected "
                f"({len(features)}, {CLASS_COUNT})."
            )
        return _soft_normalize(predicted)

    def _load_cache(self, path: Path, *, need_probabilities: bool) -> _Cache:
        with np.load(path, allow_pickle=False) as archive:
            files = archive.files
            features = np.asarray(archive[self._feature_key(files)], dtype=np.float64)
            raw_labels = np.asarray(archive[self._label_key(files)])
            probability_key = self._probability_key(files)
            probabilities = (
                None
                if probability_key is None
                else np.asarray(archive[probability_key], dtype=np.float64)
            )
            source_names = (
                tuple(self._text(value) for value in np.asarray(archive["class_names"]).reshape(-1))
                if "class_names" in files
                else None
            )

        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
            raise ValueError(f"{path.name} does not contain exact 76-D feature rows.")
        if not np.isfinite(features).all():
            raise ValueError(f"{path.name} contains non-finite feature values.")
        raw_labels = raw_labels.reshape(-1)
        if len(raw_labels) != len(features):
            raise ValueError(f"{path.name} features and labels are not aligned.")

        keep = np.ones(len(features), dtype=bool)
        if raw_labels.dtype.kind in {"U", "S"}:
            label_names = np.asarray([self._text(value) for value in raw_labels])
            allowed_names = (*self.class_names, self.reject_label)
            keep = np.isin(label_names, allowed_names)
            labels = np.asarray(
                [
                    -1 if value == self.reject_label else self.class_to_idx.get(value, -2)
                    for value in label_names
                ],
                dtype=np.int64,
            )
        else:
            try:
                numeric_labels = raw_labels.astype(np.int64)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path.name} labels must be integer indices or strings.") from error
            if not np.array_equal(raw_labels, numeric_labels):
                raise ValueError(f"{path.name} contains non-integral numeric labels.")
            if source_names is None:
                labels = numeric_labels
            else:
                if (numeric_labels < -1).any() or (numeric_labels >= len(source_names)).any():
                    raise ValueError(f"{path.name} contains out-of-range source labels.")
                label_names = np.asarray(
                    [
                        self.reject_label if index == -1 else source_names[index]
                        for index in numeric_labels
                    ]
                )
                allowed_names = (*self.class_names, self.reject_label)
                keep = np.isin(label_names, allowed_names)
                labels = np.asarray(
                    [
                        -1 if value == self.reject_label else self.class_to_idx.get(value, -2)
                        for value in label_names
                    ],
                    dtype=np.int64,
                )

        features = features[keep]
        labels = labels[keep]
        if not len(features):
            raise ValueError(f"{path.name} has no rows for the configured feedback labels.")
        if (labels < -1).any() or (labels >= CLASS_COUNT).any():
            raise ValueError(
                f"{path.name} labels are outside -1 plus the configured ten-class order; "
                "include class_names when reducing an older cache."
            )

        if probabilities is not None:
            if probabilities.ndim != 2 or probabilities.shape[0] != len(keep):
                raise ValueError(f"{path.name} probability rows are not aligned.")
            probabilities = probabilities[keep]
            if probabilities.shape[1] != CLASS_COUNT:
                if source_names is None or probabilities.shape[1] != len(source_names):
                    raise ValueError(f"{path.name} probabilities do not follow the ten-class order.")
                try:
                    columns = [source_names.index(name) for name in self.class_names]
                except ValueError as error:
                    raise ValueError(f"{path.name} is missing a configured probability class.") from error
                probabilities = probabilities[:, columns]
            probabilities = _soft_normalize(probabilities)
        elif need_probabilities:
            probabilities = self._predict_base(features)
        return _Cache(
            features=features.astype(np.float64),
            labels=labels.astype(np.int64),
            probabilities=probabilities,
            source=path.name,
        )

    def _balanced_rows(self, cache: _Cache, per_class: int) -> _Cache:
        selected: list[int] = []
        for class_index in range(-1, CLASS_COUNT):
            indices = np.flatnonzero(cache.labels == class_index)
            selected.extend(indices[:per_class].tolist())
        chosen = np.asarray(selected, dtype=np.int64)
        return _Cache(
            features=cache.features[chosen],
            labels=cache.labels[chosen],
            probabilities=(
                None if cache.probabilities is None else cache.probabilities[chosen]
            ),
            source=cache.source,
        )

    def _validation_rows(
        self,
        feature: np.ndarray,
        label: int,
        base_probability: np.ndarray,
        state: _AdapterState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, list[str]]:
        static_parts: list[_Cache] = []
        warnings: list[str] = []
        for name, path, per_class in (
            ("validation", self.validation_path, self.validation_rows_per_class),
            ("replay", self.replay_path, max(1, self.validation_rows_per_class // 4)),
        ):
            if not path.is_file():
                continue
            try:
                cache = self._balanced_rows(
                    self._load_cache(path, need_probabilities=True), per_class
                )
                static_parts.append(cache)
            except (OSError, ValueError, KeyError) as error:
                warnings.append(f"{name} cache skipped: {error}")

        if self.require_holdout_for_safe:
            if not static_parts:
                detail = "; ".join(warnings) if warnings else "no cache file is present"
                raise RuntimeError(f"Safe learning requires a usable holdout cache: {detail}.")
            static_labels = np.concatenate([part.labels for part in static_parts])
            covered = set(static_labels[static_labels >= 0].tolist())
            if covered != set(range(CLASS_COUNT)) or not np.any(static_labels == -1):
                raise RuntimeError(
                    "Safe learning requires holdout coverage for all ten classes "
                    "and the -1 reject rows."
                )

        parts = list(static_parts)
        # Previously accepted reviewed examples are useful regression anchors,
        # especially when deployment-domain samples are not in the static cache.
        if len(state.prototypes):
            parts.append(
                _Cache(
                    state.prototypes.astype(np.float64),
                    state.labels.copy(),
                    state.base_probabilities.copy(),
                    "reviewed_history",
                )
            )
        if parts:
            features = np.vstack([part.features for part in parts])
            labels = np.concatenate([part.labels for part in parts])
            probabilities = np.vstack(
                [part.probabilities for part in parts if part.probabilities is not None]
            )
            source = "+".join(part.source for part in parts)
            return features, labels, probabilities, source, warnings

        warnings.append(
            "No usable holdout cache; safety is limited to the reviewed sample."
        )
        return (
            feature.reshape(1, -1),
            np.asarray([label], dtype=np.int64),
            base_probability.reshape(1, -1),
            "reviewed_sample_only",
            warnings,
        )

    def _candidate_with_example(
        self,
        state: _AdapterState,
        feature: np.ndarray,
        label: int,
        base_probability: np.ndarray,
        influence: float,
    ) -> _AdapterState:
        candidate = state.clone()
        candidate.prototypes = np.vstack([candidate.prototypes, feature]).astype(np.float32)
        candidate.labels = np.concatenate([candidate.labels, np.asarray([label], dtype=np.int64)])
        candidate.base_probabilities = np.vstack(
            [candidate.base_probabilities, base_probability]
        ).astype(np.float64)
        candidate.influences = np.concatenate(
            [candidate.influences, np.asarray([influence], dtype=np.float64)]
        )
        same_class = np.flatnonzero(candidate.labels == label)
        if len(same_class) > self.max_examples_per_class:
            remove = int(same_class[0])
            keep = np.arange(len(candidate.labels)) != remove
            candidate.prototypes = candidate.prototypes[keep]
            candidate.labels = candidate.labels[keep]
            candidate.base_probabilities = candidate.base_probabilities[keep]
            candidate.influences = candidate.influences[keep]
        return candidate

    def _gate(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        local_gain: float,
        *,
        rejection_feedback: bool,
        unknown_probability_shift: float,
    ) -> tuple[bool, dict[str, float]]:
        present = np.asarray(before["class_present"], dtype=bool)
        before_per_class = np.asarray(before["per_class_f1"], dtype=np.float64)
        after_per_class = np.asarray(after["per_class_f1"], dtype=np.float64)
        class_drops = before_per_class[present] - after_per_class[present]
        maximum_class_drop = float(np.max(class_drops)) if len(class_drops) else 0.0
        changes = {
            "objective_gain": float(local_gain),
            "probability_gain": 0.0 if rejection_feedback else float(local_gain),
            "rejection_gain": float(local_gain) if rejection_feedback else 0.0,
            "macro_f1_drop": float(before["macro_f1"] - after["macro_f1"]),
            "maximum_per_class_f1_drop": maximum_class_drop,
            "accuracy_drop": float(before["accuracy"] - after["accuracy"]),
            "log_loss_increase": float(after["log_loss"] - before["log_loss"]),
            "unknown_false_acceptance_increase": float(
                after["unknown_false_acceptance_rate"]
                - before["unknown_false_acceptance_rate"]
            ),
            "overall_rejection_accuracy_drop": float(
                before["overall_rejection_accuracy"]
                - after["overall_rejection_accuracy"]
            ),
            "maximum_unknown_probability_shift": float(unknown_probability_shift),
        }
        required_gain = (
            self.minimum_rejection_gain
            if rejection_feedback
            else self.minimum_probability_gain
        )
        passed = bool(
            local_gain >= required_gain
            and changes["macro_f1_drop"] <= self.allowed_macro_f1_drop + 1e-12
            and maximum_class_drop <= self.allowed_per_class_f1_drop + 1e-12
            and changes["accuracy_drop"] <= self.allowed_accuracy_drop + 1e-12
            and changes["log_loss_increase"] <= self.allowed_log_loss_increase + 1e-12
            and changes["unknown_false_acceptance_increase"] <= 1e-12
            and unknown_probability_shift
            <= self.allowed_unknown_probability_shift + 1e-12
        )
        return passed, changes

    def learn(
        self,
        feature_vector: Any,
        actual_label: str,
        *,
        probabilities: Any | None = None,
        base_probabilities: Any | None = None,
        mode: str = "safe",
        reviewed: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """Attempt a validation-gated update from one reviewed live sample."""

        if force or mode != "safe":
            raise PermissionError("Only validation-gated safe learning is supported.")
        if not reviewed:
            raise ValueError("Online learning accepts only explicitly reviewed feedback.")
        if actual_label not in self.class_to_idx and actual_label != self.reject_label:
            raise ValueError(f"Unknown actual label: {actual_label}")
        feature_rows, was_vector = self._coerce_features(feature_vector)
        if not was_vector:
            raise ValueError("Online learning accepts exactly one reviewed feature vector.")
        feature = feature_rows[0]
        supplied = base_probabilities if base_probabilities is not None else probabilities
        if supplied is None:
            base_probability = self._predict_base(feature_rows)[0]
        else:
            probability_rows, probability_was_vector = self._coerce_public_probabilities(supplied)
            if not probability_was_vector:
                raise ValueError("Online learning accepts exactly one base probability row.")
            base_probability = probability_rows[0]
        label = -1 if actual_label == self.reject_label else self.class_to_idx[actual_label]
        rejection_feedback = label == -1

        with self._lock:
            if self._load_error is not None:
                raise RuntimeError(self._load_error)
            current = self._state
            validation_x, validation_y, validation_p, validation_source, warnings = self._validation_rows(
                feature, label, base_probability, current
            )
            current_local_probabilities, current_local_acceptance, _ = (
                self._apply_state_with_quality(
                    base_probability.reshape(1, -1), feature.reshape(1, -1), current
                )
            )
            current_local = float(
                1.0 - current_local_acceptance[0]
                if rejection_feedback
                else current_local_probabilities[0, label]
            )
            before_probabilities, before_acceptance, _ = self._apply_state_with_quality(
                validation_p, validation_x, current
            )
            before_metrics = _metrics(
                before_probabilities,
                validation_y,
                CLASS_COUNT,
                before_acceptance,
                self.adapter_acceptance_floor,
            )

            selected: _AdapterState | None = None
            selected_metrics: dict[str, Any] | None = None
            selected_changes: dict[str, float] | None = None
            selected_influence: float | None = None
            candidate_local = current_local
            # Prefer the strongest correction that clears every safety gate.
            for influence in (1.0, 0.75, 0.5, 0.25, 0.125):
                candidate = self._candidate_with_example(
                    current, feature, label, base_probability, influence
                )
                candidate_local_probabilities, candidate_local_acceptance, _ = (
                    self._apply_state_with_quality(
                        base_probability.reshape(1, -1), feature.reshape(1, -1), candidate
                    )
                )
                candidate_local = float(
                    1.0 - candidate_local_acceptance[0]
                    if rejection_feedback
                    else candidate_local_probabilities[0, label]
                )
                after_probabilities, after_acceptance, _ = self._apply_state_with_quality(
                    validation_p, validation_x, candidate
                )
                after_metrics = _metrics(
                    after_probabilities,
                    validation_y,
                    CLASS_COUNT,
                    after_acceptance,
                    self.adapter_acceptance_floor,
                )
                unknown = validation_y == -1
                if np.any(unknown):
                    total_variation = 0.5 * np.abs(
                        after_probabilities[unknown] - before_probabilities[unknown]
                    ).sum(axis=1)
                    unknown_probability_shift = float(np.max(total_variation))
                else:
                    unknown_probability_shift = 0.0
                passed, changes = self._gate(
                    before_metrics,
                    after_metrics,
                    candidate_local - current_local,
                    rejection_feedback=rejection_feedback,
                    unknown_probability_shift=unknown_probability_shift,
                )
                if passed:
                    selected = candidate
                    selected_metrics = after_metrics
                    selected_changes = changes
                    selected_influence = influence
                    break
                selected_metrics = after_metrics
                selected_changes = changes

            now = _utc_now()
            attempt_number = current.attempted_updates + 1
            if selected is None:
                rejected = current.clone()
                rejected.attempted_updates = attempt_number
                rejected.rejected_updates += 1
                rejected.last_updated_utc = now
                rejected.last_validation = {
                    "accepted": False,
                    "actual_label": actual_label,
                    "source": validation_source,
                    "before": before_metrics,
                    "candidate": selected_metrics,
                    "changes": selected_changes,
                    "warnings": warnings,
                }
                self._atomic_save_state(rejected, self.state_path)
                self._state = rejected
                return {
                    "status": "rejected",
                    "mode": "safe",
                    "safety_gate_passed": False,
                    "actual_label": actual_label,
                    "probability_before": None if rejection_feedback else current_local,
                    "probability_after": None if rejection_feedback else candidate_local,
                    "rejection_score_before": current_local if rejection_feedback else None,
                    "rejection_score_after": candidate_local if rejection_feedback else None,
                    "validation_source": validation_source,
                    "validation_before": before_metrics,
                    "validation_after": selected_metrics,
                    "changes": selected_changes,
                    "warnings": warnings,
                    "counters": self._counter_dict(rejected),
                    "message": "Candidate rejected by the validation safety gate; live adaptation is unchanged.",
                }

            selected.attempted_updates = attempt_number
            selected.accepted_updates += 1
            selected.last_updated_utc = now
            selected.last_validation = {
                "accepted": True,
                "actual_label": actual_label,
                "source": validation_source,
                "before": before_metrics,
                "candidate": selected_metrics,
                "changes": selected_changes,
                "influence": selected_influence,
                "warnings": warnings,
            }
            backup = self._backup_state(current, attempt_number)
            self._atomic_save_state(selected, self.state_path)
            self._state = selected
            self._prune_backups()
            selected_local_probabilities, selected_local_acceptance, _ = (
                self._apply_state_with_quality(
                    base_probability.reshape(1, -1),
                    feature.reshape(1, -1),
                    selected,
                )
            )
            selected_local = float(
                1.0 - selected_local_acceptance[0]
                if rejection_feedback
                else selected_local_probabilities[0, label]
            )
            return {
                "status": "accepted",
                "mode": "safe",
                "safety_gate_passed": True,
                "actual_label": actual_label,
                "probability_before": None if rejection_feedback else current_local,
                "probability_after": None if rejection_feedback else selected_local,
                "rejection_score_before": current_local if rejection_feedback else None,
                "rejection_score_after": selected_local if rejection_feedback else None,
                "prototype_influence": selected_influence,
                "validation_source": validation_source,
                "validation_before": before_metrics,
                "validation_after": selected_metrics,
                "changes": selected_changes,
                "warnings": warnings,
                "backup": backup.name,
                "counters": self._counter_dict(selected),
                "message": "Reviewed correction passed validation and is live in the NumPy adapter.",
            }

    def safe_learn(
        self,
        feature_vector: Any,
        actual_label: str,
        probabilities: Any | None = None,
    ) -> dict[str, Any]:
        return self.learn(
            feature_vector,
            actual_label,
            probabilities=probabilities,
            mode="safe",
            reviewed=True,
        )

    @staticmethod
    def _counter_dict(state: _AdapterState) -> dict[str, int]:
        return {
            "attempted_updates": int(state.attempted_updates),
            "accepted_updates": int(state.accepted_updates),
            "rejected_updates": int(state.rejected_updates),
            "forced_updates": 0,
            "rollback_count": int(state.rollback_count),
            "active_prototypes": int(len(state.prototypes)),
            "negative_prototypes": int(np.count_nonzero(state.labels == -1)),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
            replay = self._cache_status(self.replay_path)
            validation = self._cache_status(self.validation_path)
            holdout_usable = any(
                row["usable"]
                and row.get("covered_classes") == CLASS_COUNT
                and row.get("has_reject_rows") is True
                for row in (replay, validation)
            )
            return {
                "ready": self._load_error is None,
                "safe_mode_ready": bool(
                    self._load_error is None
                    and (holdout_usable or not self.require_holdout_for_safe)
                ),
                "adapter_active": bool(len(state.prototypes)),
                "class_names": list(self.class_names),
                "reject_label": self.reject_label,
                "feedback_labels": [*self.class_names, self.reject_label],
                "class_count": CLASS_COUNT,
                "feature_count": FEATURE_COUNT,
                "direction_semantic_version": self.direction_semantic_version,
                "base_model_sha256": self.base_model_sha256,
                "state_path": str(self.state_path),
                "state_exists": self.state_path.is_file(),
                "state_error": self._load_error,
                "state_notice": self._state_notice,
                "quarantined_state_path": (
                    None
                    if self._quarantined_state_path is None
                    else str(self._quarantined_state_path)
                ),
                "replay_cache": replay,
                "validation_cache": validation,
                "require_holdout_for_safe": self.require_holdout_for_safe,
                **self._counter_dict(state),
                "update_count": int(state.accepted_updates),
                "last_updated_utc": state.last_updated_utc,
                "last_validation": state.last_validation,
                "backup_count": len(self.list_backups()),
                "force_learning_enabled": False,
                "policy": (
                    "Raw ONNX probabilities plus a bounded local NumPy adapter; "
                    "safe reviewed updates only; negative feedback scales the "
                    "existing known-gesture mass."
                ),
            }

    def _cache_status(self, path: Path) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": str(path),
            "present": path.is_file(),
            "valid": False,
            "usable": False,
            "rows": 0,
            "error": None,
        }
        if not path.is_file():
            return result
        try:
            cache = self._load_cache(path, need_probabilities=False)
            result["valid"] = True
            result["rows"] = int(len(cache.features))
            result["covered_classes"] = int(
                len(np.unique(cache.labels[cache.labels >= 0]))
            )
            result["has_reject_rows"] = bool(np.any(cache.labels == -1))
            result["probabilities_stored"] = cache.probabilities is not None
            result["usable"] = bool(cache.probabilities is not None or self._base_predictor is not None)
        except (OSError, ValueError, KeyError) as error:
            result["error"] = str(error)
        return result

    def _state_metadata(self, state: _AdapterState) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "class_names": list(self.class_names),
            "reject_label": self.reject_label,
            "feature_names": list(self.feature_names),
            "direction_semantic_version": self.direction_semantic_version,
            "base_model_sha256": self.base_model_sha256,
            "attempted_updates": int(state.attempted_updates),
            "accepted_updates": int(state.accepted_updates),
            "rejected_updates": int(state.rejected_updates),
            "rollback_count": int(state.rollback_count),
            "created_utc": state.created_utc,
            "last_updated_utc": state.last_updated_utc,
            "last_validation": state.last_validation,
            "adapter": {
                "adaptation_strength": self.adaptation_strength,
                "kernel_bandwidth": self.kernel_bandwidth,
                "minimum_similarity": self.minimum_similarity,
                "rejection_strength": self.rejection_strength,
                "adapter_acceptance_floor": self.adapter_acceptance_floor,
            },
        }

    def _atomic_save_state(self, state: _AdapterState, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            self._state_metadata(state), separators=(",", ":"), allow_nan=False
        )
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary_name = handle.name
                np.savez_compressed(
                    handle,
                    metadata_json=np.asarray(metadata),
                    feature_center=state.feature_center.astype(np.float64),
                    feature_scale=state.feature_scale.astype(np.float64),
                    prototypes=state.prototypes.astype(np.float32),
                    labels=state.labels.astype(np.int64),
                    base_probabilities=state.base_probabilities.astype(np.float64),
                    influences=state.influences.astype(np.float64),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass

    def _load_state(self, path: Path) -> _AdapterState:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "metadata_json", "feature_center", "feature_scale", "prototypes",
                "labels", "base_probabilities", "influences",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"missing state arrays: {sorted(missing)}")
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            center = np.asarray(archive["feature_center"], dtype=np.float64)
            scale = np.asarray(archive["feature_scale"], dtype=np.float64)
            prototypes = np.asarray(archive["prototypes"], dtype=np.float32)
            labels = np.asarray(archive["labels"], dtype=np.int64)
            base_probabilities = np.asarray(archive["base_probabilities"], dtype=np.float64)
            influences = np.asarray(archive["influences"], dtype=np.float64)

        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise _IncompatibleAdapterState("unsupported schema version")
        if tuple(metadata.get("class_names", ())) != self.class_names:
            raise _IncompatibleAdapterState(
                "class order differs from the configured ten-class contract"
            )
        if metadata.get("reject_label", "no_gesture") != self.reject_label:
            raise _IncompatibleAdapterState(
                "reject label differs from the configured runtime contract"
            )
        if tuple(metadata.get("feature_names", ())) != self.feature_names:
            raise _IncompatibleAdapterState(
                "feature order differs from the configured 76-D contract"
            )
        if metadata.get("direction_semantic_version") != self.direction_semantic_version:
            raise _IncompatibleAdapterState(
                "direction semantics differ from the calibrated NCM contract"
            )
        if (
            self.base_model_sha256 is not None
            and metadata.get("base_model_sha256") != self.base_model_sha256
        ):
            raise _IncompatibleAdapterState(
                "base ONNX model differs from the persisted adapter state"
            )
        adapter_metadata = metadata.get("adapter") or {}
        expected_parameters = {
            "adaptation_strength": self.adaptation_strength,
            "kernel_bandwidth": self.kernel_bandwidth,
            "minimum_similarity": self.minimum_similarity,
            "rejection_strength": self.rejection_strength,
            "adapter_acceptance_floor": self.adapter_acceptance_floor,
        }
        for name, expected in expected_parameters.items():
            try:
                actual = float(adapter_metadata[name])
            except (KeyError, TypeError, ValueError) as error:
                raise _IncompatibleAdapterState(
                    f"persisted adapter parameter is missing: {name}"
                ) from error
            if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
                raise _IncompatibleAdapterState(
                    f"persisted adapter parameter differs: {name}"
                )
        row_count = len(prototypes) if prototypes.ndim == 2 else -1
        if prototypes.shape != (row_count, FEATURE_COUNT):
            raise ValueError("prototype array has an invalid shape")
        if labels.shape != (row_count,) or base_probabilities.shape != (row_count, CLASS_COUNT):
            raise ValueError("persisted prototype rows are not aligned")
        if influences.shape != (row_count,):
            raise ValueError("persisted prototype influences are not aligned")
        if center.shape != (FEATURE_COUNT,) or scale.shape != (FEATURE_COUNT,):
            raise ValueError("persisted feature normalization has an invalid shape")
        numeric = [center, scale, prototypes, base_probabilities, influences]
        if not all(np.isfinite(value).all() for value in numeric):
            raise ValueError("persisted state contains non-finite values")
        if (scale <= 0).any() or (labels < -1).any() or (labels >= CLASS_COUNT).any():
            raise ValueError("persisted state contains out-of-range values")
        if (influences <= 0).any() or (influences > 1).any():
            raise ValueError("persisted influences must be in (0, 1]")
        base_probabilities = _soft_normalize(base_probabilities) if row_count else base_probabilities
        counters = [
            int(metadata.get("attempted_updates", 0)),
            int(metadata.get("accepted_updates", 0)),
            int(metadata.get("rejected_updates", 0)),
            int(metadata.get("rollback_count", 0)),
        ]
        if any(value < 0 for value in counters):
            raise ValueError("persisted counters cannot be negative")
        if counters[1] + counters[2] > counters[0]:
            raise ValueError("persisted counters are inconsistent")
        return _AdapterState(
            feature_center=center,
            feature_scale=scale,
            prototypes=prototypes,
            labels=labels,
            base_probabilities=base_probabilities,
            influences=influences,
            attempted_updates=counters[0],
            accepted_updates=counters[1],
            rejected_updates=counters[2],
            rollback_count=counters[3],
            created_utc=str(metadata.get("created_utc") or _utc_now()),
            last_updated_utc=metadata.get("last_updated_utc"),
            last_validation=metadata.get("last_validation"),
        )

    def _backup_state(self, state: _AdapterState, attempt_number: int) -> Path:
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        token = secrets.token_hex(3)
        path = self.backup_directory / (
            f"state_before_{attempt_number:06d}_{timestamp}_{token}.npz"
        )
        self._atomic_save_state(state, path)
        return path

    def _prune_backups(self) -> None:
        backups = sorted(
            self.backup_directory.glob("state_before_*.npz"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for path in backups[self.max_backups :]:
            try:
                path.unlink()
            except OSError:
                continue

    def list_backups(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.backup_directory.is_dir():
                return []
            rows: list[dict[str, Any]] = []
            for path in self.backup_directory.glob("state_before_*.npz"):
                try:
                    stat = path.stat()
                    rows.append(
                        {
                            "name": path.name,
                            "bytes": int(stat.st_size),
                            "modified_utc": datetime.fromtimestamp(
                                stat.st_mtime, timezone.utc
                            ).isoformat(),
                            "_modified_ns": stat.st_mtime_ns,
                        }
                    )
                except OSError:
                    continue
            rows.sort(key=lambda row: int(row["_modified_ns"]), reverse=True)
            for row in rows:
                row.pop("_modified_ns", None)
            return rows

    def rollback(self, backup_name: str) -> dict[str, Any]:
        if not backup_name or Path(backup_name).name != backup_name:
            raise ValueError("A plain managed backup file name is required.")
        backup_path = (self.backup_directory / backup_name).resolve()
        if backup_path.parent != self.backup_directory.resolve():
            raise ValueError("Backup path is outside the managed backup directory.")
        if not backup_path.is_file() or not backup_path.name.startswith("state_before_"):
            raise FileNotFoundError("The requested online-adapter backup does not exist.")
        with self._lock:
            restored = self._load_state(backup_path)
            current = self._state
            rollback_backup = self._backup_state(
                current, current.attempted_updates + 1
            )
            # Counters describe lifetime operations, not only the restored model.
            restored.attempted_updates = max(
                current.attempted_updates, restored.attempted_updates
            )
            restored.accepted_updates = max(
                current.accepted_updates, restored.accepted_updates
            )
            restored.rejected_updates = max(
                current.rejected_updates, restored.rejected_updates
            )
            restored.rollback_count = current.rollback_count + 1
            restored.last_updated_utc = _utc_now()
            restored.last_validation = {
                "accepted": True,
                "source": "rollback",
                "restored_backup": backup_name,
            }
            self._atomic_save_state(restored, self.state_path)
            self._state = restored
            self._load_error = None
            self._prune_backups()
            return {
                "status": "rolled_back",
                "restored_backup": backup_name,
                "current_state_backup": rollback_backup.name,
                "last_updated_utc": restored.last_updated_utc,
                "counters": self._counter_dict(restored),
            }


# The old Joblib edition used this service name.  Keeping the alias makes the
# ONNX integration a small import change while the implementation stays an
# adapter over immutable base probabilities.
OnlineLearningService = OnlineLearningAdapter


__all__ = ["OnlineLearningAdapter", "OnlineLearningService"]
