from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MATPLOTLIB_CACHE = PROJECT_ROOT / ".runtime" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

from backend.artifact_integrity import ArtifactRegistry
from backend.config import CLASS_NAMES, GESTURE_TO_ACTION, MODELS_DIRECTORY, load_runtime_config
from backend.model_runtime import InferenceEngine


EXPECTED_CLASSES = CLASS_NAMES
EXPECTED_REJECT_LABEL = "no_gesture"
EXPECTED_FEEDBACK_LABELS = [*EXPECTED_CLASSES, EXPECTED_REJECT_LABEL]
EXPECTED_ACTIONS = GESTURE_TO_ACTION
EXPECTED_FEATURE_COUNT = 76
EXPECTED_TARGET_FPS = 10.0
MINIMUM_OFFLINE_ACCURACY = 0.985
MINIMUM_OFFLINE_MACRO_F1 = 0.98
MINIMUM_KNOWN_ACCEPTANCE = 0.96
MAXIMUM_UNKNOWN_FALSE_ACCEPTANCE = 0.03


def _contract_errors(config: object, metadata: dict[str, object]) -> list[str]:
    errors: list[str] = []
    class_names = list(getattr(config, "class_names", []))
    feedback_labels = list(getattr(config, "feedback_labels", []))
    actions = dict(getattr(config, "gesture_to_action", {}))
    feature_names = list(getattr(config, "feature_names", []))
    raw = dict(getattr(config, "raw", {}))

    if class_names != EXPECTED_CLASSES:
        errors.append(f"class order is {class_names}, expected {EXPECTED_CLASSES}")
    if EXPECTED_REJECT_LABEL in class_names:
        errors.append("no_gesture must not be an ONNX command output")
    if feedback_labels != EXPECTED_FEEDBACK_LABELS:
        errors.append(
            "feedback labels must be the ten commands followed by no_gesture"
        )
    if getattr(config, "reject_label", None) != EXPECTED_REJECT_LABEL:
        errors.append("runtime reject label is not no_gesture")
    if actions != EXPECTED_ACTIONS:
        errors.append("gesture-to-action mapping differs from the release contract")
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        errors.append(f"feature count is {len(feature_names)}, expected 76")
    if not np.isclose(float(getattr(config, "target_fps", 0.0)), EXPECTED_TARGET_FPS):
        errors.append("runtime target must be exactly 10 FPS")
    if int(getattr(config, "frame_interval_ms", 0)) != 100:
        errors.append("10 FPS runtime must use a 100 ms frame interval")

    camera = dict(raw.get("camera_orientation") or {})
    direction = dict(raw.get("directional_resolution") or {})
    low_light = dict(raw.get("low_light_enhancement") or {})
    landmarks = dict(raw.get("landmark_smoothing") or {})
    temporal = dict(raw.get("temporal_smoothing") or {})
    learning = dict(raw.get("online_learning") or {})
    if camera.get("mirror_horizontal") is not False:
        errors.append("camera_orientation.mirror_horizontal must be false")
    if direction.get("horizontal_mirror") is not False:
        errors.append("directional_resolution.horizontal_mirror must be false")
    if direction.get("horizontal_semantic_swap") is not True:
        errors.append("NCM Left/Right semantic calibration must be enabled")
    if low_light.get("enabled") is not True:
        errors.append("adaptive low-light enhancement must be enabled")
    if landmarks.get("algorithm") != "velocity_adaptive_ema":
        errors.append("velocity-adaptive landmark EMA must be configured")
    if temporal.get("algorithm") != "probability_ema_with_open_set_rejection":
        errors.append("open-set probability EMA must be configured")
    if learning.get("enabled") is not True:
        errors.append("Safe Learn adapter must be enabled")
    if learning.get("negative_feedback_label") != EXPECTED_REJECT_LABEL:
        errors.append("Safe Learn negative-feedback label must be no_gesture")

    if list(metadata.get("output_class_order") or []) != EXPECTED_CLASSES:
        errors.append("metadata output order differs from the ten-command contract")
    if metadata.get("known_mass_output") != "known_gesture_mass":
        errors.append("metadata does not declare the known-gesture-mass output")
    source_mapping = dict(metadata.get("source_class_mapping") or {})
    if not source_mapping.get("left") or not source_mapping.get("right"):
        errors.append("metadata must document the calibrated Left and Right sources")
    if not source_mapping.get("fist") or not source_mapping.get("thumb_down"):
        errors.append("metadata must document the Fist and Thumbs Down sources")
    horizontal_calibration = dict(
        metadata.get("horizontal_direction_calibration") or {}
    )
    if horizontal_calibration.get("camera_pixels_mirrored") is not False:
        errors.append("metadata must declare unmirrored camera pixels")
    if not (horizontal_calibration.get("source_columns_swapped") is True
            or horizontal_calibration.get("training_labels_ncm_calibrated") is True):
        errors.append("metadata must declare calibrated horizontal labels or a source-column swap")
    if horizontal_calibration.get("positive_index_dx_command") != "left":
        errors.append("metadata positive index dx must resolve to Left")
    if horizontal_calibration.get("negative_index_dx_command") != "right":
        errors.append("metadata negative index dx must resolve to Right")
    metadata_input = dict(metadata.get("input") or {})
    metadata_shape = list(metadata_input.get("shape") or [])
    if metadata_input.get("dtype") != "float32" or metadata_shape[-1:] != [76]:
        errors.append("metadata input must be float32 [N, 76]")
    runtime_note = str(metadata.get("runtime_note") or "")
    if "internal" not in runtime_note or "no_gesture" not in runtime_note:
        errors.append("metadata must document no_gesture as rejection-only at the command output")
    quality = dict(metadata.get("quality") or {})
    try:
        accuracy = float(quality["accuracy"])
        macro_f1 = float(quality["macro_f1"])
        known_acceptance = float(quality["known_acceptance_rate"])
        unknown_false_acceptance = float(quality["unknown_false_acceptance_rate"])
    except (KeyError, TypeError, ValueError):
        errors.append("metadata is missing numeric offline/open-set quality evidence")
    else:
        if accuracy < MINIMUM_OFFLINE_ACCURACY:
            errors.append("offline known-class accuracy is below 0.985")
        if macro_f1 < MINIMUM_OFFLINE_MACRO_F1:
            errors.append("offline macro F1 is below 0.98")
        if known_acceptance < MINIMUM_KNOWN_ACCEPTANCE:
            errors.append("offline known-sample acceptance is below 0.96")
        if unknown_false_acceptance > MAXIMUM_UNKNOWN_FALSE_ACCEPTANCE:
            errors.append("offline unknown false acceptance exceeds 0.03")
    return errors


def main() -> int:
    registry = ArtifactRegistry()
    integrity = registry.verify()
    if not integrity["verified"]:
        print("ONNX artifact integrity: FAIL")
        for error in integrity["errors"]:
            print(f"  - {error}")
        return 1
    print(
        "ONNX artifact integrity: PASS | release fingerprint: "
        + str(integrity["release_fingerprint"])
    )

    config = load_runtime_config()
    metadata_path = MODELS_DIRECTORY / "gesture_mlp_onnx_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"Ten-gesture metadata: FAIL | {error}")
        return 1
    errors = _contract_errors(config, metadata)
    if errors:
        print("Ten-gesture runtime contract: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(
        "Ten-gesture runtime contract: PASS | "
        "10 command outputs + no_gesture rejection | 10 FPS / 100 ms | "
        "unmirrored with calibrated Left/Right semantics"
    )

    engine = InferenceEngine(config)
    try:
        status = engine.status()
        if not status["ready"]:
            for error in status["model"]["errors"]:
                print(f"  - {error}")
            if status["mediapipe"].get("error"):
                print("  - " + str(status["mediapipe"]["error"]))
            return 1
        zero_features = np.zeros(EXPECTED_FEATURE_COUNT, dtype=np.float32)
        probabilities, name, _ = engine.model_manager.predict(zero_features)
        if name != "ONNX" or probabilities.shape != (len(EXPECTED_CLASSES),):
            print("ONNX classifier smoke test returned an unexpected contract.")
            return 1
        if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0):
            print("ONNX classifier smoke test returned invalid probabilities.")
            return 1
        classifier = engine.model_manager.models.get("ONNX")
        if classifier is None or not hasattr(classifier, "predict_with_quality"):
            print("ONNX classifier does not expose open-set quality output.")
            return 1
        probability_rows, known_mass = classifier.predict_with_quality(zero_features)
        if probability_rows.shape != (1, len(EXPECTED_CLASSES)):
            print("ONNX classifier returned an unexpected ten-probability shape.")
            return 1
        if known_mass.shape != (1,) or not np.isfinite(known_mass).all():
            print("ONNX classifier returned an invalid known-mass score.")
            return 1
        print("ONNX graph/session load: PASS | provider: CPUExecutionProvider")
        print("MediaPipe initialization: PASS")
        print(
            "Classifier smoke test: PASS | "
            "76 float32 features -> 10 probabilities + known-mass score"
        )
        quality = dict(metadata.get("quality") or {})
        print(
            "Offline qualification metadata: "
            f"accuracy={float(quality.get('accuracy', 0.0)):.5f} | "
            f"macro_f1={float(quality.get('macro_f1', 0.0)):.5f} | "
            "not a live NCM-camera measurement"
        )
        print(
            "Live Webcam and NCM validation: REQUIRED | test all ten gestures, "
            "both hands, varied angles, low light, empty scenes, landmark jitter, "
            "unmirrored directions, and observed FPS"
        )
        return 0
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
