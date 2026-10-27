"""Retired Dorsal poses cannot become any command, even after a long hold."""
import numpy as np
import pytest

from backend.config import ACTIVE_CLASS_NAMES, CLASS_NAMES, DORSAL_REJECTION_REASON, RuntimeConfig
from backend.data_collection import collection_plan
from backend.geometry import GeometryResolver, hand_surface_orientation
from backend.hand_selection import gate_arguments
from backend.runtime import TemporalGate
from backend.tests.test_gesture_orientation import rotate_screen
from backend.tests.test_reported_peace_dorsal import reported_dorsal


@pytest.mark.parametrize("raw_label", [name for name in CLASS_NAMES if name not in {"stop_inverted", "open_palm"}])
@pytest.mark.parametrize("angle", range(-180, 180, 30))
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("thumb_offset", [(0, 0), (120, -100), (-120, 100)])
def test_dorsal_is_negative_for_every_classifier_label_and_wrist_angle(raw_label, angle, mirror, thumb_offset):
    # The two trained open-hand outputs use surface validation to separate
    # Palm from Stop inverted in the upper arc. Retired Dorsal and unrelated
    # classifier outputs must still never recover it.
    config = RuntimeConfig()
    points = reported_dorsal().copy()
    points[4] += thumb_offset
    points = rotate_screen(points, angle, mirror=mirror)
    side = next(side for side in ("Left", "Right")
                if hand_surface_orientation(points, side, .99)["dorsal_visible"])
    row = np.eye(len(CLASS_NAMES))[config.class_to_idx[raw_label]]
    resolved, details = GeometryResolver(config).resolve(
        row, points, handedness=side, handedness_confidence=.99)
    pose = details["pose_validation"]
    assert pose["gesture"] == "no_gesture"
    assert not pose["valid"] and not pose["geometry_supported"]
    assert pose["reason"] == DORSAL_REJECTION_REASON
    gate = TemporalGate(config)
    for timestamp in (1., 1.1, 1.2, 1.5, 1.8, 2.):
        decision = gate.update(resolved, now=timestamp,
                               **gate_arguments(details, {"known_gesture_mass": 1.}))
        assert decision.predicted_gesture == "no_gesture" and not decision.execute


def test_disabled_output_clears_previous_hold_even_with_every_recovery_flag():
    config = RuntimeConfig()
    gate = TemporalGate(config)
    row = np.eye(len(CLASS_NAMES))[config.class_to_idx["like"]]
    for timestamp in (1., 1.1, 1.2):
        decision = gate.update(row, now=timestamp)
    assert decision.execute
    dorsal = np.eye(len(CLASS_NAMES))[config.class_to_idx["dorsal"]]
    for timestamp in (1.3, 1.4, 1.8):
        decision = gate.update(dorsal, now=timestamp, geometry_supported=True,
                               directional_recovery=True, dorsal_range_recovery=True,
                               sideways_palm_recovery=True)
        assert not decision.execute and decision.predicted_gesture == "no_gesture"
    assert gate.update(row, now=1.9).stable_frames == 1


def test_dorsal_only_has_classifier_negative_collection_prompts():
    config = RuntimeConfig()
    assert "dorsal" not in config.gesture_to_action
    assert "dorsal" not in config.feedback_labels
    assert "dorsal" not in ACTIVE_CLASS_NAMES
    plan = collection_plan()
    assert not any(step["label"] == "dorsal" for step in plan)
    for view in ("front", "yaw_left", "yaw_right", "casual"):
        step = next(step for step in plan if step["id"] == f"no_gesture/dorsal_{view}")
        assert step["training_use"] == "hand_classifier"


def test_legacy_positive_dorsal_is_ingested_as_negative_without_rewriting_provenance(tmp_path):
    import json
    from pathlib import Path
    from backend.data_collection import CollectionStore
    from backend.geometry import landmarks_to_feature
    from backend.tests.test_collection_and_rejection import jpeg_bytes
    from research.collection_dataset import load_collection_dataset

    store = CollectionStore(tmp_path)
    store.add_participant("person-legacy")
    points = reported_dorsal() / 1000.
    store.observe(1, jpeg_bytes(), {"landmarks": points.tolist(),
                                  "feature_vector": landmarks_to_feature(points).tolist()})
    saved = store.save(token=store.freeze()["token"], participant_id="person-legacy",
                       session_id="old-session", step_id="no_gesture/dorsal_casual")
    image = Path(saved["image_path"])
    sidecar = image.with_suffix(".json")
    record = json.loads(sidecar.read_text())
    legacy = tmp_path / "participants/person-legacy/dorsal/casual/old-session" / image.name
    legacy.parent.mkdir(parents=True)
    image.rename(legacy)
    record.update(label="dorsal", view="casual", step_id="dorsal/casual",
                  image_path=legacy.relative_to(tmp_path).as_posix())
    legacy.with_suffix(".json").write_text(json.dumps(record))
    sidecar.unlink()

    data, report = load_collection_dataset(tmp_path)
    assert data["label"].tolist() == ["no_gesture"]
    assert data["y"].tolist() == [len(CLASS_NAMES)]
    assert report["policy_relabels"][0]["original_label"] == "dorsal"
    assert json.loads(legacy.with_suffix(".json").read_text())["label"] == "dorsal"


def test_legacy_dorsal_feedback_creates_only_a_negative_prototype(tmp_path):
    from backend.online_learning import OnlineLearningAdapter
    from backend.tests.test_online_learning import _base_row, _config
    adapter = OnlineLearningAdapter(_config(), models_directory=tmp_path)
    feature = np.zeros(76)
    result = adapter.learn(feature, "dorsal", probabilities=_base_row())
    assert result["status"] == "accepted"
    assert adapter._state.labels.tolist() == [-1]
    assert adapter.status()["negative_prototypes"] == 1


def test_legacy_dorsal_video_span_is_a_classifier_negative():
    from backend.video_lab import validate_segments
    span = validate_segments([{"start": 0, "end": 2, "label": "dorsal"}], 3)[0]
    assert span["label"] == "no_gesture"
    assert span["negative_reason"] == "dorsal_casual"
    assert span["training_use"] == "hand_classifier"


def test_persisted_positive_dorsal_prototype_reloads_as_negative(tmp_path):
    from backend.online_learning import OnlineLearningAdapter
    from backend.tests.test_online_learning import _base_row, _config
    adapter = OnlineLearningAdapter(_config(), models_directory=tmp_path)
    adapter.learn(np.zeros(76), "left", probabilities=_base_row())
    adapter._state.labels[0] = CLASS_NAMES.index("dorsal")
    adapter._atomic_save_state(adapter._state, adapter.state_path)
    reloaded = OnlineLearningAdapter(_config(), models_directory=tmp_path)
    assert reloaded._state.labels.tolist() == [-1]
    assert reloaded.status()["negative_prototypes"] == 1


def test_startup_contract_keeps_model_tensor_order_separate_from_active_feedback():
    import json
    from backend.config import MODEL_LABELS, MODELS_DIRECTORY, load_runtime_config
    from scripts.onnx_preflight import _contract_errors
    config = load_runtime_config()
    metadata = json.loads((MODELS_DIRECTORY / "gesture_mlp_onnx_metadata.json").read_text())
    assert metadata["internal_class_order"] == MODEL_LABELS
    assert "dorsal" not in config.feedback_labels
    assert _contract_errors(config, metadata) == []


def test_adapter_cannot_relabel_dorsal_but_its_negative_mass_still_applies(tmp_path):
    from types import SimpleNamespace
    from backend.tests.test_online_integration import _manager
    config = RuntimeConfig()
    replacement = np.eye(len(CLASS_NAMES))[config.class_to_idx["like"]]
    adapter = SimpleNamespace(apply_with_quality=lambda *args, **kwargs: (
        replacement, {"known_gesture_mass": .10, "negative_support": 1.}))
    manager, _, feature = _manager(tmp_path, adapter)
    manager.models["ONNX"].probabilities = np.eye(len(CLASS_NAMES))[config.class_to_idx["dorsal"]]
    probabilities, _, _, quality = manager.predict_detailed(feature)
    assert CLASS_NAMES[int(probabilities.argmax())] == "dorsal"
    assert quality["known_gesture_mass"] == .10

