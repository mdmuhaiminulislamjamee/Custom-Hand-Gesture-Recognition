"""Regression checks for Peace, Rock, and the retired call sign."""
from __future__ import annotations

import json
import numpy as np

from backend.config import CLASS_NAMES, PROJECT_ROOT, RuntimeConfig, load_runtime_config
from backend.geometry import GeometryResolver, landmarks_to_feature, peace_finger_geometry
from backend.model_runtime import ModelManager
from backend.runtime import TemporalGate
from scripts.train_twelve_gesture_candidate import REJECT, _relabel


COMMANDS = CLASS_NAMES


def _source_hand(variant: str, *, thumb: str | None = None) -> np.ndarray:
    with np.load(PROJECT_ROOT / "data/ten_gesture/hagrid_train.npz", allow_pickle=False) as data:
        matching = np.flatnonzero(data["variant"] == variant)
        if thumb == "extended":
            matching = matching[data["X"][matching, 68] >= .7]
        elif thumb == "folded":
            matching = matching[data["X"][matching, 68] < .4]
        if variant == "source_peace":
            features = data["X"][matching]
            matching = matching[(features[:, 69] >= .65) & (features[:, 70] >= .65)
                                & (features[:, 71] <= .55) & (features[:, 72] <= .55)]
        assert len(matching)
        return data["landmarks"][matching[0]].copy()


def _probabilities(name: str) -> np.ndarray:
    row = np.zeros(len(COMMANDS), dtype=np.float64)
    row[COMMANDS.index(name)] = .98
    row[COMMANDS.index("left")] += .02 if name != "left" else 0.0
    return row


def _edge_on_peace_pose() -> np.ndarray:
    """Approximate the reported camera view with two long raised fingers."""
    return np.asarray([
        [469, 626], [493, 569], [503, 541], [526, 545], [540, 553],
        [455, 550], [450, 513], [458, 466], [460, 402],
        [475, 555], [472, 508], [477, 462], [477, 405],
        [492, 557], [504, 525], [514, 492], [516, 491],
        [505, 578], [518, 554], [530, 542], [540, 552],
    ], dtype=np.float32)


def _close_finger_peace_pose() -> np.ndarray:
    """Reported Peace shape with long, nearly parallel index/middle fingers."""
    points = _edge_on_peace_pose()
    points[6:9, 0] += 12
    return points


def test_requested_candidate_actions_are_mapped():
    contract = json.loads((PROJECT_ROOT / "artifacts/twelve_gesture/candidate_contract.json").read_text())
    assert contract["class_names"] == COMMANDS
    assert contract["gesture_to_action"]["peace"] == "Turn 90 Degrees"
    assert "source_call" not in contract["class_names"]
    assert contract["gesture_to_action"]["rock"] == "Backup"


def test_retired_call_source_is_a_rejection_class_with_heldout_gate():
    row = {
        "y": np.asarray([10], dtype=np.int64),
        "variant": np.asarray(["source_call"]),
    }
    assert _relabel(row)["y"].tolist() == [REJECT]
    evaluation = json.loads((PROJECT_ROOT / "artifacts/twelve_gesture/evaluation.json").read_text())
    assert evaluation["test"]["retired_call_negative_count"] >= 100
    assert evaluation["test"]["retired_call_false_accept_rate"] <= .02


def test_multi_finger_model_labels_are_not_rewritten_as_directions():
    resolver = GeometryResolver(RuntimeConfig(class_names=COMMANDS))
    for name, variant in (("peace", "source_peace"),
                          ("rock", "source_rock")):
        resolved, details = resolver.resolve(_probabilities(name), _source_hand(variant))
        assert COMMANDS[int(resolved.argmax())] == name
        assert details["pose_validation"]["valid"]


def test_rock_accepts_folded_and_extended_thumb():
    resolver = GeometryResolver(RuntimeConfig(class_names=COMMANDS))
    for thumb in ("folded", "extended"):
        resolved, details = resolver.resolve(
            _probabilities("rock"), _source_hand("source_rock", thumb=thumb)
        )
        assert COMMANDS[int(resolved.argmax())] == "rock"
        assert details["pose_validation"]["valid"]


def test_clear_peace_overrides_ambiguous_one_finger_score():
    resolver = GeometryResolver(RuntimeConfig(class_names=COMMANDS))
    row = np.zeros(len(COMMANDS), dtype=np.float64)
    row[COMMANDS.index("left")] = .60
    row[COMMANDS.index("peace")] = .40
    resolved, details = resolver.resolve(row, _source_hand("source_peace"))
    assert COMMANDS[int(resolved.argmax())] == "peace"
    assert details["pose_validation"]["valid"]


def test_edge_on_peace_with_short_ring_finger_overrides_direction():
    config = RuntimeConfig(class_names=COMMANDS)
    image = _edge_on_peace_pose()
    row = np.zeros(len(COMMANDS), dtype=np.float64)
    row[COMMANDS.index("up")] = .85
    row[COMMANDS.index("peace")] = .15

    resolved, details = GeometryResolver(config).resolve(row, image)
    shape = details["hand_shape"]["peace_image_geometry"]
    pose = details["pose_validation"]
    assert shape["valid"]
    assert shape["ring_reach_ratio"] < .65
    assert pose["gesture"] == "peace"
    assert pose["valid"] and pose["geometry_supported"]
    assert COMMANDS[int(resolved.argmax())] == "peace"


def test_close_parallel_index_and_middle_fingers_execute_peace():
    config = load_runtime_config()
    image = _close_finger_peace_pose()
    world = np.pad(image, ((0, 0), (0, 1)))
    geometry = peace_finger_geometry(world)
    assert .04 < geometry["tip_separation_ratio"] < .22
    assert geometry["valid"]

    model = ModelManager(config)
    probabilities, _, _, quality = model.predict_detailed(landmarks_to_feature(image))
    resolved, details = GeometryResolver(config).resolve(
        probabilities, image, world_landmarks=world,
    )
    pose = details["pose_validation"]
    assert pose["gesture"] == "peace"
    assert pose["valid"] and pose["geometry_supported"]

    gate = TemporalGate(config)
    for now in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        decision = gate.update(
            resolved, now=now,
            known_gesture_mass=quality["known_gesture_mass"],
            pose_valid=pose["valid"], geometry_supported=pose["geometry_supported"],
        )
    assert decision.predicted_gesture == "peace"
    assert decision.execute


def test_collapsed_duplicate_fingertips_are_not_peace():
    image = _edge_on_peace_pose()
    image[8] = image[12]
    geometry = peace_finger_geometry(image)
    assert geometry["tip_separation_ratio"] == 0.0
    assert not geometry["valid"]


def test_edge_on_peace_rejects_a_truly_raised_third_finger():
    image = _edge_on_peace_pose()
    image[14:17] = np.asarray([
        [502, 510], [509, 450], [516, 405],
    ], dtype=np.float32)
    assert not peace_finger_geometry(image)["valid"]


def test_relaxed_ring_check_does_not_accept_a_wide_no_gesture_pose():
    with np.load(PROJECT_ROOT / "data/ten_gesture/hagrid_train.npz", allow_pickle=False) as data:
        assert data["variant"][27045] == "source_no_gesture"
        image = data["landmarks"][27045]
    geometry = peace_finger_geometry(image)
    assert geometry["ring_reach_ratio"] < .65
    assert geometry["tip_separation_ratio"] > .35
    assert not geometry["valid"]


def test_shipped_onnx_accepts_edge_on_peace_after_hold():
    config = load_runtime_config()
    image = _edge_on_peace_pose()
    model = ModelManager(config)
    probabilities, _, _, quality = model.predict_detailed(landmarks_to_feature(image))
    resolved, details = GeometryResolver(config).resolve(probabilities, image)
    pose = details["pose_validation"]
    assert pose["gesture"] == "peace"
    assert pose["valid"] and pose["geometry_supported"]

    gate = TemporalGate(config)
    for now in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        decision = gate.update(
            resolved, now=now,
            known_gesture_mass=quality["known_gesture_mass"],
            pose_valid=pose["valid"], geometry_supported=pose["geometry_supported"],
        )
    assert decision.predicted_gesture == "peace"
    assert decision.execute


def test_clear_peace_with_weak_model_support_cannot_trigger_one_finger_action():
    resolver = GeometryResolver(RuntimeConfig(class_names=COMMANDS))
    row = np.zeros(len(COMMANDS), dtype=np.float64)
    row[COMMANDS.index("left")] = .90
    row[COMMANDS.index("peace")] = .10
    resolved, details = resolver.resolve(row, _source_hand("source_peace"))
    assert COMMANDS[int(resolved.argmax())] == "left"
    assert not details["pose_validation"]["valid"]


def test_foreshortened_two_finger_pose_recovers_peace_with_low_model_mass():
    config = RuntimeConfig(class_names=COMMANDS)
    resolver = GeometryResolver(config)
    world = np.pad(_source_hand("source_peace"), ((0, 0), (0, 1)))
    image = world[:, :2].copy()
    image[6:9] = image[5] + .35 * (image[6:9] - image[5])
    row = np.zeros(len(COMMANDS))
    row[COMMANDS.index("left")] = .90
    row[COMMANDS.index("peace")] = .10

    resolved, details = resolver.resolve(row, image, world_landmarks=world)
    pose = details["pose_validation"]
    assert not details["hand_shape"]["peace_image_geometry"]["valid"]
    assert details["hand_shape"]["peace_world_geometry"]["valid"]
    assert pose["gesture"] == "peace"
    assert pose["valid"] and pose["geometry_supported"]

    gate = TemporalGate(config)
    for now in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6):
        decision = gate.update(
            resolved, now=now, known_gesture_mass=.31,
            pose_valid=pose["valid"], geometry_supported=pose["geometry_supported"],
        )
    assert decision.predicted_gesture == "peace"
    assert decision.execute


def test_single_middle_finger_vetoes_confident_peace():
    config = RuntimeConfig(class_names=COMMANDS)
    resolver = GeometryResolver(config)
    image = _source_hand("source_peace")
    world = np.pad(image, ((0, 0), (0, 1)))
    world[6:9] = world[5] + .20 * (world[6:9] - world[5])

    resolved, details = resolver.resolve(
        _probabilities("peace"), image, world_landmarks=world,
    )
    pose = details["pose_validation"]
    assert details["hand_shape"]["peace_image_geometry"]["valid"]
    assert not details["hand_shape"]["peace_world_geometry"]["valid"]
    assert COMMANDS[int(resolved.argmax())] == "peace"
    assert not pose["valid"]
    decision = TemporalGate(config).update(
        resolved, known_gesture_mass=.98, pose_valid=pose["valid"],
    )
    assert decision.predicted_gesture == "no_gesture"


def test_single_middle_finger_without_depth_is_not_peace():
    config = RuntimeConfig(class_names=COMMANDS)
    image = _source_hand("source_peace")
    image[6:9] = image[5] + .20 * (image[6:9] - image[5])
    _, details = GeometryResolver(config).resolve(_probabilities("peace"), image)
    assert not details["pose_validation"]["valid"]


def test_world_peace_shape_does_not_override_unrelated_model_evidence():
    config = RuntimeConfig(class_names=COMMANDS)
    image = _source_hand("source_peace")
    world = np.pad(image, ((0, 0), (0, 1)))
    row = np.zeros(len(COMMANDS))
    row[COMMANDS.index("left")] = .98
    row[COMMANDS.index("peace")] = .02
    resolved, _ = GeometryResolver(config).resolve(
        row, image, world_landmarks=world,
    )
    assert COMMANDS[int(resolved.argmax())] != "peace"


def test_low_mass_peace_shape_still_needs_vocabulary_evidence():
    config = RuntimeConfig(class_names=COMMANDS)
    image = _source_hand("source_peace")
    resolved, details = GeometryResolver(config).resolve(
        _probabilities("peace"), image,
    )
    pose = details["pose_validation"]
    decision = TemporalGate(config).update(
        resolved, known_gesture_mass=.13,
        pose_valid=pose["valid"], geometry_supported=pose["geometry_supported"],
    )
    assert decision.predicted_gesture == "no_gesture"
