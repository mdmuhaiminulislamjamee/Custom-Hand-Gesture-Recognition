"""Real hand shape remains valid when its screen/depth orientation changes.

These are synthetic landmark regressions, not pixel replays of the supplied
camera images. Context (waist, face contact, ownership) is tested separately.
"""

import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.geometry import GeometryResolver, hand_surface_orientation
from backend.hand_selection import gate_arguments
from backend.runtime import TemporalGate
from backend.tests.test_geometry import closed_hand_pose


SCREEN_ANGLES = tuple(range(-180, 180, 30))


def four_finger_hand():
    """Four long parallel fingers and a distinguishable thumb/palm winding."""
    points = np.zeros((21, 2), dtype=np.float32)
    points[0] = [.50, .90]
    points[1:5] = [[.42, .81], [.36, .72], [.32, .63], [.30, .54]]
    for mcp, x in zip((5, 9, 13, 17), (.445, .48, .515, .55)):
        points[mcp:mcp + 4] = [[x, .72], [x, .60], [x, .48], [x, .36]]
    return points


def rotate_screen(points, angle, *, mirror=False):
    radians = np.deg2rad(angle)
    rotation = np.asarray([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)],
    ])
    transformed = (points - points[0]) @ rotation.T + points[0]
    if mirror:
        transformed[:, 0] = 2 * transformed[0, 0] - transformed[:, 0]
    return transformed


def resolve(points, label, *, handedness=None, world=None):
    config = RuntimeConfig()
    row = np.eye(len(config.class_names))[config.class_to_idx[label]]
    resolved, details = GeometryResolver(config).resolve(
        row, points, handedness=handedness,
        handedness_confidence=.99 if handedness else None,
        world_landmarks=world, require_fist_depth=world is not None,
    )
    return config, resolved, details


def assert_held_geometry_recovers(config, probabilities, details, gesture):
    """Independent strong shape may recover low mass only after a real hold."""
    gate = TemporalGate(config)
    arguments = gate_arguments(details, dict(known_gesture_mass=.20))
    for timestamp in (1., 1.1, 1.2, 1.3):
        assert not gate.update(probabilities, now=timestamp, **arguments).execute
    decision = gate.update(probabilities, now=1.61, **arguments)
    assert decision.execute
    assert decision.predicted_gesture == gesture
    assert decision.known_gesture_mass == .20


@pytest.mark.parametrize("angle", SCREEN_ANGLES)
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("surface,label", [("dorsal", "dorsal"), ("palmar", "open_palm")])
def test_four_extended_fingers_accept_every_screen_rotation_and_both_hands(angle, mirror, surface, label):
    hand = four_finger_hand()
    if surface == "dorsal":
        hand[1:5] = [[.46, .83], [.425, .76], [.42, .68], [.42, .60]]
    points = rotate_screen(hand, angle, mirror=mirror)
    handedness = next(
        side for side in ("Right", "Left")
        if hand_surface_orientation(points, side, .99)["surface"] == surface
    )
    # The classifier can miss the domain. A strongly observed surface and
    # four extended fingers should resolve to the correct open-hand command.
    config, probabilities, details = resolve(points, "like", handedness=handedness)
    pose = details["pose_validation"]
    assert details["hand_shape"]["hand_surface"] == surface
    assert pose["valid"] and pose["geometry_supported"]
    assert pose["gesture"] == label
    assert config.class_names[int(probabilities.argmax())] == label
    assert_held_geometry_recovers(config, probabilities, details, label)


def rotated_closed_fist(angle, tilt, axis, mirror):
    """Apply actual rigid 3D rotations before projecting the closed joints."""
    original = closed_hand_pose(thumb_down=False)
    world = np.column_stack((original - original[0], np.zeros(21)))
    radians = np.deg2rad(tilt)
    cosine, sine = np.cos(radians), np.sin(radians)
    depth_rotation = (
        np.asarray([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
        if axis == "x" else
        np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    )
    radians = np.deg2rad(angle)
    screen_rotation = np.asarray([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians), np.cos(radians), 0],
        [0, 0, 1],
    ])
    world = world @ depth_rotation.T @ screen_rotation.T
    if mirror:
        world[:, 0] *= -1
    points = world[:, :2] + original[0]
    return points, world


@pytest.mark.parametrize("angle", SCREEN_ANGLES)
@pytest.mark.parametrize("mirror", [False, True])
def test_closed_fist_accepts_every_screen_rotation_with_world_joints(angle, mirror):
    points, world = rotated_closed_fist(angle, 0, "x", mirror)
    config, probabilities, details = resolve(points, "fist", world=world)
    pose = details["pose_validation"]
    assert pose["valid"] and pose["geometry_supported"]
    assert pose["gesture"] == "fist"
    assert config.class_names[int(probabilities.argmax())] == "fist"
    assert_held_geometry_recovers(config, probabilities, details, "fist")


@pytest.mark.parametrize("tilt", [-65, -35, 35, 65])
@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("mirror", [False, True])
def test_closed_fist_depth_rotation_toward_or_away_does_not_block_its_shape(tilt, axis, mirror):
    points, world = rotated_closed_fist(0, tilt, axis, mirror)
    config, probabilities, details = resolve(points, "fist", world=world)
    assert details["pose_validation"]["valid"]
    assert details["pose_validation"]["gesture"] == "fist"
    assert config.class_names[int(probabilities.argmax())] == "fist"


def test_recorded_fist_remains_fist_when_the_shipped_model_calls_its_rotation_thumb_down():
    from backend.config import PROJECT_ROOT, load_runtime_config
    from backend.geometry import landmarks_to_feature
    from backend.model_runtime import ModelManager

    config = load_runtime_config()
    manager = ModelManager(config)
    assert manager.ready, manager.errors
    with np.load(PROJECT_ROOT / "artifacts/ten_gesture/public_test.npz") as data:
        original = data["landmarks"][2900].copy()
    for angle in range(0, 360, 45):
        points = rotate_screen(original, angle)
        world = np.column_stack((points - points[0], np.zeros(21)))
        probabilities, _, _, quality = manager.predict_detailed(landmarks_to_feature(points))
        resolved, details = GeometryResolver(config).resolve(
            probabilities, points, handedness="Right", handedness_confidence=.99,
            world_landmarks=world, require_fist_depth=True,
        )
        assert details["pose_validation"]["valid"]
        assert config.class_names[int(resolved.argmax())] == "fist"
        gate = TemporalGate(config)
        for index in range(8):
            decision = gate.update(resolved, now=1 + index * .1,
                                   **gate_arguments(details, quality))
        assert decision.execute and decision.predicted_gesture == "fist"


@pytest.mark.parametrize("label,index", [("like", 2000), ("thumb_down", 3600)])
def test_recorded_thumb_command_survives_screen_rotations(label, index):
    from backend.config import PROJECT_ROOT, load_runtime_config
    from backend.geometry import landmarks_to_feature
    from backend.model_runtime import ModelManager

    config = load_runtime_config()
    manager = ModelManager(config)
    assert manager.ready, manager.errors
    with np.load(PROJECT_ROOT / "artifacts/ten_gesture/public_test.npz") as data:
        original = data["landmarks"][index].copy()
    for angle in range(0, 360, 45):
        points = rotate_screen(original, angle)
        world = np.column_stack((points - points[0], np.zeros(21)))
        probabilities, _, _, quality = manager.predict_detailed(landmarks_to_feature(points))
        resolved, details = GeometryResolver(config).resolve(
            probabilities, points, world_landmarks=world,
        )
        assert details["hand_shape"]["world_thumb_only"]
        assert details["pose_validation"]["valid"]
        assert config.class_names[int(resolved.argmax())] == label
        gate = TemporalGate(config)
        for frame in range(8):
            decision = gate.update(resolved, now=1 + frame * .1,
                                   **gate_arguments(details, quality))
        assert decision.execute and decision.predicted_gesture == label


def test_two_extended_fingers_cannot_take_the_rotated_thumb_shortcut():
    from backend.tests.test_geometry import reported_expanded_thumb_left_pose

    points = reported_expanded_thumb_left_pose()
    world = np.column_stack((points - points[0], np.zeros(21)))
    _, _, details = resolve(points, "like", world=world)
    assert not details["hand_shape"]["world_thumb_only"]


@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
@pytest.mark.parametrize("tilt", [-75, 75])
def test_depth_tilted_single_index_uses_world_shape_and_projected_direction(direction, tilt):
    from backend.tests.test_multiview_training import directional_points

    original = directional_points(direction)
    world = np.column_stack((original - original[0], np.zeros(21)))
    radians = np.deg2rad(tilt)
    cosine, sine = np.cos(radians), np.sin(radians)
    rotation = (
        np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
        if direction in {"left", "right"} else
        np.asarray([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
    )
    world = world @ rotation.T
    points = world[:, :2] + original[0]
    config, probabilities, details = resolve(points, "like", world=world)
    assert details["directional"]["world_directional"]
    assert details["pose_validation"]["valid"]
    assert config.class_names[int(probabilities.argmax())] == direction
    assert_held_geometry_recovers(config, probabilities, details, direction)


def test_index_aimed_directly_at_lens_has_no_invented_screen_direction():
    from backend.tests.test_multiview_training import directional_points

    original = directional_points("right")
    world = np.column_stack((original - original[0], np.zeros(21)))
    radians = np.deg2rad(89)
    rotation = np.asarray([[np.cos(radians), 0, np.sin(radians)],
                           [0, 1, 0], [-np.sin(radians), 0, np.cos(radians)]])
    world = world @ rotation.T
    points = world[:, :2] + original[0]
    _, _, details = resolve(points, "right", world=world)
    assert details["directional"]["world_index_only"]
    assert not details["directional"]["valid"]
    config, probabilities, continued = resolve_with_prior(points, world, "right")
    assert continued["directional"]["continued_direction"]
    assert continued["pose_validation"]["valid"]
    assert config.class_names[int(probabilities.argmax())] == "right"
    gate = TemporalGate(config)
    for index in range(6):
        decision = gate.update(probabilities, now=1 + index * .1,
                               **gate_arguments(continued, dict(known_gesture_mass=0.)))
    assert decision.execute and decision.predicted_gesture == "right"


def resolve_with_prior(points, world, direction):
    config = RuntimeConfig()
    probabilities = np.eye(len(config.class_names))[config.class_to_idx["like"]]
    resolved, details = GeometryResolver(config).resolve(
        probabilities, points, world_landmarks=world,
        preferred_direction=direction,
    )
    return config, resolved, details


def test_live_selected_hand_keeps_right_through_near_head_on_turn(monkeypatch):
    import threading
    from types import SimpleNamespace

    import cv2
    import backend.model_runtime as runtime_module
    from backend.model_runtime import InferenceEngine, RuntimeSession
    from backend.tests.test_hand_detection import simulated_detector
    from backend.tests.test_multiview_training import directional_points

    config = RuntimeConfig()
    original = directional_points("right")
    planar = np.column_stack((original - original[0], np.zeros(21)))

    def view(angle):
        radians = np.deg2rad(angle)
        rotation = np.asarray([[np.cos(radians), 0, np.sin(radians)],
                               [0, 1, 0], [-np.sin(radians), 0, np.cos(radians)]])
        world = planar @ rotation.T
        return world[:, :2] * 1.2 + [.5, .5], world

    angle = [75]
    points = [view(angle[0])]
    clock = [1.]
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: clock[0])
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine._lock = config, threading.Lock()
    engine.resolver = GeometryResolver(config)
    engine.detector = simulated_detector()
    engine.detector._run_candidates = lambda rgb, crop, *, video: [
        (.1, points[0][0], SimpleNamespace(category_name="Right", score=.99), points[0][1])
    ]
    probability = np.eye(len(config.class_names))[config.class_to_idx["like"]]
    quality = lambda: {"known_gesture_mass": 0. if angle[0] == 85 else .20}
    engine.model_manager = SimpleNamespace(
        ready=True, models={"ONNX": object()}, resolve_name=lambda name: "ONNX",
        predict_detailed=lambda *args: (probability, "ONNX", .01, quality()),
        predict_many=lambda *args: {"ONNX": (probability, .01, quality())},
    )
    frame = cv2.imencode(".jpg", np.full((480, 640, 3), 140, np.uint8))[1].tobytes()
    session = RuntimeSession(config)
    for index in range(10):
        clock[0] = 1 + index * .1
        result = engine.process_frame(frame, session)
    assert result["runtime_prediction"] == "right"
    owner = result["hand_selection"]["selected_track_id"]
    angle[0], points[0] = 85, view(85)
    for index in range(1, 8):
        clock[0] = 2 + index * .1
        result = engine.process_frame(frame, session)
        assert result["runtime_prediction"] == "right"
        assert result["hand_selection"]["selected_track_id"] == owner
    angle[0], points[0] = 105, view(105)
    for index in range(1, 10):
        clock[0] = 3 + index * .1
        result = engine.process_frame(frame, session)
    assert result["runtime_prediction"] == "left"
    assert result["hand_selection"]["selected_track_id"] == owner
    fresh = RuntimeSession(config)
    angle[0], points[0] = 85, view(85)
    clock[0] = 5.
    first_view = engine.process_frame(frame, fresh)
    assert first_view["runtime_prediction"] == config.reject_label


@pytest.mark.parametrize("angle", SCREEN_ANGLES)
@pytest.mark.parametrize("shape", ["folded_middle", "rock", "peace"])
@pytest.mark.parametrize("model_label", ["dorsal", "open_palm", "fist"])
def test_rotation_cannot_promote_partially_extended_fingers_to_open_hand_or_fist(angle, shape, model_label):
    points = four_finger_hand()
    folded_mcps = {"folded_middle": (9,), "rock": (9, 13), "peace": (13, 17)}[shape]
    for mcp in folded_mcps:
        x, y = points[mcp]
        points[mcp:mcp + 4] = [[x, y], [x, y - .12], [x + .05, y - .07], [x + .015, y + .02]]
    points = rotate_screen(points, angle)
    handedness = next(
        side for side in ("Right", "Left")
        if hand_surface_orientation(points, side, .99)["surface"] ==
        ("palmar" if model_label == "open_palm" else "dorsal")
    )
    _, _, details = resolve(points, model_label, handedness=handedness)
    pose = details["pose_validation"]
    # A genuine Peace recovery is permitted. It must never turn into one of
    # the commands whose finger extension/closure requirements are unmet.
    assert not (pose["valid"] and pose["gesture"] in {"dorsal", "open_palm", "fist"})


def approximate_raised_dorsal():
    """Approximate first-report image joints, without claiming pixel replay."""
    return np.asarray([
        [662, 309], [684, 320], [699, 347], [719, 375], [746, 392],
        [712, 380], [714, 417], [713, 436], [715, 449],
        [691, 373], [694, 414], [693, 446], [696, 463],
        [669, 373], [671, 414], [674, 451], [678, 471],
        [648, 371], [646, 401], [644, 421], [644, 435],
    ], dtype=np.float32)


@pytest.mark.parametrize("mirror", [False, True])
def test_approximate_raised_dorsal_image_keeps_four_downward_fingers(mirror):
    points = approximate_raised_dorsal()
    if mirror:
        points[:, 0] = 1324 - points[:, 0]
    handedness = "Left" if mirror else "Right"
    config, probabilities, details = resolve(points, "dorsal", handedness=handedness)
    assert details["hand_shape"]["hand_surface"] == "dorsal"
    assert details["pose_validation"]["valid"]
    assert config.class_names[int(probabilities.argmax())] == "dorsal"


@pytest.mark.parametrize("mirror", [False, True])
def test_shipped_onnx_approximate_raised_dorsal_reaches_a_real_held_command(mirror):
    from backend.geometry import landmarks_to_feature
    from backend.model_runtime import ModelManager

    config = RuntimeConfig()
    manager = ModelManager(config)
    assert manager.ready, manager.errors
    points = approximate_raised_dorsal()
    if mirror:
        points[:, 0] = 1324 - points[:, 0]
    points = points / [960, 500]
    probabilities, _, _, quality = manager.predict_detailed(landmarks_to_feature(points))
    resolved, details = GeometryResolver(config).resolve(
        probabilities, points, handedness="Left" if mirror else "Right",
        handedness_confidence=.99,
    )
    assert details["pose_validation"]["valid"]
    assert details["pose_validation"]["gesture"] == "dorsal"
    gate = TemporalGate(config)
    for index in range(8):
        decision = gate.update(resolved, now=1 + index * .1, **gate_arguments(details, quality))
    assert decision.execute
    assert decision.predicted_gesture == "dorsal"
    assert decision.known_gesture_mass == quality["known_gesture_mass"]
