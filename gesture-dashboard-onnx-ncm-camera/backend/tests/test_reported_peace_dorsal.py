"""Approximate visible screenshot skeletons, not MediaPipe pixel replays."""
import numpy as np
import pytest

from backend.config import PROJECT_ROOT, RuntimeConfig, load_runtime_config
from backend.geometry import GeometryResolver, hand_surface_orientation, landmarks_to_feature
from backend.runtime import TemporalGate


def reported_peace(image_number):
    # Wrist, thumb, index, middle, ring, little in MediaPipe order.
    poses = {
        1: [[450,540], [482,500], [493,461], [516,467], [524,475],
            [440,450], [455,390], [470,358], [484,337],
            [450,430], [445,385], [440,356], [436,330],
            [443,459], [493,440], [497,422], [525,463],
            [449,469], [490,461], [515,464], [524,475]],
        3: [[417,542], [445,500], [462,465], [489,462], [504,471],
            [414,419], [422,365], [428,331], [430,297],
            [403,429], [408,377], [415,334], [420,300],
            [403,455], [455,420], [482,433], [488,438],
            [410,461], [463,443], [493,450], [503,458]],
        4: [[453,550], [497,509], [515,474], [537,465], [547,474],
            [473,410], [484,363], [494,330], [496,297],
            [453,420], [466,365], [480,326], [489,290],
            [444,441], [499,416], [529,432], [541,436],
            [447,471], [502,452], [527,460], [549,459]],
        5: [[419,484], [454,454], [486,425], [506,429], [530,454],
            [472,374], [515,337], [541,308], [564,286],
            [461,380], [503,337], [535,307], [562,292],
            [451,398], [496,399], [516,397], [525,397],
            [445,423], [497,427], [533,423], [535,444]],
        6: [[445,516], [472,475], [490,429], [505,428], [524,449],
            [442,380], [454,332], [463,299], [470,270],
            [430,393], [443,340], [457,298], [466,268],
            [426,416], [480,384], [493,382], [500,388],
            [438,441], [483,417], [500,419], [505,427]],
    }
    return np.asarray(poses[image_number], dtype=np.float32)


def reported_dorsal():
    return np.asarray([
        [455,418], [499,429], [526,457], [539,486], [545,506],
        [525,492], [525,529], [523,544], [521,555],
        [503,501], [503,537], [501,554], [502,567],
        [481,505], [484,537], [485,554], [488,567],
        [458,503], [461,529], [464,542], [467,553],
    ], dtype=np.float32)


def world_for_image(points):
    # Synthetic depth preserving the displayed silhouette. Real depth is not
    # recoverable from screenshots; separate tests exercise depth disagreement.
    return np.column_stack((points, -.4 * (points[:, 1] - points[0, 1])))


@pytest.mark.parametrize('image_number', [1, 3, 4, 5, 6])
@pytest.mark.parametrize('mirror', [False, True])
@pytest.mark.parametrize('with_depth', [False, True])
def test_reported_two_finger_poses_override_confident_up(image_number, mirror, with_depth):
    points = reported_peace(image_number)
    if mirror:
        points[:, 0] *= -1
    config = RuntimeConfig()
    row = np.eye(len(config.class_names))[config.class_to_idx['up']]
    resolved, details = GeometryResolver(config).resolve(
        row, points, world_landmarks=world_for_image(points) if with_depth else None,
    )
    pose = details['pose_validation']
    assert pose['gesture'] == 'peace'
    assert pose['valid'] and pose['geometry_supported']
    gate = TemporalGate(config)
    for now in (1., 1.1, 1.2, 1.3, 1.4):
        decision = gate.update(resolved, now=now, known_gesture_mass=1.,
                               pose_valid=pose['valid'], geometry_supported=pose['geometry_supported'])
    assert decision.predicted_gesture == 'peace' and decision.execute


def rotate_to_xy(points, xy):
    axis = points[[5, 9, 13, 17]].mean(axis=0) - points[0]
    angle = np.deg2rad(xy) - np.arctan2(axis[1], axis[0])
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    return (points - points[0]) @ rotation.T + points[0]


@pytest.mark.parametrize('xy', [0, 1, 15, 30, 67, 90, 120, 149, 150])
@pytest.mark.parametrize('mirror', [False, True])
def test_reported_dorsal_accepts_requested_angle_range_with_low_mass(xy, mirror):
    points = reported_dorsal()
    if mirror:
        points[:, 0] *= -1
    points = rotate_to_xy(points, xy)
    config = RuntimeConfig()
    hand = next(hand for hand in ('Left', 'Right')
                if hand_surface_orientation(points, hand, .99)['dorsal_visible'])
    row = np.eye(len(config.class_names))[config.class_to_idx['open_palm']]
    resolved, details = GeometryResolver(config).resolve(
        row, points, handedness=hand, handedness_confidence=.99,
        world_landmarks=world_for_image(points),
    )
    pose = details['pose_validation']
    assert pose['gesture'] == 'dorsal'
    assert pose['valid'] and pose['geometry_supported']
    gate = TemporalGate(config)
    for now in (1., 1.1, 1.2, 1.3, 1.4, 1.5):
        decision = gate.update(
            resolved, now=now, known_gesture_mass=.05,
            pose_valid=pose['valid'], geometry_supported=pose['geometry_supported'],
            dorsal_range_recovery=pose['dorsal_range_recovery'],
        )
        if now < 1.4:
            assert not decision.execute
    assert decision.predicted_gesture == 'dorsal' and decision.execute
    assert decision.known_gesture_mass == .05


@pytest.mark.parametrize('xy', [-90, -1, 151, 180])
def test_dorsal_rejects_angles_outside_requested_range(xy):
    config = RuntimeConfig()
    points = rotate_to_xy(reported_dorsal(), xy)
    hand = next(hand for hand in ('Left', 'Right')
                if hand_surface_orientation(points, hand, .99)['dorsal_visible'])
    row = np.eye(len(config.class_names))[config.class_to_idx['dorsal']]
    _, details = GeometryResolver(config).resolve(
        row, points, handedness=hand, handedness_confidence=.99,
        world_landmarks=world_for_image(points),
    )
    assert not details['pose_validation']['valid']
    assert not details['pose_validation']['dorsal_range_recovery']


@pytest.mark.parametrize('variant', ['palmar', 'unknown_surface', 'curled', 'depth_outside'])
def test_low_mass_dorsal_recovery_requires_verified_back_and_four_extended_fingers(variant):
    config = RuntimeConfig()
    points = reported_dorsal()
    hand = next(hand for hand in ('Left', 'Right')
                if hand_surface_orientation(points, hand, .99)['dorsal_visible'])
    world = world_for_image(points)
    if variant == 'palmar':
        hand = 'Left' if hand == 'Right' else 'Right'
    elif variant == 'unknown_surface':
        hand = None
    elif variant == 'curled':
        points[8] = points[5] + [0, 2]
        world = world_for_image(points)
    else:
        world = world_for_image(rotate_to_xy(points, -30))
    row = np.eye(len(config.class_names))[config.class_to_idx['dorsal']]
    resolved, details = GeometryResolver(config).resolve(
        row, points, handedness=hand, handedness_confidence=.99, world_landmarks=world,
    )
    pose = details['pose_validation']
    assert not pose['dorsal_range_recovery']
    gate = TemporalGate(config)
    for now in (1., 1.2, 1.4, 1.6):
        decision = gate.update(resolved, now=now, known_gesture_mass=.05,
                               pose_valid=pose['valid'], geometry_supported=pose['geometry_supported'],
                               dorsal_range_recovery=pose['dorsal_range_recovery'])
        assert not decision.execute


@pytest.mark.parametrize('finger', [5, 9, 13, 17])
def test_strong_peace_recovery_requires_exactly_two_raised_fingers(finger):
    config = RuntimeConfig()
    points = reported_peace(4)
    if finger in (5, 9):
        points[finger + 1:finger + 4] = points[finger] + .15 * (
            points[finger + 1:finger + 4] - points[finger])
    else:
        points[finger + 1:finger + 4] = points[finger] + (points[6:9] - points[5])
    row = np.eye(len(config.class_names))[config.class_to_idx['up']]
    _, details = GeometryResolver(config).resolve(row, points, world_landmarks=world_for_image(points))
    assert not details['hand_shape']['peace_pair_recovery']
    assert details['pose_validation']['gesture'] != 'peace'


def test_real_pointing_tracks_with_duplicate_middle_joints_do_not_become_peace():
    config = load_runtime_config()
    from backend.model_runtime import ModelManager
    model = ModelManager(config)
    with np.load(PROJECT_ROOT / 'data/ten_gesture/hagrid_train.npz', allow_pickle=False) as data:
        for i in (248, 249, 250, 251):
            assert data['variant'][i] == 'direction_rotation'
            points = data['landmarks'][i]
            probabilities, _, _, _ = model.predict_detailed(landmarks_to_feature(points))
            _, details = GeometryResolver(config).resolve(probabilities, points)
            assert not details['hand_shape']['peace_pair_recovery']
            assert details['pose_validation']['gesture'] != 'peace'


def test_reported_peace_shapes_resolve_production_onnx_directions():
    from backend.model_runtime import ModelManager
    config = load_runtime_config()
    model = ModelManager(config)
    for image_number in (1, 3, 4, 5, 6):
        points = reported_peace(image_number)
        probabilities, _, _, quality = model.predict_detailed(landmarks_to_feature(points))
        resolved, details = GeometryResolver(config).resolve(
            probabilities, points, world_landmarks=world_for_image(points))
        pose = details['pose_validation']
        assert pose['gesture'] == 'peace' and pose['valid']
        gate = TemporalGate(config)
        for now in (1., 1.2, 1.4, 1.6):
            decision = gate.update(resolved, now=now, known_gesture_mass=quality['known_gesture_mass'],
                                   pose_valid=pose['valid'], geometry_supported=pose['geometry_supported'])
        # Approximate skeletons do not reproduce the screenshot's model mass.
        # Preserve vocabulary rejection if the reconstructed input is unknown.
        if quality['known_gesture_mass'] < config.geometry_recovery_mass_floor:
            assert decision.predicted_gesture == 'no_gesture' and not decision.execute
        else:
            assert decision.predicted_gesture == 'peace' and decision.execute


def test_live_inference_emits_low_mass_dorsal_action(monkeypatch):
    import cv2
    import threading
    from types import SimpleNamespace
    from backend.model_runtime import InferenceEngine, RuntimeSession
    import backend.model_runtime as runtime_module

    config = RuntimeConfig()
    points = reported_dorsal() / 960.
    hand = next(hand for hand in ('Left', 'Right')
                if hand_surface_orientation(points, hand, .99)['dorsal_visible'])
    row = np.eye(len(config.class_names))[config.class_to_idx['open_palm']]
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config = config
    engine._lock = threading.Lock()
    engine.resolver = GeometryResolver(config)
    engine.detector = SimpleNamespace(
        ready=True, detect=lambda image: points, world_landmarks=lambda: world_for_image(points),
        diagnostics=lambda: {'selected_handedness': hand, 'selected_handedness_confidence': .99})
    engine.model_manager = SimpleNamespace(
        ready=True, models={'ONNX': None}, resolve_name=lambda name: 'ONNX',
        predict_many=lambda feature, names: {'ONNX': (row, 0., {'known_gesture_mass': .05})})
    _, encoded = cv2.imencode('.jpg', np.full((640, 640, 3), 145, dtype=np.uint8))
    session = RuntimeSession(config)
    for now in (1., 1.1, 1.2, 1.3, 1.4, 1.5):
        monkeypatch.setattr(runtime_module.time, 'monotonic', lambda: now)
        result = engine.process_frame(encoded.tobytes(), session)
        if now < 1.45:
            assert result['runtime_action'] == 'Wait / No Action'
    assert result['runtime_prediction'] == 'dorsal'
    assert result['runtime_action'] == config.gesture_to_action['dorsal']
    assert result['known_gesture_mass'] == .05
