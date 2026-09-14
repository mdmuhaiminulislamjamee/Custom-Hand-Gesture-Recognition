import numpy as np
import pytest

import backend.geometry as geometry
from backend.config import RuntimeConfig
from backend.geometry import GeometryResolver, hand_surface_orientation, landmarks_to_feature
from backend.runtime import TemporalGate


def representative_hand():
    # Non-degenerate synthetic 21-point hand geometry for feature-contract tests.
    points = np.zeros((21, 2), dtype=np.float32)
    points[0] = [0.5, 0.9]
    bases = [0.32, 0.43, 0.54, 0.65, 0.76]
    for finger, base_x in enumerate(bases):
        start = 1 if finger == 0 else 5 + (finger - 1) * 4
        count = 4
        for joint in range(count):
            points[start + joint] = [base_x + 0.015 * joint, 0.78 - 0.13 * joint]
    return points


def test_feature_contract_is_76_dimensions():
    feature = landmarks_to_feature(representative_hand())
    assert feature.shape == (76,)
    assert np.isfinite(feature).all()


def test_features_are_translation_and_scale_invariant_except_screen_directions():
    points = representative_hand()
    translated_scaled = points * 1.7 + np.asarray([0.2, -0.1], dtype=np.float32)
    first = landmarks_to_feature(points)
    second = landmarks_to_feature(translated_scaled)
    assert np.allclose(first, second, atol=1e-5)


def test_ncm_horizontal_semantics_are_swapped_without_mirroring(monkeypatch):
    config = RuntimeConfig()
    resolver = GeometryResolver(config)
    probabilities = np.zeros(len(config.class_names), dtype=np.float64)
    probabilities[config.class_to_idx["left"]] = 1.0
    from backend.tests.test_multiview_training import directional_points
    points = directional_points("left")
    positive, positive_details = resolver._directional(probabilities, points)
    points = directional_points("right")
    negative, negative_details = resolver._directional(probabilities, points)

    assert config.class_names[int(np.argmax(positive))] == "left"
    assert config.class_names[int(np.argmax(negative))] == "right"
    assert positive_details["horizontal_mirror"] is False
    assert negative_details["horizontal_mirror"] is False
    assert positive_details["horizontal_semantic_swap"] is True


def side_view_pointing_hand():
    # Foreshortened folded fingers appear straight in 2-D; their tips still
    # end well behind the extended index. This defeated the angle-only gate.
    points = np.zeros((21, 2), dtype=np.float32)
    points[1:5] = [[0.2, -0.2], [0.4, -0.3], [0.6, -0.35], [0.7, -0.3]]
    for mcp, y in ((5, -0.3), (9, 0.0), (13, 0.2), (17, 0.4)):
        step = 0.4 if mcp == 5 else 0.05
        for joint in range(4):
            points[mcp + joint] = [1.0 + step * joint, y]
    return points


@pytest.mark.parametrize("gesture,rotation", [
    ("left", 0), ("right", np.pi),
])
def test_side_view_pointing_reaches_confirmed_action(gesture, rotation):
    config = RuntimeConfig()
    points = side_view_pointing_hand()
    transform = np.asarray([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)],
    ])
    points = (points @ transform.T) * 0.15 + 0.5
    probabilities = np.full(8, 0.01)
    probabilities[config.class_to_idx[gesture]] = 0.93
    resolved, details = GeometryResolver(config).resolve(probabilities, points)
    assert details["directional"]["mean_non_index_extension"] > 0.70
    assert details["directional"]["model_supported_pose"] is True
    assert details["pose_validation"]["valid"] is True
    gate = TemporalGate(config)
    for now in (1.0, 1.1, 1.2):
        decision = gate.update(resolved, now=now, pose_valid=details["pose_validation"]["valid"])
    assert decision.execute is True
    assert decision.predicted_gesture == gesture
    # The new geometry path must never bypass unknown-gesture rejection.
    assert gate.update(resolved, known_gesture_mass=0.1).execute is False


@pytest.mark.parametrize("pose", ["open_palm", "rock", "fist", "diagonal"])
def test_side_view_fallback_rejects_non_pointing_poses(pose):
    config = RuntimeConfig()
    points = side_view_pointing_hand()
    if pose == "open_palm":
        for mcp in (9, 13, 17):
            points[mcp:mcp + 4, 0] = points[5:9, 0]
    elif pose == "rock":
        points[17:21, 0] = points[5:9, 0]
    elif pose == "fist":
        points[5:9, 0] = points[9:13, 0]
    else:
        angle = np.pi / 4
        points = points @ np.asarray([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]]).T
    probabilities = np.zeros(8)
    probabilities[0] = 1.0
    _, details = GeometryResolver(config).resolve(probabilities, points)
    assert details["pose_validation"]["valid"] is False


@pytest.mark.parametrize("probabilities", [
    [0.6, 0.4, 0, 0, 0, 0, 0, 0],
    [0.01, 0.99, 0, 0, 0, 0, 0, 0],
])
def test_side_view_fallback_requires_confident_matching_model(probabilities):
    _, details = GeometryResolver(RuntimeConfig()).resolve(
        np.asarray(probabilities), side_view_pointing_hand()
    )
    assert details["pose_validation"]["valid"] is False


def test_clear_dorsal_survives_classifier_domain_miss_but_requires_longer_hold():
    config = RuntimeConfig()
    points = 1.0 - representative_hand()
    points[:, 0] *= 0.4  # Four fingers held together, as in the camera regression.
    probabilities = np.eye(8)[config.class_to_idx['like']]
    resolved, details = GeometryResolver(config).resolve(probabilities, points)
    pose = details['pose_validation']
    assert pose['gesture'] == 'dorsal'
    assert pose['valid'] and pose['geometry_supported']
    gate = TemporalGate(config)
    for now in (1.0, 1.1, 1.2, 1.3):
        decision = gate.update(resolved, now=now, known_gesture_mass=.01,
                               pose_valid=pose['valid'], geometry_supported=pose['geometry_supported'])
        assert not decision.execute
    decision = gate.update(resolved, now=1.4, known_gesture_mass=.01,
                           pose_valid=True, geometry_supported=True)
    assert decision.execute
    assert decision.predicted_gesture == 'dorsal'
    assert decision.known_gesture_mass == .01  # Never falsify the classifier score.
    assert decision.reason == 'stable Dorsal geometry'
    assert not gate.update(resolved, known_gesture_mass=.01, geometry_supported=False).execute


def test_geometry_support_never_bypasses_invalid_pose_or_other_class_rejection():
    gate = TemporalGate(RuntimeConfig())
    assert not gate.update(np.eye(8)[0], known_gesture_mass=.01, geometry_supported=True).execute
    decision = gate.update(np.eye(8)[6], known_gesture_mass=.01, geometry_supported=True, pose_valid=False)
    assert decision.predicted_gesture == 'no_gesture'


@pytest.mark.parametrize('gesture', ['left', 'right', 'up', 'down'])
def test_strong_direction_geometry_recovers_camera_angle_mass_miss(gesture):
    config = RuntimeConfig()
    gate = TemporalGate(config)
    row = np.eye(8)[config.class_to_idx[gesture]]
    for now in (1.0, 1.1, 1.2, 1.3):
        decision = gate.update(row, now=now, known_gesture_mass=.01,
                               pose_valid=True, geometry_supported=True,
                               directional_recovery=True)
        assert not decision.execute
    decision = gate.update(row, now=1.41, known_gesture_mass=.01,
                           pose_valid=True, geometry_supported=True,
                           directional_recovery=True)
    assert decision.execute
    assert decision.predicted_gesture == gesture
    assert decision.reason == 'stable directional geometry'


def test_geometry_flag_cannot_recover_low_mass_non_directional_pose():
    config = RuntimeConfig()
    gate = TemporalGate(config)
    row = np.eye(8)[config.class_to_idx['like']]
    for now in (1.0, 1.1, 1.2, 1.3, 1.4):
        decision = gate.update(row, now=now, known_gesture_mass=.01,
                               pose_valid=True, geometry_supported=True)
    assert not decision.execute
    assert decision.predicted_gesture == 'no_gesture'


def test_close_finger_palmar_hand_recovers_low_model_mass_after_longer_hold():
    config = RuntimeConfig()
    # Keep the four extended fingers close together. Finger separation is not
    # part of the open-palm command contract.
    dorsal_view = representative_hand()
    for finger, x in zip((5, 9, 13, 17), (0.46, 0.49, 0.52, 0.55)):
        dorsal_view[finger:finger + 4, 0] = x
    # Mirror the same physical right hand to expose its palmar surface.
    palmar_view = dorsal_view.copy()
    palmar_view[:, 0] = 2 * palmar_view[0, 0] - palmar_view[:, 0]
    probabilities = np.full(8, 0.001)
    probabilities[config.class_to_idx['open_palm']] = 0.993

    resolved, details = GeometryResolver(config).resolve(
        probabilities,
        palmar_view,
        handedness='Right',
        handedness_confidence=.99,
    )
    assert details['hand_shape']['hand_surface'] == 'palmar'
    assert details['hand_shape']['palm_geometry_supported'] is True
    assert details['pose_validation']['valid'] is True
    assert details['pose_validation']['geometry_supported'] is True

    gate = TemporalGate(config)
    for now in (1.0, 1.1, 1.2, 1.3, 1.4):
        decision = gate.update(
            resolved,
            now=now,
            known_gesture_mass=.11,
            pose_valid=True,
            geometry_supported=True,
        )
        assert not decision.execute
    decision = gate.update(
        resolved,
        now=1.5,
        known_gesture_mass=.11,
        pose_valid=True,
        geometry_supported=True,
    )
    assert decision.execute
    assert decision.predicted_gesture == 'open_palm'


def test_reverse_side_of_open_hand_is_not_accepted_as_open_palm():
    config = RuntimeConfig()
    dorsal_view = representative_hand()
    probabilities = np.full(8, 0.001)
    probabilities[config.class_to_idx['open_palm']] = 0.993

    surface = hand_surface_orientation(dorsal_view, 'Right', .99)
    _, details = GeometryResolver(config).resolve(
        probabilities,
        dorsal_view,
        handedness='Right',
        handedness_confidence=.99,
    )
    assert surface['surface'] == 'dorsal'
    assert details['hand_shape']['palm_geometry_supported'] is False
    assert details['pose_validation']['valid'] is False
    assert details['pose_validation']['gesture'] == 'open_palm'


def test_short_but_straight_index_finger_is_accepted_as_down():
    config = RuntimeConfig()
    # A real MediaPipe observation whose global index line is straight and
    # dominant, while local joint-angle noise reports only 0.29 extension.
    points = np.asarray([
        [0.488658, 0.738636], [0.527680, 0.811998], [0.524535, 0.920559],
        [0.473730, 0.986109], [0.412335, 0.991226], [0.450721, 0.973023],
        [0.437886, 1.052321], [0.451771, 1.073015], [0.437740, 1.099466],
        [0.403624, 0.940830], [0.386686, 0.994190], [0.438739, 0.928792],
        [0.437037, 0.902793], [0.361884, 0.895542], [0.359641, 0.938635],
        [0.419862, 0.877003], [0.418049, 0.853186], [0.323825, 0.840056],
        [0.333479, 0.881354], [0.385482, 0.841272], [0.389241, 0.819932],
    ], dtype=np.float32)
    probabilities = np.full(8, 0.001)
    probabilities[config.class_to_idx['down']] = 0.993
    resolved, details = GeometryResolver(config).resolve(probabilities, points)
    direction = details['directional']
    assert direction['index_extension'] < .45
    assert direction['short_straight_index'] is True
    assert direction['strong_geometry'] is True
    assert details['pose_validation']['valid'] is True

    gate = TemporalGate(config)
    for now in (1.0, 1.1, 1.2, 1.3):
        assert not gate.update(
            resolved,
            now=now,
            known_gesture_mass=.01,
            pose_valid=True,
            geometry_supported=True,
            directional_recovery=True,
        ).execute
    decision = gate.update(
        resolved,
        now=1.41,
        known_gesture_mass=.01,
        pose_valid=True,
        geometry_supported=True,
        directional_recovery=True,
    )
    assert decision.execute
    assert decision.predicted_gesture == 'down'


def test_steep_edge_on_pointing_pose_is_accepted_as_down():
    config = RuntimeConfig()
    # Regression for the NCM view where folded fingers project as straight
    # chains and the outer hand edge appears beyond the visible index tip.
    points = np.asarray([
        [368, 169], [405, 183], [430, 213], [437, 242], [436, 266],
        [342, 242], [361, 243], [367, 270], [366, 281],
        [383, 241], [391, 269], [391, 282], [390, 286],
        [410, 242], [414, 267], [409, 278], [408, 284],
        [434, 242], [439, 277], [447, 305], [451, 333],
    ], dtype=np.float32)
    # This domain view can be outside the learned feature distribution and may
    # favor another class, so the bounded geometry path must carry the result.
    probabilities = np.full(8, 0.001)
    probabilities[config.class_to_idx['ok']] = 0.993

    resolved, details = GeometryResolver(config).resolve(probabilities, points)
    direction = details['directional']
    assert direction['gesture'] == 'down'
    assert direction['edge_on_down'] is True
    assert direction['strong_geometry'] is True
    assert details['pose_validation']['valid'] is True
    assert details['pose_validation']['geometry_supported'] is True
    assert config.class_names[int(np.argmax(resolved))] == 'down'

    gate = TemporalGate(config)
    for now in (1.0, 1.1, 1.2, 1.3):
        assert not gate.update(
            resolved,
            now=now,
            known_gesture_mass=.001,
            pose_valid=True,
            geometry_supported=True,
            directional_recovery=True,
        ).execute
    decision = gate.update(
        resolved,
        now=1.41,
        known_gesture_mass=.001,
        pose_valid=True,
        geometry_supported=True,
        directional_recovery=True,
    )
    assert decision.execute
    assert decision.predicted_gesture == 'down'


def test_curled_edge_on_pointing_pose_is_accepted_as_down():
    config = RuntimeConfig()
    # NCM regression where the folded index/middle/ring joints curl back in 2-D
    # while every MCP-to-tip vector and the outer hand edge remain downward.
    points = np.asarray([
        [565, 170], [610, 211], [624, 240], [628, 260], [628, 270],
        [497, 255], [497, 324], [507, 316], [515, 297],
        [537, 264], [537, 348], [546, 337], [551, 307],
        [579, 269], [577, 363], [582, 351], [584, 321],
        [629, 269], [633, 355], [637, 410], [638, 449],
    ], dtype=np.float32)
    probabilities = np.full(8, 0.001)
    probabilities[config.class_to_idx['ok']] = 0.993

    resolved, details = GeometryResolver(config).resolve(probabilities, points)
    direction = details['directional']
    assert direction['gesture'] == 'down'
    assert direction['index_extension'] < .10
    assert direction['curled_edge_down'] is True
    assert direction['strong_geometry'] is True
    assert details['pose_validation']['valid'] is True
    assert details['pose_validation']['geometry_supported'] is True
    assert config.class_names[int(np.argmax(resolved))] == 'down'

    gate = TemporalGate(config)
    for now in (1.0, 1.1, 1.2, 1.3):
        assert not gate.update(
            resolved,
            now=now,
            known_gesture_mass=.001,
            pose_valid=True,
            geometry_supported=True,
            directional_recovery=True,
        ).execute
    decision = gate.update(
        resolved,
        now=1.41,
        known_gesture_mass=.001,
        pose_valid=True,
        geometry_supported=True,
        directional_recovery=True,
    )
    assert decision.execute
    assert decision.predicted_gesture == 'down'


@pytest.mark.parametrize('points', [
    np.asarray([
        [493, 199], [510, 220], [520, 240], [522, 260], [520, 278],
        [447, 231], [439, 270], [448, 268], [456, 270],
        [470, 239], [459, 289], [466, 280], [474, 273],
        [497, 245], [477, 306], [484, 293], [492, 274],
        [524, 254], [524, 298], [525, 322], [526, 353],
    ], dtype=np.float32),
    np.asarray([
        [332, 176], [350, 195], [365, 215], [365, 240], [343, 294],
        [280, 215], [270, 266], [278, 264], [288, 247],
        [307, 225], [294, 286], [301, 276], [307, 263],
        [338, 231], [316, 296], [320, 290], [326, 290],
        [369, 241], [375, 290], [380, 320], [384, 348],
    ], dtype=np.float32),
    np.asarray([
        [379, 175], [395, 195], [415, 215], [420, 240], [411, 307],
        [346, 211], [340, 247], [350, 250], [361, 262],
        [374, 222], [356, 264], [368, 268], [376, 270],
        [401, 235], [386, 294], [397, 303], [411, 307],
        [426, 253], [435, 292], [441, 325], [449, 354],
    ], dtype=np.float32),
])
def test_curled_edge_down_camera_variants_are_accepted(points):
    config = RuntimeConfig()
    probabilities = np.full(8, 0.001)
    probabilities[config.class_to_idx['ok']] = 0.993

    resolved, details = GeometryResolver(config).resolve(probabilities, points)

    direction = details['directional']
    assert direction['gesture'] == 'down'
    assert direction['curled_edge_down'] is True
    assert direction['strong_geometry'] is True
    assert details['pose_validation']['valid'] is True
    assert details['pose_validation']['geometry_supported'] is True
    assert config.class_names[int(np.argmax(resolved))] == 'down'


@pytest.mark.parametrize('points', [
    np.asarray([
        [456, 164], [506, 193], [540, 245], [542, 298], [520, 333],
        [386, 228], [385, 299], [400, 299], [408, 278],
        [422, 232], [421, 330], [438, 321], [446, 291],
        [467, 235], [464, 345], [477, 340], [486, 305],
        [517, 237], [519, 333], [512, 383], [510, 420],
    ], dtype=np.float32),
    np.asarray([
        [168, 148], [187, 151], [202, 161], [211, 177], [215, 195],
        [156, 175], [155, 201], [162, 203], [168, 203],
        [167, 175], [166, 205], [175, 205], [180, 204],
        [179, 176], [179, 207], [188, 206], [193, 205],
        [194, 178], [208, 190], [213, 218], [227, 239],
    ], dtype=np.float32),
    np.asarray([
        [304, 212], [320, 219], [331, 244], [331, 276], [318, 303],
        [268, 246], [268, 270], [278, 270], [283, 268],
        [280, 249], [279, 276], [290, 274], [295, 275],
        [292, 251], [292, 284], [303, 281], [309, 282],
        [309, 254], [322, 263], [320, 306], [301, 328],
    ], dtype=np.float32),
    np.asarray([
        [348, 174], [389, 194], [414, 229], [420, 269], [413, 296],
        [311, 237], [316, 295], [321, 295], [328, 274],
        [341, 239], [346, 315], [351, 307], [358, 286],
        [376, 235], [382, 320], [384, 306], [385, 298],
        [414, 229], [420, 269], [427, 334], [431, 365],
    ], dtype=np.float32),
])
def test_clustered_edge_down_camera_variants_are_accepted(points):
    config = RuntimeConfig()
    probabilities = np.full(8, 0.001)
    probabilities[config.class_to_idx['ok']] = 0.993

    resolved, details = GeometryResolver(config).resolve(probabilities, points)

    direction = details['directional']
    assert direction['gesture'] == 'down'
    assert direction['clustered_edge_down'] is True
    assert direction['strong_geometry'] is True
    assert details['pose_validation']['valid'] is True
    assert details['pose_validation']['geometry_supported'] is True
    assert config.class_names[int(np.argmax(resolved))] == 'down'

    gate = TemporalGate(config)
    for now in (1.0, 1.1, 1.2, 1.3):
        assert not gate.update(
            resolved,
            now=now,
            known_gesture_mass=1e-8,
            pose_valid=True,
            geometry_supported=True,
            directional_recovery=True,
        ).execute
    decision = gate.update(
        resolved,
        now=1.41,
        known_gesture_mass=1e-8,
        pose_valid=True,
        geometry_supported=True,
        directional_recovery=True,
    )
    assert decision.execute
    assert decision.predicted_gesture == 'down'


def test_clustered_edge_down_requires_the_lower_outer_tip():
    config = RuntimeConfig()
    points = np.asarray([
        [304, 212], [320, 219], [331, 244], [331, 276], [318, 303],
        [268, 246], [268, 270], [278, 270], [283, 268],
        [280, 249], [279, 276], [290, 274], [295, 275],
        [292, 251], [292, 284], [303, 281], [309, 282],
        [309, 254], [312, 265], [311, 283], [310, 297],
    ], dtype=np.float32)
    probabilities = np.full(8, 0.001)
    probabilities[config.class_to_idx['ok']] = 0.993

    resolved, details = GeometryResolver(config).resolve(probabilities, points)

    assert details['directional']['clustered_edge_down'] is False
    assert config.class_names[int(np.argmax(resolved))] != 'down'


def test_curled_edge_down_fallback_requires_the_full_fingertip_cascade():
    config = RuntimeConfig()
    points = np.asarray([
        [565, 170], [610, 211], [624, 240], [628, 260], [628, 270],
        [497, 255], [497, 324], [507, 316], [515, 297],
        [537, 264], [537, 348], [546, 337], [551, 307],
        [579, 269], [577, 363], [582, 351], [584, 321],
        [629, 269], [633, 355], [637, 410], [638, 449],
    ], dtype=np.float32)
    points[12, 1] = points[8, 1] - 8
    probabilities = np.full(8, 0.001)
    probabilities[config.class_to_idx['ok']] = 0.993

    resolved, details = GeometryResolver(config).resolve(probabilities, points)

    assert details['directional']['curled_edge_down'] is False
    assert config.class_names[int(np.argmax(resolved))] != 'down'


def test_intended_right_pose_tolerates_moderate_camera_perspective():
    config = RuntimeConfig()
    points = side_view_pointing_hand()
    angle = np.deg2rad(138)  # right with a 42-degree upward perspective component
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    points = (points - points[0]) @ rotation.T + points[0]
    probabilities = np.full(8, 0.01)
    probabilities[config.class_to_idx['right']] = 0.93
    resolved, details = GeometryResolver(config).resolve(probabilities, points)
    direction = details['directional']
    assert .73 <= direction['axis_dominance'] < .78
    assert direction['gesture'] == 'right'
    assert direction['valid'] is True
    assert config.class_names[int(np.argmax(resolved))] == 'right'
