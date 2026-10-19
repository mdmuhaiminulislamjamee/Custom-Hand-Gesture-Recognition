from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (DISABLED_GESTURES, DORSAL_REJECTION_REASON,
                     OPEN_HAND_LOWER_ARC_REJECTION_REASON, RuntimeConfig)


MINIMUM_PEACE_TIP_SEPARATION_RATIO = 0.04
FIST_TOWARD_CAMERA_PITCH_DEGREES = 30.0
FIST_MINIMUM_DEPTH_ALIGNMENT = -float(np.sin(np.deg2rad(FIST_TOWARD_CAMERA_PITCH_DEGREES)))
FIST_ORIENTATION_EPSILON = 1e-6
OPEN_HAND_MAX_TILT_FROM_UP_DEGREES = 90.0


def open_hand_xy_orientation(landmarks_xy: np.ndarray, image_shape=None) -> dict:
    """Accept screen-right (0), up (-90), through screen-left (-180).

    Both horizontal endpoints are inclusive, including the equivalent +180
    representation of left. Source dimensions remove normalized-XY stretching.
    """
    points = np.asarray(landmarks_xy, dtype=np.float64).reshape(21, 2)
    axis = points[[5, 9, 13, 17]].mean(axis=0) - points[0]
    if image_shape is not None:
        axis *= np.asarray([image_shape[1], image_shape[0]])
    tilt = float(np.degrees(np.arctan2(axis[0], -axis[1])))
    xy = float(np.degrees(np.arctan2(axis[1], axis[0])))
    if abs(xy - 180.) <= 1e-4:
        xy = -180.
    elif abs(xy) <= 1e-4:
        xy = 0.
    allowed = bool(np.linalg.norm(axis) > 1e-8
                   and abs(tilt) <= OPEN_HAND_MAX_TILT_FROM_UP_DEGREES + 1e-4)
    return {"xy_degrees": xy, "tilt_from_up_degrees": tilt, "allowed_direction": allowed}


def stop_inverted_geometry(landmarks_xy: np.ndarray, image_shape=None) -> dict:
    """Four extended fingers; measure screen tilt from up, independent of hand side.

    Finger separation is deliberately unrestricted. Runtime passes source image
    dimensions so the angle is measured in pixels rather than stretched UV space.
    Surface visibility is checked separately using detector handedness.
    """
    points = np.asarray(landmarks_xy, dtype=np.float64).reshape(21, 2)
    orientation = open_hand_xy_orientation(points, image_shape)
    # Joint angles and relative reaches also need square pixels. Measuring
    # normalized UV makes a sideways little finger appear folded/too short in
    # a wide camera image, despite a straight chain in the displayed overlay.
    if image_shape is not None:
        points = points * np.asarray([image_shape[1], image_shape[0]])
    extensions = np.asarray([
        finger_extension_score(points, *finger)
        for finger in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    ])
    palm_scale = max(float(np.linalg.norm(points[9] - points[0])), 1e-8)
    reaches = np.asarray([np.linalg.norm(points[tip] - points[mcp]) / palm_scale
                          for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20))])
    extended = bool(extensions.min() >= .60 and extensions.mean() >= .80 and reaches.min() >= .40)
    # Independent recovery is deliberately stricter than model-assisted shape
    # validation. It requires four long, straight chains and an open thumb.
    # This covers the reported camera views without turning a curled hand or
    # an arbitrary retired Dorsal prediction into a command.
    strong = bool(
        extended and extensions.min() >= .85 and extensions.mean() >= .92
        and reaches[:3].min() >= .60 and reaches.mean() >= .70
        and finger_extension_score(points, 1, 2, 3, 4) >= .75
        and np.linalg.norm(points[4] - points[8]) / palm_scale >= .50
    )
    return {**orientation,
            "extended_fingers": extended, "valid": extended and orientation["allowed_direction"],
            "minimum_extension": float(extensions.min()),
            "minimum_reach": float(reaches.min()),
            "strong_geometry": strong and orientation["allowed_direction"]}


def stop_inverted_depth_geometry(world_landmarks: np.ndarray | None) -> dict:
    """Veto a clear curl without requiring perfect estimated 3-D extension.

    This corroborates a strongly verified image silhouette. MediaPipe world
    joints are estimates: a short finger/palm ratio or one noisy joint angle
    does not disprove four long, straight image chains. A genuinely folded
    chain has both bent joints and substantially shortened end-to-end reach.
    """
    if world_landmarks is None:
        return {"valid": True, "available": False, "reason": None, "curled_fingers": []}
    world = np.asarray(world_landmarks, dtype=np.float64)
    if world.shape != (21, 3) or not np.isfinite(world).all():
        return {"valid": False, "available": True,
                "reason": "Stop inv: estimated 3D joints are invalid", "curled_fingers": []}
    mcps = (5, 9, 13, 17)
    reaches = np.asarray([np.linalg.norm(world[m + 3] - world[m]) for m in mcps])
    paths = np.asarray([sum(np.linalg.norm(world[j + 1] - world[j])
                            for j in range(m, m + 3)) for m in mcps])
    extensions = np.asarray([finger_extension_score(world, m, m + 1, m + 2, m + 3)
                             for m in mcps])
    straightness = reaches / np.maximum(paths, 1e-8)
    curled = (extensions < .35) & (straightness < .65)
    degenerate = bool(reaches.min() <= 1e-8 or paths.min() <= 1e-8)
    valid = not degenerate and not bool(curled.any())
    reason = ("Stop inv: estimated 3D finger chains are incomplete" if degenerate else
              "Stop inv: estimated 3D joints show a curled finger" if curled.any() else None)
    return {"valid": valid, "available": True, "reason": reason,
            "curled_fingers": [name for name, folded in zip(
                ("index", "middle", "ring", "pinky"), curled) if folded],
            "minimum_extension": float(extensions.min()),
            "minimum_straightness": float(straightness.min())}


def expanded_thumb_pointing_geometry(points):
    """Strict horizontal index, separate thumb and three deeply curled fingers.

    Ordinary directions allow either thumb position. This stronger silhouette
    recovers the reported L-shaped pose without a general low-mass bypass.
    """
    points = np.asarray(points, dtype=np.float32).reshape(21, 2)
    scale = max(_palm_gap_scale(points), 1e-8)
    vector = points[8] - points[5]
    reach = float(np.linalg.norm(vector))
    unit = vector / max(reach, 1e-8)
    path = sum(float(np.linalg.norm(points[b] - points[a])) for a, b in ((5, 6), (6, 7), (7, 8)))
    thumb = points[4] - points[2]
    thumb_path = float(np.linalg.norm(points[3] - points[2]) + np.linalg.norm(points[4] - points[3]))
    thumb_norm = float(np.linalg.norm(thumb))
    other_extensions = [finger_extension_score(points, *ids)
                        for ids in ((9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))]
    other_reaches = [float(np.linalg.norm(points[tip] - points[mcp])) / scale
                     for mcp, tip in ((9, 12), (13, 16), (17, 20))]
    lead = min(float(np.dot(points[8] - points[tip], unit)) / scale for tip in (12, 16, 20))
    return bool(
        abs(float(unit[0])) >= .94
        and finger_extension_score(points, 5, 6, 7, 8) >= .90
        and reach / max(path, 1e-8) >= .96 and reach / scale >= .75 and lead >= .60
        and max(other_extensions) <= .30 and max(other_reaches) <= .50
        and finger_extension_score(points, 1, 2, 3, 4) >= .85
        and thumb_norm / max(thumb_path, 1e-8) >= .90
        and float(np.linalg.norm(points[4] - points[0])) / scale >= 1.35
        and abs(float(np.dot(thumb / max(thumb_norm, 1e-8), unit))) <= .65
        and float(np.linalg.norm(points[4] - points[8])) / scale >= .75
    )


def arched_thumb_pointing_geometry(points):
    """Recover a long horizontal index beside an arched, separated thumb.

    In the NCM side view the thumb's two joint angles can look bent while its
    tip still lies well beyond the wrist and above the pointing index. Require
    all three other fingertips on the opposite side of that index and well
    behind it before allowing low-mass directional recovery.
    """
    points = np.asarray(points, dtype=np.float32).reshape(21, 2)
    scale = max(_palm_gap_scale(points), 1e-8)
    index = points[8] - points[5]
    reach = float(np.linalg.norm(index))
    unit = index / max(reach, 1e-8)
    index_path = sum(float(np.linalg.norm(points[b] - points[a]))
                     for a, b in ((5, 6), (6, 7), (7, 8)))
    other_extensions = [finger_extension_score(points, m, m + 1, m + 2, m + 3)
                        for m in (9, 13, 17)]
    other_tips = points[[12, 16, 20]]
    other_reaches = np.linalg.norm(other_tips - points[[9, 13, 17]], axis=1) / scale
    lead = float(np.min((points[8] - other_tips) @ unit) / scale)
    thumb = points[4] - points[2]
    thumb_norm = float(np.linalg.norm(thumb))
    thumb_path = (float(np.linalg.norm(points[3] - points[2]))
                  + float(np.linalg.norm(points[4] - points[3])))
    # Signed 2-D cross products keep this test invariant to horizontal
    # mirroring and to whether the thumb is above or below the index.
    def side(point):
        offset = point - points[5]
        return float((unit[0] * offset[1] - unit[1] * offset[0]) / scale)
    thumb_base_side = side(points[2])
    thumb_sign = 1.0 if thumb_base_side >= 0 else -1.0
    thumb_extension = finger_extension_score(points, 1, 2, 3, 4)
    thumb_straightness = thumb_norm / max(thumb_path, 1e-8)
    thumb_reach = float(np.linalg.norm(points[4] - points[0])) / scale
    thumb_alignment = abs(float(np.dot(thumb / max(thumb_norm, 1e-8), unit)))
    thumb_index_gap = float(np.linalg.norm(points[4] - points[8])) / scale
    long_arch = bool(
        reach / scale >= 1.20 and thumb_extension >= .30
        and thumb_straightness >= .88 and thumb_reach >= 1.30
        and thumb_alignment <= .87 and thumb_index_gap >= .80
        and abs(thumb_base_side) >= .40
        and all(thumb_sign * side(tip) <= -.20 for tip in other_tips))
    # A close, oblique view makes the arched thumb look locally bent and the
    # index shorter relative to the palm. Require three much shorter curled
    # fingers and a separate thumb on the other side of the index instead of
    # demanding an almost straight thumb. This also handles rectangular images.
    compact_arch = bool(
        reach / scale >= .95 and float(other_reaches.max()) <= .60
        and thumb_extension >= .10 and thumb_straightness >= .78
        and thumb_reach >= 1.20 and thumb_alignment <= .95
        and thumb_index_gap >= .75 and abs(thumb_base_side) >= .30
        and all(thumb_sign * side(tip) <= -.15 for tip in other_tips))
    # A side view foreshortens the index and aligns the arched thumb with it.
    # The folded pinky can also look locally straight. Establish its curl by
    # requiring EVERY other tip to return behind its PIP along the index axis,
    # remain near its own MCP, and stay across the index from the thumb.
    # A raised middle/ring/pinky cannot satisfy these return-path checks.
    side_view_arch = bool(
        reach / scale >= .70 and float(other_reaches.max()) <= .70
        and max(float(np.dot(points[m + 3] - points[m], unit)) / scale
                for m in (9, 13, 17)) <= .18
        and all(float(np.dot(points[m + 3] - points[m + 1], unit)) / scale <= -.04
                for m in (9, 13, 17))
        and thumb_extension >= .10 and thumb_straightness >= .75
        and thumb_reach >= 1.20 and thumb_alignment <= .99
        and thumb_index_gap >= .60 and abs(thumb_base_side) >= .18
        and thumb_sign * side(points[4]) >= .10
        and all(thumb_sign * side(tip) <= -.15 for tip in other_tips))
    return bool(
        abs(float(unit[0])) >= .94
        and finger_extension_score(points, 5, 6, 7, 8) >= .90
        and reach / max(index_path, 1e-8) >= .97
        and (long_arch or compact_arch or side_view_arch) and lead >= .70
        and (max(other_extensions) <= .40 or side_view_arch)
        and float(other_reaches.max()) <= .90
        and thumb_sign * side(points[4]) >= .06
    )


def horizontal_pointing_geometry(landmarks_xy: np.ndarray, image_shape) -> dict:
    """One long horizontal index with three folded/converging finger chains.

    Thumb position is unrestricted. Source pixels keep joint angles and
    shortening comparable across camera aspect ratios. Two narrow edge views
    cover projected folds whose local joint angles alone look straight.
    """
    points = np.asarray(landmarks_xy, dtype=np.float64).reshape(21, 2)
    points = (points - points[0]) * [image_shape[1], image_shape[0]]
    scale = max(float(np.linalg.norm(points[9] - points[0])), 1e-8)
    vector = points[8] - points[5]
    reach = float(np.linalg.norm(vector))
    unit = vector / max(reach, 1e-8)
    normal = np.asarray([-unit[1], unit[0]])
    path = sum(float(np.linalg.norm(points[j + 1] - points[j])) for j in (5, 6, 7))
    mcps = np.asarray([9, 13, 17])
    tips = mcps + 3
    spans = points[tips] - points[mcps]
    extensions = np.asarray([finger_extension_score(points, m, m + 1, m + 2, m + 3)
                             for m in mcps])
    reaches = np.linalg.norm(spans, axis=1) / scale
    returns = (points[tips] - points[mcps + 1]) @ unit / scale
    lead = float(np.min((points[8] - points[tips]) @ unit) / scale)
    folded = bool(reaches.max() <= .75 and extensions.max() <= .55
                  and returns.max() <= -.04 and lead >= .55)
    base_sides = (points[mcps] - points[5]) @ normal / scale
    tip_sides = (points[tips] - points[5]) @ normal / scale
    convergence = np.abs(base_sides) - np.abs(tip_sides)
    converging = bool(
        np.all(base_sides > .15) or np.all(base_sides < -.15))
    converging = bool(converging and convergence.min() >= .18
                      and np.ptp(tip_sides) <= .60
                      and reaches.max() * scale <= .90 * reach and lead >= .25)
    knuckle_span = max(float(np.linalg.norm(points[a] - points[b]))
                       for a in (5, 9, 13, 17) for b in (5, 9, 13, 17)) / scale
    collapsed = bool(knuckle_span <= .20 and np.count_nonzero(extensions <= .45) >= 2
                     and reaches.max() * scale <= .90 * reach
                     and lead >= .20)
    valid = bool(
        abs(float(unit[0])) >= .94 and reach / scale >= .70
        and finger_extension_score(points, 5, 6, 7, 8) >= .90
        and reach / max(path, 1e-8) >= .97
        and (folded or converging or collapsed)
    )
    # A separate thumb, long index, and three short returning chains provide
    # independent image evidence when estimated depth foreshortens the index.
    # Keep this stricter than the ordinary side-view silhouette.
    clear_single_index = bool(
        valid and folded and reach / scale >= .80 and lead >= .75
        and reach / scale >= .70 * knuckle_span
        and reaches.max() <= .65
        and np.linalg.norm(points[4] - points[0]) / scale >= .90
        and np.linalg.norm(points[4] - points[8]) / scale >= .50
    )
    return {"valid": valid, "folded_fingers": folded,
            "clear_single_index": clear_single_index,
            "converging_fingers": converging, "collapsed_knuckles": collapsed,
            "index_reach_ratio": reach / scale, "index_lead_ratio": lead,
            "knuckle_span_ratio": knuckle_span}


def _unit_direction(vector: np.ndarray, description: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError(f"Degenerate {description} direction.")
    return np.asarray(vector, dtype=np.float32) / norm


def _palm_gap_scale(points: np.ndarray) -> float:
    references = np.asarray([
        np.linalg.norm(points[5] - points[0]),
        np.linalg.norm(points[9] - points[0]),
        np.linalg.norm(points[17] - points[0]),
        np.linalg.norm(points[17] - points[5]),
    ], dtype=np.float32)
    positive = references[references > 1e-8]
    if not len(positive):
        raise ValueError("Degenerate palm scale for fingertip gap.")
    return float(np.median(positive))


def thumb_index_gap_ratio(landmarks_xy: np.ndarray) -> float:
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)
    return float(np.linalg.norm(points[4] - points[8]) / _palm_gap_scale(points))


def joint_angle_degrees(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    first = np.asarray(point_a, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)
    second = np.asarray(point_c, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def finger_extension_score(points: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> float:
    mean_angle = 0.5 * (
        joint_angle_degrees(points[mcp], points[pip], points[dip])
        + joint_angle_degrees(points[pip], points[dip], points[tip])
    )
    return float(np.clip((mean_angle - 110.0) / 60.0, 0.0, 1.0))


def peace_finger_geometry(landmarks: np.ndarray) -> dict[str, float | bool]:
    """Check both raised fingers independently, including their actual reach.

    Joint angles alone can call a short, folded index finger straight when the
    camera sees its joints edge-on. The middle finger by itself is not Peace.
    A small nonzero tip gap rejects a collapsed duplicate track without
    requiring the index and middle fingers to form a wide V. The same check
    works with 2-D or MediaPipe's 3-D world landmarks.
    """
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape not in {(21, 2), (21, 3)} or not np.isfinite(points).all():
        return {"valid": False, "strong_geometry": False, "index_extension": 0.0,
                "middle_extension": 0.0, "index_reach_ratio": 0.0,
                "index_forward_ratio": 0.0, "tip_separation_ratio": 0.0}
    index = points[8] - points[5]
    middle = points[12] - points[9]
    middle_reach = float(np.linalg.norm(middle))
    index_reach = float(np.linalg.norm(index))
    middle_unit = middle / max(middle_reach, 1e-8)
    index_reach_ratio = index_reach / max(middle_reach, 1e-8)
    index_forward_ratio = float(np.dot(index, middle_unit) / max(middle_reach, 1e-8))
    palm_scale = max(float(np.linalg.norm(points[9] - points[0])), 1e-8)
    tip_separation_ratio = float(np.linalg.norm(points[8] - points[12]) / palm_scale)
    index_extension = finger_extension_score(points, 5, 6, 7, 8)
    middle_extension = finger_extension_score(points, 9, 10, 11, 12)
    ring_extension = finger_extension_score(points, 13, 14, 15, 16)
    pinky_extension = finger_extension_score(points, 17, 18, 19, 20)
    near_parallel_pair = tip_separation_ratio <= .35
    # A folded finger can have almost straight projected joints when viewed
    # edge-on. Its fingertip still falls well short of the raised middle tip.
    # Only allow this ambiguity for a close projected V; wider hard negatives
    # can have the same short ring reach.
    ring_reach_ratio = float(np.linalg.norm(points[16] - points[13]) / max(middle_reach, 1e-8))
    pinky_reach_ratio = float(np.linalg.norm(points[20] - points[17]) / max(middle_reach, 1e-8))
    raised_other_finger = bool(
        (ring_extension > .65 and (ring_reach_ratio >= .65 or not near_parallel_pair))
        or (pinky_extension > .65 and (pinky_reach_ratio >= .65 or not near_parallel_pair))
    )
    # In the reported side views curled ring/little fingers reach sideways,
    # sometimes with straight projected joints. Measure their forward reach
    # along the two raised fingers rather than their sideways displacement.
    pair_axis = index / max(index_reach, 1e-8) + middle_unit
    pair_axis /= max(float(np.linalg.norm(pair_axis)), 1e-8)
    tip_forward = (points[[8, 12, 16, 20]] - points[0]) @ pair_axis / palm_scale
    pair_lead = float(tip_forward[:2].min() - tip_forward[2:].max())
    other_forward = (points[[16, 20]] - points[[13, 17]]) @ pair_axis / palm_scale
    strong_geometry = bool(
        index_extension >= .80 and middle_extension >= .80
        and .70 <= index_reach_ratio <= 1.35
        and index_forward_ratio >= .65
        and min(index_reach, middle_reach) / palm_scale >= .80
        and .015 <= tip_separation_ratio <= .55
        and float(np.linalg.norm(points[5] - points[9])) / palm_scale >= .05
        # Distinct PIP tracks distinguish two close fingers from a duplicated
        # index track, even when their tips nearly overlap.
        and float(np.linalg.norm(points[6] - points[10])) / palm_scale >= .05
        and pair_lead >= .65 and float(other_forward.max()) <= .60
    )
    valid = bool(
        index_extension >= .55
        and middle_extension >= .45
        and not raised_other_finger
        and .62 <= index_reach_ratio <= 1.45
        and index_forward_ratio >= .42
        and tip_separation_ratio >= MINIMUM_PEACE_TIP_SEPARATION_RATIO
    )
    return {
        "valid": valid or strong_geometry,
        "strong_geometry": strong_geometry,
        "pair_lead_ratio": pair_lead,
        "index_extension": index_extension,
        "middle_extension": middle_extension,
        "index_reach_ratio": index_reach_ratio,
        "index_forward_ratio": index_forward_ratio,
        "tip_separation_ratio": tip_separation_ratio,
        "ring_reach_ratio": ring_reach_ratio,
        "pinky_reach_ratio": pinky_reach_ratio,
    }


def landmarks_to_feature(landmarks_xy: np.ndarray) -> np.ndarray:
    """Exact 76-D geometry used by the supplied v18 training notebook."""
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2).copy()
    raw_points = points.copy()
    pinch_gap_ratio = thumb_index_gap_ratio(points)

    thumb_extension = finger_extension_score(raw_points, 1, 2, 3, 4)
    finger_extensions = np.asarray([
        finger_extension_score(raw_points, 5, 6, 7, 8),
        finger_extension_score(raw_points, 9, 10, 11, 12),
        finger_extension_score(raw_points, 13, 14, 15, 16),
        finger_extension_score(raw_points, 17, 18, 19, 20),
    ], dtype=np.float32)
    other_fingers_folded = float(1.0 - finger_extensions[1:].mean())
    palm_references = np.asarray([
        np.linalg.norm(raw_points[5] - raw_points[0]),
        np.linalg.norm(raw_points[9] - raw_points[0]),
        np.linalg.norm(raw_points[17] - raw_points[0]),
        np.linalg.norm(raw_points[17] - raw_points[5]),
    ], dtype=np.float32)
    positive_references = palm_references[palm_references > 1e-8]
    if not len(positive_references):
        raise ValueError("Degenerate palm scale for engineered hand shape.")
    palm_scale = float(np.median(positive_references))
    shape_features = np.asarray([
        thumb_extension,
        *finger_extensions,
        other_fingers_folded,
        float(np.linalg.norm(raw_points[4] - raw_points[5]) / palm_scale),
        float(np.linalg.norm(raw_points[4] - raw_points[0]) / palm_scale),
    ], dtype=np.float32)

    points -= points[0]
    orientation_features = np.concatenate([
        _unit_direction(points[8] - points[5], "index MCP-to-tip"),
        _unit_direction(points[8], "wrist-to-index-tip"),
    ])
    middle_mcp = points[9]
    if np.linalg.norm(middle_mcp) < 1e-8:
        raise ValueError("Degenerate wrist-to-middle-finger geometry.")
    angle = -np.pi / 2 - np.arctan2(middle_mcp[1], middle_mcp[0])
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    points = points @ rotation.T
    scale = float(np.linalg.norm(points, axis=1).max())
    if scale < 1e-8:
        raise ValueError("Degenerate landmark scale.")
    points /= scale
    if points[5, 0] < points[17, 0]:
        points[:, 0] *= -1
    radial_distances = np.linalg.norm(points, axis=1)
    feature = np.concatenate([
        points.reshape(-1),
        radial_distances,
        orientation_features,
        np.asarray([pinch_gap_ratio], dtype=np.float32),
        shape_features,
    ]).astype(np.float32)
    if feature.shape != (76,):
        raise AssertionError(f"Expected 76 features, produced {feature.shape}.")
    return feature


def single_index_pose_score(landmarks_xy: np.ndarray) -> float:
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)
    index_extension = finger_extension_score(points, 5, 6, 7, 8)
    other_extensions = np.asarray([
        finger_extension_score(points, 9, 10, 11, 12),
        finger_extension_score(points, 13, 14, 15, 16),
        finger_extension_score(points, 17, 18, 19, 20),
    ])
    other_fingers_folded = float(1.0 - other_extensions.mean())
    palm_scale = max(float(np.linalg.norm(points[9] - points[0])), 1e-6)
    index_reach = float(np.linalg.norm(points[8] - points[0]))
    other_tip_reach = float(max(
        np.linalg.norm(points[12] - points[0]),
        np.linalg.norm(points[16] - points[0]),
        np.linalg.norm(points[20] - points[0]),
    ))
    prominence = float(np.clip(
        ((index_reach - other_tip_reach) / palm_scale + 0.10) / 0.70,
        0.0, 1.0,
    ))
    return float(np.clip(
        0.50 * index_extension + 0.35 * other_fingers_folded + 0.15 * prominence,
        0.0, 1.0,
    ))


def return_main_pose_geometry(landmarks_xy: np.ndarray) -> dict[str, float | int | bool]:
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)
    fingers = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    extensions = np.asarray([
        finger_extension_score(points, *finger) for finger in fingers
    ], dtype=np.float32)
    vectors = np.asarray([points[tip] - points[mcp] for mcp, _, _, tip in fingers])
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms < 1e-8):
        return {"score": 0.0, "strong_geometry": False, "downward_finger_count": 0}
    units = vectors / norms[:, None]
    downward_count = int(((units[:, 1] >= 0.50) & (extensions >= 0.50)).sum())
    mean_direction = units.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_direction))
    if mean_norm < 1e-8:
        downward_score = parallel_score = 0.0
    else:
        mean_direction /= mean_norm
        downward_score = float(np.clip((mean_direction[1] - 0.30) / 0.60, 0.0, 1.0))
        parallel_score = float(np.clip((mean_norm - 0.70) / 0.30, 0.0, 1.0))
    palm_scale = max(float(np.linalg.norm(points[9] - points[0])), 1e-6)
    tips = (8, 12, 16, 20)
    mean_gap = float(np.mean([
        np.linalg.norm(points[right] - points[left]) / palm_scale
        for left, right in zip(tips[:-1], tips[1:])
    ]))
    together = float(np.clip((0.78 - mean_gap) / 0.52, 0.0, 1.0))
    extended = float(extensions.mean())
    minimum = float(extensions.min())
    base = (
        0.48 * (0.70 * extended + 0.30 * minimum)
        + 0.27 * downward_score
        + 0.15 * together
        + 0.10 * parallel_score
    )
    score = float(np.clip(base * (0.75 + 0.25 * together), 0.0, 1.0))
    strong = bool(
        score >= 0.76 and downward_count >= 4 and minimum >= 0.50
    )
    return {
        "score": score,
        "strong_geometry": strong,
        "downward_finger_count": downward_count,
        "extended_score": extended,
        "minimum_extension_score": minimum,
        "downward_score": downward_score,
        "together_score": together,
        "parallel_score": parallel_score,
    }


def zoom_pose_geometry(landmarks_xy: np.ndarray) -> dict[str, float | bool]:
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)
    thumb_extension = finger_extension_score(points, 1, 2, 3, 4)
    index_extension = finger_extension_score(points, 5, 6, 7, 8)
    others = np.asarray([
        finger_extension_score(points, 9, 10, 11, 12),
        finger_extension_score(points, 13, 14, 15, 16),
        finger_extension_score(points, 17, 18, 19, 20),
    ])
    folded = float(1.0 - others.mean())
    references = np.asarray([
        np.linalg.norm(points[5] - points[0]), np.linalg.norm(points[9] - points[0]),
        np.linalg.norm(points[17] - points[0]), np.linalg.norm(points[17] - points[5]),
    ])
    positive = references[references > 1e-8]
    if not len(positive):
        return {"score": 0.0, "strong_geometry": False, "thumb_opposition_score": 0.0, "pair_reach_score": 0.0}
    scale = float(np.median(positive))
    thumb_reach = float(np.linalg.norm(points[4] - points[0]) / scale)
    index_reach = float(np.linalg.norm(points[8] - points[0]) / scale)
    thumb_mcp_ratio = float(np.linalg.norm(points[4] - points[5]) / scale)
    other_reaches = np.asarray([
        np.linalg.norm(points[12] - points[0]) / scale,
        np.linalg.norm(points[16] - points[0]) / scale,
        np.linalg.norm(points[20] - points[0]) / scale,
    ])
    thumb_reach_score = float(np.clip((thumb_reach - 0.65) / 0.75, 0.0, 1.0))
    index_reach_score = float(np.clip((index_reach - 0.80) / 0.90, 0.0, 1.0))
    pair_reach = float(np.clip((min(thumb_reach, index_reach) - 0.75) / 0.70, 0.0, 1.0))
    prominence = 0.5 * (thumb_reach + index_reach) - float(other_reaches.mean())
    prominence_score = float(np.clip((prominence + 0.05) / 0.55, 0.0, 1.0))
    opposition = float(np.clip((thumb_mcp_ratio - 0.38) / 0.72, 0.0, 1.0))
    score = float(np.clip(
        0.27 * folded + 0.10 * thumb_extension + 0.11 * index_extension
        + 0.12 * thumb_reach_score + 0.11 * index_reach_score
        + 0.10 * pair_reach + 0.08 * prominence_score + 0.11 * opposition,
        0.0, 1.0,
    ))
    strong = bool(
        score >= 0.60 and folded >= 0.35 and thumb_reach_score >= 0.25
        and index_reach_score >= 0.20 and pair_reach >= 0.15
        and prominence_score >= 0.08 and opposition >= 0.25
    )
    return {
        "score": score,
        "strong_geometry": strong,
        "thumb_extension": thumb_extension,
        "index_extension": index_extension,
        "other_fingers_folded": folded,
        "thumb_reach_score": thumb_reach_score,
        "index_reach_score": index_reach_score,
        "pair_reach_score": pair_reach,
        "pair_prominence_score": prominence_score,
        "thumb_opposition_score": opposition,
        "thumb_tip_index_mcp_ratio": thumb_mcp_ratio,
    }


def foreshortened_fist_geometry(
    landmarks_xy: np.ndarray,
) -> dict[str, float | int | bool]:
    """Recognize a camera-facing fist whose curl is hidden by perspective.

    In this view the PIP/DIP angle score can look partly extended even though
    every fingertip has folded back toward the palm.  Projecting MCPs and tips
    onto the wrist-to-palm axis is invariant to in-plane rotation and mirroring,
    while the compact-thumb checks keep Like and Thumb Down out of this path.
    """
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)
    wrist = points[0]
    mcps = points[[5, 9, 13, 17]]
    tips = points[[8, 12, 16, 20]]
    palm_center = mcps.mean(axis=0)
    palm_axis = palm_center - wrist
    palm_scale = float(np.linalg.norm(palm_axis))
    if palm_scale < 1e-8:
        return {
            "strong_geometry": False,
            "retracted_finger_count": 0,
            "mean_curlback_ratio": 0.0,
            "maximum_tip_radius_ratio": float("inf"),
            "thumb_radius_ratio": float("inf"),
        }
    palm_unit = palm_axis / palm_scale
    mcp_projection = (mcps - wrist) @ palm_unit
    tip_projection = (tips - wrist) @ palm_unit
    curlback = (mcp_projection - tip_projection) / palm_scale
    tip_radii = np.linalg.norm(tips - palm_center, axis=1) / palm_scale
    thumb_radius = float(np.linalg.norm(points[4] - palm_center) / palm_scale)
    retracted_count = int((curlback >= 0.05).sum())
    mean_curlback = float(curlback.mean())
    maximum_tip_radius = float(tip_radii.max())
    strong = bool(
        retracted_count == 4
        and mean_curlback >= 0.16
        and maximum_tip_radius <= 0.85
        and thumb_radius <= 0.85
    )
    return {
        "strong_geometry": strong,
        "retracted_finger_count": retracted_count,
        "mean_curlback_ratio": mean_curlback,
        "maximum_tip_radius_ratio": maximum_tip_radius,
        "thumb_radius_ratio": thumb_radius,
    }


def _mirrored_view_alignment(axis: np.ndarray, xy_degrees: float, zx_degrees: float) -> float:
    """Compare a unit palm axis with a reported view or its horizontal mirror.

    XY and ZX determine the 3-D direction; YZ is a redundant display angle.
    These references describe camera views, not evidence of finger closure.
    """
    xy, zx = np.deg2rad([xy_degrees, zx_degrees])
    reference = np.asarray([abs(np.cos(xy)), np.sin(xy), np.cos(xy) / np.tan(zx)])
    reference /= np.linalg.norm(reference)
    return float(np.dot([abs(axis[0]), axis[1], axis[2]], reference))


def dorsal_geometry_support(geometry: dict) -> bool:
    """Independent evidence for an unmistakable four-finger downward command.

    More restrictive than the ordinary model-assisted dorsal validator. This
    handles a camera-domain miss without weakening rejection for other shapes.
    """
    return bool(
        geometry.get("score", 0) >= 0.90
        and geometry.get("downward_finger_count", 0) == 4
        and geometry.get("minimum_extension_score", 0) >= 0.80
        and geometry.get("downward_score", 0) >= 0.90
        and geometry.get("together_score", 0) >= 0.58
    )


def hand_surface_orientation(
    landmarks_xy: np.ndarray,
    handedness: str | None,
    handedness_confidence: float | None = None,
    *,
    minimum_handedness_confidence: float = 0.75,
    surface_winding_margin: float = 0.20,
) -> dict[str, float | bool | str | None]:
    """Estimate whether an unmirrored NCM frame shows palm or dorsal skin.

    The 2-D landmarks alone are ambiguous: a right dorsal hand has the same
    winding as a left palmar hand. MediaPipe handedness resolves that ambiguity.
    On the unmirrored NCM feed, dorsal right hands have a positive wrist/index/
    pinky winding and dorsal left hands have a negative winding. The opposite
    sign exposes the palm. Near-edge-on hands are deliberately left unknown.
    """
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)
    hand = str(handedness or "").strip().lower()
    confidence = 0.0 if handedness_confidence is None else float(handedness_confidence)
    index_palm = points[5] - points[0]
    pinky_palm = points[17] - points[0]
    denominator = float(np.linalg.norm(index_palm) * np.linalg.norm(pinky_palm))
    if (
        hand not in {"left", "right"}
        or not np.isfinite(confidence)
        or confidence < minimum_handedness_confidence
        or denominator < 1e-8
    ):
        return {
            "surface": None,
            "palm_visible": False,
            "dorsal_visible": False,
            "surface_score": 0.0,
            "handedness": hand or None,
            "handedness_confidence": confidence,
        }
    winding = float(
        (index_palm[0] * pinky_palm[1] - index_palm[1] * pinky_palm[0])
        / denominator
    )
    # Positive means dorsal for a right hand and palmar for a left hand.
    dorsal_score = winding * (1.0 if hand == "right" else -1.0)
    surface = (
        "dorsal"
        if dorsal_score >= surface_winding_margin
        else "palmar"
        if dorsal_score <= -surface_winding_margin
        else None
    )
    return {
        "surface": surface,
        "palm_visible": surface == "palmar",
        "dorsal_visible": surface == "dorsal",
        "surface_score": dorsal_score,
        "handedness": hand,
        "handedness_confidence": confidence,
    }


def _set_probability_floor(probabilities: np.ndarray, target_index: int, floor: float) -> np.ndarray:
    adjusted = np.asarray(probabilities, dtype=np.float64).copy()
    if adjusted[target_index] < floor:
        other_total = float(adjusted.sum() - adjusted[target_index])
        if other_total > 1e-12:
            adjusted *= (1.0 - floor) / other_total
        else:
            adjusted.fill((1.0 - floor) / max(1, len(adjusted) - 1))
        adjusted[target_index] = floor
    adjusted /= max(float(adjusted.sum()), 1e-12)
    return adjusted


@dataclass(slots=True)
class GeometryResolver:
    config: RuntimeConfig

    def _directional(
        self, probabilities: np.ndarray, landmarks: np.ndarray,
        world_landmarks: np.ndarray | None = None,
        preferred_direction: str | None = None,
        image_shape=None,
    ) -> tuple[np.ndarray, dict | None]:
        points = np.asarray(landmarks, dtype=np.float32).reshape(21, 2)
        raw = self.config.class_names[int(np.argmax(probabilities))]
        world_index_only = False
        world_folded_pointing = False
        world_pointing_contradiction = world_landmarks is not None
        world_projection_fraction = 0.0
        if world_landmarks is not None:
            world = np.asarray(world_landmarks, dtype=np.float32)
            if world.shape == (21, 3) and np.isfinite(world).all():
                world_vector = world[8] - world[5]
                world_reach = float(np.linalg.norm(world_vector))
                world_projection_fraction = float(np.linalg.norm(world_vector[:2])) / max(world_reach, 1e-8)
                world_palm = max(float(np.linalg.norm(world[9] - world[0])), 1e-8)
                world_path = sum(float(np.linalg.norm(world[b] - world[a]))
                                 for a, b in ((5, 6), (6, 7), (7, 8)))
                world_index_extension = finger_extension_score(world, 5, 6, 7, 8)
                world_index_straightness = world_reach / max(world_path, 1e-8)
                other_world_extensions = [finger_extension_score(world, m, m + 1, m + 2, m + 3)
                                          for m in (9, 13, 17)]
                other_world_reaches = [float(np.linalg.norm(world[m + 3] - world[m])) / world_palm
                                       for m in (9, 13, 17)]
                world_index_only = bool(
                    world_index_extension >= .75
                    and world_index_straightness >= .88
                    and world_reach / world_palm >= .55
                    and max(other_world_extensions) <= .55
                    and max(other_world_reaches) <= .65
                )
                # An oblique wrist estimate can shorten the world palm while
                # the visible index and three curled chains remain clear.
                # For the strict pixel silhouette below, compare the folded
                # chains with the index itself instead of that palm estimate.
                world_unit = world_vector / max(world_reach, 1e-8)
                world_folded_pointing = bool(
                    world_index_extension >= .75
                    and world_index_straightness >= .88
                    and all(
                        other_world_extensions[i] <= .55
                        and np.linalg.norm(world[m + 3] - world[m]) <= .70 * world_reach
                        and np.dot(world[m + 3] - world[m + 1], world_unit) <= -.04 * world_reach
                        for i, m in enumerate((9, 13, 17))
                    )
                )
                # A near-head-on world index or a noisy joint estimate is not
                # proof against a long, straight index in the camera image.
                # Veto the stricter image recovery only for a genuinely curled
                # index or a long, straight second finger in world space.
                world_pointing_contradiction = bool(
                    world_reach <= 1e-8
                    or (world_index_extension < .35 and world_index_straightness < .65)
                    or any(
                        other_world_extensions[i] >= .75
                        and np.linalg.norm(world[m + 3] - world[m]) >= .60 * world_reach
                        and np.linalg.norm(world[m + 3] - world[m]) / max(sum(
                            float(np.linalg.norm(world[j + 1] - world[j]))
                            for j in range(m, m + 3)), 1e-8) >= .90
                        for i, m in enumerate((9, 13, 17))
                    )
                )
        arched_thumb_pointing = arched_thumb_pointing_geometry(points)
        expanded_thumb_pointing = bool(
            expanded_thumb_pointing_geometry(points) or arched_thumb_pointing
        )
        horizontal_shape = (horizontal_pointing_geometry(points, image_shape)
                            if image_shape is not None else {"valid": False})
        horizontal_pointing = bool(
            horizontal_shape["valid"] and (
                world_landmarks is None or world_index_only
                or (horizontal_shape["folded_fingers"] and world_folded_pointing)
                or (horizontal_shape["clear_single_index"] and not world_pointing_contradiction))
        )
        continued_direction = bool(
            world_index_only and world_projection_fraction < .20
            and preferred_direction in {"left", "right", "up", "down"}
        )
        # Peace and Rock also extend the index finger. The one-finger
        # directional recovery below must not rewrite a confidently classified
        # multi-finger command into Left/Right/Up/Down.
        if raw in {"peace", "rock"} and not (expanded_thumb_pointing or world_index_only or horizontal_pointing):
            return probabilities, {
                "valid": True,
                "reason": "multi-finger command retains model label",
                "strong_geometry": False,
                "model_supported_pose": False,
            }
        pose_score = single_index_pose_score(points)
        vector = points[8] - points[5]
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            if continued_direction:
                return _set_probability_floor(
                    probabilities, self.config.class_to_idx[preferred_direction], .98
                ), {
                    "valid": True, "gesture": preferred_direction,
                    "reason": "continuing the selected hand's last visible direction",
                    "strong_geometry": True, "model_supported_pose": True,
                    "world_index_only": True, "continued_direction": True,
                    "world_projection_fraction": world_projection_fraction,
                }
            return probabilities, {
                "valid": raw not in {"left", "right", "up", "down"},
                "reason": "degenerate index-finger direction",
            }
        non_index_extensions = np.asarray([
            finger_extension_score(points, 9, 10, 11, 12),
            finger_extension_score(points, 13, 14, 15, 16),
            finger_extension_score(points, 17, 18, 19, 20),
        ], dtype=np.float32)
        if (raw in {"left", "right", "up", "down"}
                and "peace" in self.config.class_to_idx and not world_index_only):
            index_extension = finger_extension_score(points, 5, 6, 7, 8)
            index_span = float(np.linalg.norm(points[8] - points[5]))
            middle_span = float(np.linalg.norm(points[12] - points[9]))
            pinky_span = float(np.linalg.norm(points[20] - points[17]))
            peace_shape = bool(
                index_extension >= .65
                and non_index_extensions[0] >= .65
                and float(non_index_extensions[1:].max()) <= .55
                and middle_span >= .60 * index_span
            )
            rock_shape = bool(
                index_extension >= .65
                and non_index_extensions[2] >= .65
                and float(non_index_extensions[:2].max()) <= .55
                and pinky_span >= .45 * index_span
            )
            for name, supported in (("peace", peace_shape), ("rock", rock_shape)):
                if supported:
                    model_support = float(probabilities[self.config.class_to_idx[name]])
                    if model_support >= .15:
                        return _set_probability_floor(
                            probabilities, self.config.class_to_idx[name], .92
                        ), {
                            "valid": True,
                            "reason": f"{name} finger geometry overrides an ambiguous direction",
                            "strong_geometry": False,
                            "model_supported_pose": False,
                        }
                    # Some genuine one-finger camera views have spurious
                    # straight middle/pinky landmarks. Do not veto those when
                    # the new command has essentially no model support.
                    if model_support >= .08:
                        return probabilities, {
                            "valid": False,
                            "reason": "more than one finger is extended; not a one-finger direction",
                            "strong_geometry": False,
                            "model_supported_pose": False,
                        }
        dominance = float(np.max(np.abs(vector)) / norm)
        # The world joints retain the index's length when it points partly
        # toward the lens. They establish the pose, while the image projection
        # still determines the command's camera-relative direction.
        projected_reach = norm / max(float(np.linalg.norm(points[9] - points[0])), 1e-8)
        world_directional = bool(
            world_index_only and world_projection_fraction >= .20
            and projected_reach >= .18
        )
        strict_pose = bool(
            pose_score >= 0.50
            and float(non_index_extensions.mean()) <= 0.70
        )
        dx, dy = map(float, vector)
        gesture = (
            ("down" if dy > 0 else "up")
            if abs(dy) >= abs(dx)
            # NCM command calibration is horizontally inverted relative to the
            # old source labels. This is a semantic swap, not an image mirror.
            else ("left" if dx > 0 else "right")
        )
        if horizontal_pointing:
            gesture = "left" if dx > 0 else "right"
        if continued_direction:
            gesture = preferred_direction
        # Folded fingers can project as straight segments in a side view. Only
        # relax their joint-angle check when the classifier agrees and the
        # index is straight and clearly leads every other fingertip.
        palm_scale = max(float(np.linalg.norm(points[9] - points[0])), 1e-6)
        index_path = sum(
            float(np.linalg.norm(points[joint + 1] - points[joint]))
            for joint in (5, 6, 7)
        )
        index_straightness = norm / max(index_path, 1e-8)
        index_lead = min(
            float(np.dot(points[8] - points[tip], vector / norm) / palm_scale)
            for tip in (12, 16, 20)
        )
        index_reach = norm / palm_scale
        # Compare each fingertip with its own MCP. This avoids penalising a
        # naturally short index finger merely because the neighbouring MCPs
        # begin farther along the requested screen direction.
        maximum_other_projected_reach = max(
            float(np.dot(points[tip] - points[mcp], vector / norm) / palm_scale)
            for mcp, tip in ((9, 12), (13, 16), (17, 20))
        )
        relative_index_reach = index_reach - maximum_other_projected_reach
        index_extension = finger_extension_score(points, 5, 6, 7, 8)
        projected_hand = return_main_pose_geometry(points)
        finger_mcps = np.asarray((5, 9, 13, 17))
        finger_tips = np.asarray((8, 12, 16, 20))
        finger_vectors = points[finger_tips] - points[finger_mcps]
        finger_norms = np.linalg.norm(finger_vectors, axis=1)
        finger_units = finger_vectors / np.maximum(finger_norms[:, None], 1e-8)
        finger_reaches = finger_norms / palm_scale
        finger_extensions = np.concatenate((
            np.asarray([index_extension], dtype=np.float32),
            non_index_extensions,
        ))
        palm_down_alignment = float((points[9, 1] - points[0, 1]) / palm_scale)
        fingertip_descent = np.diff(points[finger_tips, 1]) / palm_scale
        inner_fingertip_y = points[finger_tips[:3], 1]
        inner_tip_spread = float(np.ptp(inner_fingertip_y) / palm_scale)
        outer_tip_drop = float(
            (points[finger_tips[3], 1] - float(inner_fingertip_y.max())) / palm_scale
        )
        outer_reach_advantage = float(
            finger_reaches[3] / max(float(finger_reaches[:3].mean()), 1e-6)
        )
        fingertip_y = points[finger_tips, 1]
        deepest_finger_index = int(np.argmax(fingertip_y))
        remaining_fingers = np.arange(len(finger_tips)) != deepest_finger_index
        remaining_tip_y = fingertip_y[remaining_fingers]
        edge_tip_cluster_spread = float(np.ptp(remaining_tip_y) / palm_scale)
        deepest_edge_tip_drop = float(
            (fingertip_y[deepest_finger_index] - float(remaining_tip_y.max()))
            / palm_scale
        )
        deepest_edge_reach = float(finger_reaches[deepest_finger_index])
        deepest_edge_reach_advantage = float(
            deepest_edge_reach
            / max(float(finger_reaches[remaining_fingers].mean()), 1e-6)
        )
        # In a steep edge-on Down pose, MediaPipe can project the folded-finger
        # chains almost parallel with the index and place the pinky outline past
        # the index tip. Preserve this tightly bounded silhouette without
        # weakening the ordinary index-lead requirement for other directions.
        edge_on_down = bool(
            gesture == "down"
            and dominance >= .82
            and index_straightness >= .76
            and index_reach >= .55
            and index_extension >= .35
            and projected_hand["downward_finger_count"] == 3
            and projected_hand["downward_score"] >= .90
            and projected_hand["parallel_score"] >= .88
            and projected_hand["together_score"] >= .50
            and projected_hand["score"] < .76
        )
        # A second edge-on projection can make the visible index, middle, and
        # ring chains curl back toward their MCPs while the outer hand edge is
        # reconstructed as a long pinky chain. The ordered fingertip cascade
        # and four aligned MCP-to-tip vectors identify this Down silhouette.
        curled_edge_down = bool(
            gesture == "down"
            and dominance >= .88
            and index_reach >= .40
            # The wrist-to-palm line tilts slightly as the same edge-on pose
            # moves across the NCM lens, even though all finger vectors remain
            # almost vertical. Allow that perspective shift while retaining
            # the much stricter four-finger alignment below.
            and palm_down_alignment >= .85
            and float(finger_units[:, 1].min()) >= .86
            and projected_hand["parallel_score"] >= .92
            # Adjacent tips can be nearly level at this camera angle. They may
            # not reverse order, which continues to reject a folded non-Down
            # hand while accepting small landmark jitter between frames.
            and bool(np.all(fingertip_descent >= 0.0))
            and float(finger_extensions[:3].max()) <= .45
            and finger_extensions[3] >= .75
            and finger_reaches[3] >= 1.45
            and projected_hand["downward_finger_count"] == 1
            and projected_hand["score"] < .55
        )
        # At the most foreshortened NCM angle the first three fingertips merge
        # into a compact row and the outer finger can also appear locally bent.
        # The row plus a substantially lower outer tip is a stable silhouette
        # across these frames, so it does not depend on any one joint looking
        # straight or on pixel-level ordering inside the clustered row.
        clustered_edge_down = bool(
            gesture == "down"
            and dominance >= .80
            and index_reach >= .55
            and palm_down_alignment >= .80
            and float(finger_units[:, 1].min()) >= .80
            and projected_hand["downward_score"] >= .95
            and projected_hand["parallel_score"] >= .84
            and projected_hand["downward_finger_count"] <= 1
            and projected_hand["score"] < .55
            and inner_tip_spread <= .45
            and outer_tip_drop >= .75
            and float(finger_extensions[:3].max()) <= .45
            and finger_reaches[3] >= 1.55
            and outer_reach_advantage >= 1.55
        )
        # In the reported NCM view the wrist can project below, or almost level
        # with, the MCP row even though every MCP-to-tip vector points down. The
        # three shorter tips form a compact row and one edge finger is reconstructed
        # much farther down. MediaPipe can assign that edge to either the index or
        # pinky side at this angle, so the test is independent of left/right finger
        # identity. Do not require locally straight joints in the three short chains.
        reverse_palm_edge_down = bool(
            gesture == "down"
            and dominance >= .78
            and palm_down_alignment <= .35
            and deepest_finger_index in (0, 3)
            and float(finger_units[:, 1].min()) >= .75
            and projected_hand["downward_score"] >= .90
            and projected_hand["parallel_score"] >= .80
            and edge_tip_cluster_spread <= .90
            and deepest_edge_tip_drop >= .55
            and deepest_edge_reach >= 1.20
            and deepest_edge_reach_advantage >= 1.30
        )
        # In this NCM view the real index remains a long, straight downward
        # chain while the three folded fingertips collapse into a tight row.
        # Their projected joints can look locally straight, so rely on the
        # index's palm-relative reach and lead instead of their angle scores.
        prominent_index_down = bool(
            gesture == "down"
            and dominance >= .90
            and index_extension >= .75
            and index_straightness >= .90
            and index_reach >= .90
            and index_lead >= .40
            and relative_index_reach >= .20
            and palm_down_alignment >= .60
            and deepest_finger_index == 0
            and edge_tip_cluster_spread <= .45
            and deepest_edge_tip_drop >= .45
            and deepest_edge_reach_advantage >= 1.40
        )
        ordered = np.sort(probabilities)
        model_agrees = bool(
            raw == gesture
            and float(ordered[-1]) >= self.config.confidence_floor
            and float(ordered[-1] - ordered[-2]) >= self.config.probability_margin_floor
        )
        # Folded OTHER fingers supplied half the old pose score, so a folded
        # index could pass. Require extension and prominence of the index itself.
        index_only = bool(index_extension >= .45 and index_straightness >= .70 and index_lead >= .04)
        short_straight_index = bool(
            index_straightness >= .88
            and index_reach >= .27
            and max(index_lead, relative_index_reach) >= .12
            and float(non_index_extensions.mean()) <= .35
        )
        index_only = bool(index_only or short_straight_index)
        supported_pose = bool(model_agrees and index_extension >= .45
                              and index_straightness >= .80 and index_lead >= .20)
        supported_pose = bool(
            supported_pose
            or (model_agrees and short_straight_index)
            or edge_on_down
            or curled_edge_down
            or clustered_edge_down
            or reverse_palm_edge_down
            or prominent_index_down
            or expanded_thumb_pointing
            or horizontal_pointing
            or world_directional
            or continued_direction
        )
        settings = (self.config.raw.get("directional_resolution") or {})
        minimum_axis_dominance = float(settings.get("minimum_axis_dominance", 0.73))
        directional = bool(
            (dominance >= minimum_axis_dominance or horizontal_pointing)
            and (not world_index_only or world_projection_fraction >= .20
                 or continued_direction
                 or (horizontal_pointing and horizontal_shape["clear_single_index"]))
            and (
                (index_only and (strict_pose or supported_pose))
                or edge_on_down
                or curled_edge_down
                or clustered_edge_down
                or reverse_palm_edge_down
                or prominent_index_down
                or horizontal_pointing
                or world_directional
                or continued_direction
            )
        )
        strong_short_index = bool(
            short_straight_index
            and gesture == "down"
            and model_agrees
            and float(ordered[-1]) >= .95
            and dominance >= .80
        )
        details = {
            "valid": directional if raw in {"left", "right", "up", "down"} else True,
            "gesture": gesture,
            "pose_score": pose_score,
            "axis_dominance": dominance,
            "maximum_non_index_extension": float(non_index_extensions.max()),
            "mean_non_index_extension": float(non_index_extensions.mean()),
            "index_straightness": index_straightness,
            "index_lead_ratio": index_lead,
            "index_reach_ratio": index_reach,
            "maximum_other_projected_reach": maximum_other_projected_reach,
            "relative_index_reach": relative_index_reach,
            "index_extension": index_extension,
            "index_only": index_only,
            "short_straight_index": short_straight_index,
            "edge_on_down": edge_on_down,
            "curled_edge_down": curled_edge_down,
            "clustered_edge_down": clustered_edge_down,
            "reverse_palm_edge_down": reverse_palm_edge_down,
            "prominent_index_down": prominent_index_down,
            "expanded_thumb_pointing": expanded_thumb_pointing,
            "arched_thumb_pointing": arched_thumb_pointing,
            "horizontal_pointing": horizontal_pointing,
            "horizontal_pointing_geometry": horizontal_shape,
            "world_index_only": world_index_only,
            "world_folded_pointing": world_folded_pointing,
            "world_pointing_contradiction": world_pointing_contradiction,
            "world_directional": world_directional,
            "continued_direction": continued_direction,
            "world_projection_fraction": world_projection_fraction,
            "projected_index_reach_ratio": projected_reach,
            "palm_down_alignment": palm_down_alignment,
            "minimum_finger_down_alignment": float(finger_units[:, 1].min()),
            "finger_reach_ratios": finger_reaches.tolist(),
            "fingertip_descent_ratios": fingertip_descent.tolist(),
            "inner_tip_spread_ratio": inner_tip_spread,
            "outer_tip_drop_ratio": outer_tip_drop,
            "outer_reach_advantage": outer_reach_advantage,
            "deepest_finger_index": deepest_finger_index,
            "edge_tip_cluster_spread_ratio": edge_tip_cluster_spread,
            "deepest_edge_tip_drop_ratio": deepest_edge_tip_drop,
            "deepest_edge_reach_ratio": deepest_edge_reach,
            "deepest_edge_reach_advantage": deepest_edge_reach_advantage,
            "strong_geometry": bool(
                directional
                and (
                    edge_on_down
                    or curled_edge_down
                    or clustered_edge_down
                    or reverse_palm_edge_down
                    or prominent_index_down
                    or expanded_thumb_pointing
                    or horizontal_pointing
                    or world_directional
                    or continued_direction
                    or (
                        model_agrees
                        and float(ordered[-1]) >= .95
                        and (
                            (index_extension >= .85 and index_straightness >= .92
                             and index_lead >= .30 and dominance >= .86)
                            or strong_short_index
                        )
                    )
                )
            ),
            "model_supported_pose": supported_pose,
            "horizontal_mirror": False,
            "horizontal_semantic_swap": True,
        }
        if not directional:
            if raw in {"left", "right", "up", "down"}:
                details["reason"] = (
                    "Point more horizontally or vertically; the direction is diagonal"
                    if dominance < minimum_axis_dominance else
                    "Extend the index finger past the other fingertips and hold the direction"
                )
            return probabilities, details
        floor = min(0.98, 0.92 + 0.06 * max(0.0, pose_score - 0.72) / 0.28)
        details["valid"] = True
        return _set_probability_floor(
            probabilities, self.config.class_to_idx[gesture], floor
        ), details

    def _hand_shape(
        self,
        probabilities: np.ndarray,
        landmarks: np.ndarray,
        handedness: str | None = None,
        handedness_confidence: float | None = None,
        world_landmarks: np.ndarray | None = None,
        require_fist_depth: bool = False,
        model_probabilities: np.ndarray | None = None,
        image_shape=None,
    ) -> tuple[np.ndarray, dict[str, float | int | bool | str | None]]:
        points = np.asarray(landmarks, dtype=np.float32).reshape(21, 2)
        adjusted = np.asarray(probabilities, dtype=np.float64).copy()
        raw = self.config.class_names[int(np.argmax(adjusted))]
        model_raw = (
            self.config.class_names[int(np.argmax(model_probabilities))]
            if model_probabilities is not None else raw
        )
        palm_scale = max(float(np.linalg.norm(points[9] - points[0])), 1e-6)
        extensions = np.asarray([
            finger_extension_score(points, 1, 2, 3, 4),
            finger_extension_score(points, 5, 6, 7, 8),
            finger_extension_score(points, 9, 10, 11, 12),
            finger_extension_score(points, 13, 14, 15, 16),
            finger_extension_score(points, 17, 18, 19, 20),
        ], dtype=np.float32)
        peace_image = peace_finger_geometry(points)
        peace_world = (
            peace_finger_geometry(world_landmarks)
            if world_landmarks is not None else None
        )
        # Depth resolves a real V whose index finger is foreshortened in the
        # camera image. When depth is present it also vetoes a raised middle
        # finger whose folded index looks straight in the 2-D projection.
        peace_support = (
            float(adjusted[self.config.class_to_idx["peace"]])
            if "peace" in self.config.class_to_idx
            else 0.0
        )
        # A held-out V can have a slightly raised ring/pinky landmark even
        # while the two intended fingers are clearly extended. Permit that
        # only with near-certain model evidence. World landmarks, when present,
        # remain authoritative so a projected middle finger is still vetoed.
        peace_model_geometry = bool(
            raw == "peace"
            and peace_support >= .98
            and peace_image["index_extension"] >= .75
            and peace_image["middle_extension"] >= .65
            and .70 <= peace_image["index_reach_ratio"] <= 1.45
            and peace_image["index_forward_ratio"] >= .25
            and peace_image["tip_separation_ratio"] >= MINIMUM_PEACE_TIP_SEPARATION_RATIO
            and min(extensions[3], extensions[4]) <= .65
            and max(extensions[3], extensions[4]) <= .85
        )
        peace_geometry_supported = bool(
            peace_world["valid"] if peace_world is not None
            else (peace_image["valid"] or peace_model_geometry)
        )
        peace_pair_recovery = bool(
            peace_image["strong_geometry"] and peace_geometry_supported
        )
        # Two extended fingers pointing up is unmistakably Peace, correcting an Up misclassification.
        is_two_finger_up = bool(
            model_raw == "up" and peace_image["valid"]
            and float(np.linalg.norm(points[6] - points[10])) / palm_scale >= .05
            and (peace_world is None or peace_world["valid"])
        )
        # The strongly verified pair can correct even a confident Up score.
        # Ordinary/foreshortened shapes still need classifier support.
        if peace_geometry_supported and (
            raw == "peace"
            or peace_pair_recovery
            or is_two_finger_up
            or peace_support >= (0.05 if peace_world is not None else 0.15)
        ):
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx["peace"], 0.96
            )
            raw = "peace"
        thumb_vector = points[4] - points[2]
        thumb_norm = max(float(np.linalg.norm(thumb_vector)), 1e-8)
        thumb_up_score = float(-thumb_vector[1] / thumb_norm)
        non_thumb = extensions[1:]
        extended_non_thumb_count = int((non_thumb >= 0.58).sum())
        gap_scale = _palm_gap_scale(points)
        gap = float(np.linalg.norm(points[4] - points[8]) / gap_scale)
        other_thumb_gaps = np.asarray([
            np.linalg.norm(points[4] - points[tip]) / gap_scale
            for tip in (12, 16, 20)
        ], dtype=np.float32)
        other_finger_reaches = np.asarray([
            np.linalg.norm(points[tip] - points[mcp]) / gap_scale
            for mcp, tip in ((9, 12), (13, 16), (17, 20))
        ], dtype=np.float32)
        thumb_middle_gap = float(other_thumb_gaps[0])
        closest_other_thumb_gap = float(other_thumb_gaps.min())
        extended_other_count = int(np.count_nonzero(
            (extensions[2:] >= .58) & (other_finger_reaches >= .60)
        ))
        dorsal = return_main_pose_geometry(points)
        foreshortened_fist = foreshortened_fist_geometry(points)
        palm_axis = points[[5, 9, 13, 17]].mean(axis=0) - points[0]
        palm_axis_norm = max(float(np.linalg.norm(palm_axis)), 1e-8)
        palm_up_alignment = float(-palm_axis[1] / palm_axis_norm)
        # Match the HUD's signed XY angle (clockwise from screen-right).
        # Use world XY when available, just as hand_plane_angles does.
        dorsal_axis = palm_axis
        if world_landmarks is not None:
            world_points = np.asarray(world_landmarks, dtype=np.float32)
            if world_points.shape == (21, 3) and np.isfinite(world_points).all():
                world_xy = world_points[[5, 9, 13, 17], :2].mean(axis=0) - world_points[0, :2]
                if float(np.linalg.norm(world_xy)) > 1e-8:
                    dorsal_axis = world_xy
        dorsal_xy = float(np.degrees(np.arctan2(dorsal_axis[1], dorsal_axis[0])))
        # The downward/back-of-hand command occupies the forward and lower
        # screen semicircle. A sideways raised hand can have the same surface
        # and nearly parallel fingers, especially when the model reports no
        # known-gesture mass; it must not recover as Dorsal.
        dorsal_allowed_direction = -1e-4 <= dorsal_xy <= 150.0001
        # Check finger shape in a palm-aligned frame so sideways/downward
        # rotations retain the same extension/togetherness requirements.
        dorsal_rotation = np.pi / 2 - np.arctan2(palm_axis[1], palm_axis[0])
        c, s = np.cos(dorsal_rotation), np.sin(dorsal_rotation)
        aligned_dorsal = return_main_pose_geometry(
            (points - points[0]) @ np.asarray([[c, -s], [s, c]]).T
        )
        palm_axis_dominance = float(np.max(np.abs(palm_axis)) / palm_axis_norm)
        open_palm_upward = palm_up_alignment >= .70
        stop_shape = stop_inverted_geometry(points, image_shape)
        open_palm_allowed_direction = bool(stop_shape["allowed_direction"])
        surface_settings = self.config.raw.get("palm_surface_validation") or {}
        surface_validation_enabled = bool(surface_settings.get("enabled", True))
        normalized_handedness = str(handedness or "").strip().lower()
        surface_validation_active = bool(
            surface_validation_enabled and normalized_handedness in {"left", "right"}
        )
        surface_handedness = handedness if surface_validation_active else None
        surface = hand_surface_orientation(
            points,
            surface_handedness,
            handedness_confidence,
            minimum_handedness_confidence=float(
                surface_settings.get("minimum_handedness_confidence", 0.75)
            ),
            surface_winding_margin=float(
                surface_settings.get("surface_winding_margin", 0.20)
            ),
        )
        palm_visible = bool(surface["palm_visible"])
        open_pair_model_support = bool(
            model_raw in {"stop_inverted", "open_palm"}
            and float(np.max(model_probabilities if model_probabilities is not None else probabilities))
            >= self.config.confidence_floor
        )
        stop_depth = stop_inverted_depth_geometry(world_landmarks)
        stop_inverted_geometry_supported = bool(
            stop_shape["strong_geometry"] and surface["dorsal_visible"] and stop_depth["valid"])
        stop_inverted_valid = bool(
            (open_pair_model_support or stop_inverted_geometry_supported) and stop_shape["valid"]
            and surface["dorsal_visible"] and stop_depth["valid"]
        )
        stop_rejection_reason = None
        if not stop_inverted_valid:
            if not stop_shape["allowed_direction"]:
                stop_rejection_reason = OPEN_HAND_LOWER_ARC_REJECTION_REASON
            elif not stop_shape["extended_fingers"]:
                stop_rejection_reason = "Stop inv: extend all four fingers"
            elif not surface["dorsal_visible"]:
                stop_rejection_reason = ("Stop inv: show the back of the hand" if palm_visible
                                         else "Stop inv: hand surface is uncertain")
            elif not stop_depth["valid"]:
                stop_rejection_reason = stop_depth["reason"]
            else:
                stop_rejection_reason = "Stop inv: open-hand shape needs more confidence"
        dorsal_shape = aligned_dorsal
        # Deliberate Dorsal uses a closed thumb beside four straight, together
        # fingers. A loose hand supporting the chin can otherwise have the
        # same back-of-hand winding and receive a strong geometry override.
        index_axis = points[8] - points[5]
        index_axis_sq = max(float(np.dot(index_axis, index_axis)), 1e-8)
        unclipped_thumb_projection = float(np.dot(points[4] - points[5], index_axis) / index_axis_sq)
        thumb_projection = float(np.clip(unclipped_thumb_projection, 0., 1.))
        thumb_index_gap = float(np.linalg.norm(
            points[4] - (points[5] + thumb_projection * index_axis)))
        dorsal_thumb_gap_ratio = thumb_index_gap / max(float(np.linalg.norm(points[9] - points[0])), 1e-8)
        together_dorsal = bool(
            dorsal_shape.get("minimum_extension_score", 0) >= .45
            and float(non_thumb.min()) >= .45
            and float(non_thumb.mean()) >= .68
            and dorsal_thumb_gap_ratio <= 1.50
            and unclipped_thumb_projection >= 0.0
        )
        spread_dorsal = together_dorsal
        deliberate_dorsal = bool(together_dorsal or spread_dorsal)
        open_palm_valid = bool(
            extended_non_thumb_count == 4
            and float(non_thumb.mean()) >= 0.78
            and open_palm_allowed_direction
            # Offline caches do not contain handedness, so preserve their
            # historical evaluation path. A live hand with reported but weak
            # or ambiguous handedness is not allowed to claim Open Palm.
            and (not surface_validation_active or palm_visible)
        )
        palm_geometry_supported = bool(
            open_palm_valid
            and palm_visible
            and float(non_thumb.min()) >= .65
            and float(non_thumb.mean()) >= .82
            and extensions[0] >= .45
        )
        # The classifier consumes XY only, so a mirrored open hand can have
        # the same features on the other physical surface. Detector handedness
        # separates the two trained open-hand commands, even with a close thumb.
        palm_geometry_supported = bool(palm_geometry_supported or (
            open_pair_model_support and model_raw == "stop_inverted"
            and open_palm_valid and palm_visible and stop_shape["valid"]))
        # The current classifier was trained with horizontal palms among its
        # negatives. Only a strongly verified sideways palmar hand may recover
        # when that classifier reports almost no known-vocabulary mass.
        sideways_palm_recovery = bool(
            palm_geometry_supported and palm_up_alignment <= .75
        )
        palm_scale = max(float(np.linalg.norm(points[9] - points[0])), 1e-6)
        thumb_lead = float(min(points[tip, 1] - points[4, 1] for tip in (8, 12, 16, 20)) / palm_scale)
        non_thumb_reach = np.asarray([
            np.linalg.norm(points[tip] - points[mcp]) / palm_scale
            for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20))
        ], dtype=np.float32)
        raised_non_thumb_count = int(np.count_nonzero(
            (non_thumb >= .58) & (non_thumb_reach >= .70)
        ))
        # A high LIKE score cannot turn one fully raised finger into a thumb.
        # Check reach as well as joint angles: curled fingers can look straight
        # from the side, but their tips stay near the knuckles.
        like_finger_shape = bool(
            thumb_lead >= .10 and raised_non_thumb_count == 0
        )
        # The model can retain Like/Thumb Down through a screen rotation even
        # when the vertical thumb lead test changes sign. Require the world
        # joints to show one long thumb and four genuinely folded fingers.
        world_thumb_only = False
        if world_landmarks is not None:
            world_points = np.asarray(world_landmarks, dtype=np.float32)
            if world_points.shape == (21, 3) and np.isfinite(world_points).all():
                world_scale = max(float(np.linalg.norm(world_points[9] - world_points[0])), 1e-8)
                thumb_vector_world = world_points[4] - world_points[2]
                thumb_reach_world = float(np.linalg.norm(thumb_vector_world))
                thumb_path_world = float(np.linalg.norm(world_points[3] - world_points[2])
                                         + np.linalg.norm(world_points[4] - world_points[3]))
                thumb_unit_world = thumb_vector_world / max(thumb_reach_world, 1e-8)
                other_world_extensions = [finger_extension_score(
                    world_points, m, m + 1, m + 2, m + 3) for m in (5, 9, 13, 17)]
                other_world_reaches = [float(np.linalg.norm(world_points[m + 3] - world_points[m]))
                                       / world_scale for m in (5, 9, 13, 17)]
                thumb_lead_world = min(float(np.dot(
                    world_points[4] - world_points[tip], thumb_unit_world)) / world_scale
                    for tip in (8, 12, 16, 20))
                world_thumb_only = bool(
                    finger_extension_score(world_points, 1, 2, 3, 4) >= .70
                    and thumb_reach_world / max(thumb_path_world, 1e-8) >= .90
                    and thumb_reach_world / world_scale >= .50
                    and thumb_lead_world >= .35
                    and max(other_world_extensions) <= .45
                    and max(other_world_reaches) <= .50
                )
        rotated_thumb_model_support = bool(
            world_thumb_only and raw in {"like", "thumb_down"}
            and float(adjusted.max()) >= .60
            and float(adjusted.max() - np.partition(adjusted, -2)[-2]) >= .25
        )
        like_valid = bool(
            like_finger_shape
            and (
                (extensions[0] >= .60 and float(non_thumb.mean()) <= .72
                 and thumb_up_score >= .55)
                # A side-on thumb can look shortened in the camera image.
                or (raw == "like" and float(adjusted.max()) >= .90
                    and extensions[0] >= .40 and thumb_up_score >= .20)
            )
        )
        like_valid = like_valid or (raw == "like" and rotated_thumb_model_support)
        # In an edge-on fist, folded fingers can look straight in the image.
        # Their tips still cluster around the palm; when available, the world
        # joints provide an independent check of the actual finger bends.
        world_folded_fingers = False
        world_closed_fingers = False
        if world_landmarks is not None:
            world_points = np.asarray(world_landmarks, dtype=np.float32)
            if world_points.shape == (21, 3) and np.isfinite(world_points).all():
                world_non_thumb = np.asarray([
                    finger_extension_score(world_points, mcp, mcp + 1, mcp + 2, mcp + 3)
                    for mcp in (5, 9, 13, 17)
                ])
                world_folded_fingers = bool(
                    float(world_non_thumb.max()) <= .80
                    and float(world_non_thumb.mean()) <= .55
                )
                world_closed_fingers = bool(
                    float(world_non_thumb.max()) <= .55
                    and float(world_non_thumb.mean()) <= .38
                )
        compact_like_fingers = bool(
            (float(non_thumb.max()) <= .50 and float(non_thumb.mean()) <= .38)
            or (
                float(foreshortened_fist["maximum_tip_radius_ratio"]) <= 1.40
                and raised_non_thumb_count == 0
                and float(non_thumb.mean()) <= .72
            )
            or (
                float(foreshortened_fist["maximum_tip_radius_ratio"]) <= 1.50
                and world_folded_fingers
            )
        )
        # A side-on thumbs-up can receive any classifier label. Require a
        # strongly extended/upward thumb that leads compact, folded fingertips.
        like_geometry_supported = bool(
            extensions[0] >= .70
            and float(foreshortened_fist["thumb_radius_ratio"]) >= 1.10
            and compact_like_fingers
            and thumb_up_score >= .70
            and thumb_lead >= .25
            and palm_up_alignment >= -.35
        )
        like_geometry_supported = like_geometry_supported or bool(
            raw == "like" and rotated_thumb_model_support)
        like_valid = like_valid or like_geometry_supported
        thumb_down_score = float(thumb_vector[1] / thumb_norm)
        thumb_down_lead = float(
            min(points[4, 1] - points[tip, 1] for tip in (8, 12, 16, 20))
            / palm_scale
        )
        thumb_down_valid = bool(
            extensions[0] >= .55
            and float(non_thumb.mean()) <= .72
            and thumb_down_score >= .30
            and thumb_down_lead >= .15
        )
        thumb_down_valid = thumb_down_valid or bool(
            raw == "thumb_down" and float(adjusted.max()) >= .85
            and extensions[0] >= .40
            and thumb_down_score >= .25
            and thumb_down_lead >= .15
        )
        thumb_down_valid = thumb_down_valid or bool(
            raw == "thumb_down" and rotated_thumb_model_support)
        thumb_down_geometry_supported = bool(
            thumb_down_valid
            and extensions[0] >= .70
            and float(non_thumb.mean()) <= .45
            and thumb_down_score >= .70
            and thumb_down_lead >= .25
        )
        thumb_down_geometry_supported = thumb_down_geometry_supported or bool(
            raw == "thumb_down" and rotated_thumb_model_support)
        # A high model score alone is not evidence of a closed hand. In
        # particular, four partly bent but raised fingers can look like Fist
        # to the MLP. Keep a geometry requirement for every Fist decision.
        fist_compact = bool(
            float(non_thumb.max()) <= .55
            and float(non_thumb.mean()) <= .38
            and extensions[0] <= .85
        )
        # A compact, tucked-thumb fist corroborated by 3-D joint bends does
        # not need a special screen-angle cone or a minimum Fist model score.
        world_compact_fist = bool(
            fist_compact and world_closed_fingers
            and extensions[0] <= .45
            and float(foreshortened_fist["maximum_tip_radius_ratio"]) <= .85
            and float(foreshortened_fist["thumb_radius_ratio"]) <= .85
        )
        # World depth distinguishes pitch from screen rotation. Negative Z is
        # toward the camera; a 30-degree forward pitch has unit-axis Z=-0.5.
        fist_depth_alignment = None
        fist_side_view_alignment = None
        reported_reverse_alignment = None
        reported_side_alignment = None
        tilted_side_alignment = None
        if world_landmarks is not None:
            world_points = np.asarray(world_landmarks, dtype=np.float32)
            if world_points.shape == (21, 3) and np.isfinite(world_points).all():
                world_axis = world_points[[5, 9, 13, 17]].mean(axis=0) - world_points[0]
                world_axis_length = float(np.linalg.norm(world_axis))
                if world_axis_length >= 1e-8:
                    world_axis_unit = world_axis / world_axis_length
                    fist_depth_alignment = float(world_axis_unit[2])
                    # The reported side view has XY=+8, YZ=+68, ZX=+72.
                    # These are projections of one 3-D axis. A cone around
                    # that axis gives the requested 18-degree tolerance.
                    side_axis = np.asarray([
                        np.tan(np.deg2rad(72.0)),
                        1.0 / np.tan(np.deg2rad(68.0)),
                        1.0,
                    ], dtype=np.float32)
                    side_axis /= np.linalg.norm(side_axis)
                    mirrored_axis = side_axis * [-1.0, 1.0, 1.0]
                    fist_side_view_alignment = max(
                        float(np.dot(world_axis_unit, side_axis)),
                        float(np.dot(world_axis_unit, mirrored_axis)),
                    )
                    reported_reverse_alignment = _mirrored_view_alignment(
                        world_axis_unit, -130.0, -132.0,
                    )
                    reported_side_alignment = _mirrored_view_alignment(
                        world_axis_unit, -5.0, 80.0,
                    )
                    # The newly accepted side view has XY=-3, YZ=-99, ZX=107.
                    tilted_side_alignment = _mirrored_view_alignment(
                        world_axis_unit, -3.0, 107.0,
                    )
        reported_side_view = bool(
            reported_side_alignment is not None
            and reported_side_alignment >= float(np.cos(np.deg2rad(18.0)))
        )

        tilted_side_view = bool(
            tilted_side_alignment is not None
            and tilted_side_alignment >= float(np.cos(np.deg2rad(10.0)))
        )
        reported_side_fist_geometry = bool(
            (reported_side_view or tilted_side_view)
            and float(foreshortened_fist["maximum_tip_radius_ratio"]) <= .95
            and float(foreshortened_fist["thumb_radius_ratio"]) <= 1.10
            and float(np.linalg.norm(
                points[[8, 12, 16, 20]] - points[[5, 9, 13, 17]], axis=1
            ).max() / palm_axis_norm) <= .70
            and float(non_thumb.max()) <= .62
            and float(non_thumb.mean()) <= .42
            and extensions[0] <= .85
            and not thumb_down_valid
            and not like_valid
        )
        fist_side_view_geometry = bool(
            fist_side_view_alignment is not None
            and fist_side_view_alignment >= float(np.cos(np.deg2rad(18.0))) - 1e-6
            # At this edge-on view MediaPipe can place curled tips just past
            # their MCPs. Joint bend plus tip proximity is more reliable than
            # signed curlback in the image projection.
            and float(foreshortened_fist["maximum_tip_radius_ratio"]) <= .85
            and float(foreshortened_fist["thumb_radius_ratio"]) <= .85
            and float(non_thumb.max()) <= .55
            and float(non_thumb.mean()) <= .38
            and not thumb_down_valid
            and not like_valid
        )
        # Beyond the forward-pitch range, retain the established dorsal-view
        # recovery only with confident surface winding and strong finger curl.
        dorsal_fist_depth_recovery = bool(
            fist_depth_alignment is not None
            and fist_depth_alignment < FIST_MINIMUM_DEPTH_ALIGNMENT - FIST_ORIENTATION_EPSILON
            and bool(surface["dorsal_visible"])
            and world_folded_fingers
            and (foreshortened_fist["strong_geometry"] or (
                fist_compact
                and float(non_thumb.max()) <= .42
                and float(non_thumb.mean()) <= .28
                and float(foreshortened_fist["thumb_radius_ratio"]) <= .85
            ))
        )
        fist_depth_allowed = bool(
            (fist_depth_alignment is not None or not require_fist_depth)
            and (fist_depth_alignment is None
                 or fist_depth_alignment >= FIST_MINIMUM_DEPTH_ALIGNMENT - FIST_ORIENTATION_EPSILON
                 or dorsal_fist_depth_recovery)
        )
        # In image coordinates y increases downward. Horizontal fists remain
        # eligible on either side; any downward tilt leaves the command arc.
        fist_screen_axis = points[[5, 9, 13, 17]].mean(axis=0) - points[0]
        fist_screen_up_alignment = float(
            -fist_screen_axis[1] / max(float(np.linalg.norm(fist_screen_axis)), 1e-8)
        )
        fist_screen_direction_allowed = bool(
            float(np.linalg.norm(fist_screen_axis)) >= 1e-8
            and fist_screen_up_alignment >= -FIST_ORIENTATION_EPSILON
        )
        fist_allowed_direction = fist_depth_allowed and fist_screen_direction_allowed
        reverse_closed_fist = bool(
            fist_depth_alignment is not None and not fist_depth_allowed
            and (fist_compact or foreshortened_fist["strong_geometry"])
            and world_folded_fingers
            and float(foreshortened_fist["thumb_radius_ratio"]) <= .85
        )
        downward_closed_fist = bool(
            not fist_screen_direction_allowed
            and (fist_compact or foreshortened_fist["strong_geometry"])
            and world_folded_fingers
            and float(foreshortened_fist["thumb_radius_ratio"]) <= .85
        )

        fist_valid = bool(
            fist_allowed_direction
            and (fist_compact or foreshortened_fist["strong_geometry"]
                 or fist_side_view_geometry or reported_side_fist_geometry)
        )
        fist_geometry_supported = bool(
            fist_allowed_direction
            and (world_compact_fist or reported_side_fist_geometry or fist_side_view_geometry
                 or foreshortened_fist["strong_geometry"]
                 or (
                     fist_valid
                     and float(non_thumb.max()) <= .42
                     and float(non_thumb.mean()) <= .28
                 ))
        )
        tucked_thumb_fist_override = bool(
            fist_geometry_supported
            and foreshortened_fist["strong_geometry"]
            and float(foreshortened_fist["thumb_radius_ratio"]) <= .85
            and float(probabilities[self.config.class_to_idx["fist"]]) >= .08
        )
        fist_foreshortened_relabel = bool(
            fist_allowed_direction
            and (world_compact_fist or foreshortened_fist["strong_geometry"] or fist_side_view_geometry
                 or reported_side_fist_geometry
                 or (float(probabilities[self.config.class_to_idx["fist"]]) >= .08
                     and fist_geometry_supported
                     and (world_folded_fingers
                          or (float(non_thumb.max()) <= .38
                              and float(non_thumb.mean()) <= .25
                              and float(foreshortened_fist["thumb_radius_ratio"]) <= .85)))
                 or (raw in {"like", "thumb_down"} and fist_geometry_supported
                     and extensions[0] <= .45
                     and float(foreshortened_fist["thumb_radius_ratio"]) >= .75
                     and float(probabilities[self.config.class_to_idx["fist"]]) >= .08))
            # A projected Fist and the two thumb commands can share the same
            # four curled fingertips. Preserve a thumb command only when its
            # thumb geometry actually passes validation.
            and (raw != "like" or not like_valid)
            and (raw != "thumb_down" or not thumb_down_valid
                 or tucked_thumb_fist_override)
        )
        ok_valid = bool(
            gap <= 0.58
            # Only the index tip forms the pinch. All three remaining tips
            # must stay away from the thumb, with at least two truly extended.
            and closest_other_thumb_gap >= .70
            and closest_other_thumb_gap - gap >= .25
            and extended_other_count >= 2
            and extensions[1] <= 0.82
        )
        validators = {
            "stop_inverted": stop_inverted_valid,
            "open_palm": open_palm_valid,
            "like": like_valid,
            "dorsal": False,
            "ok": ok_valid,
            "fist": fist_valid,
            "thumb_down": thumb_down_valid,
            "peace": peace_geometry_supported,
        }
        # Geometry may safely resolve the two open-hand orientations even when
        # perspective causes the MLP to swap them.
        # Only the perspective-specific curl-back signature may relabel a
        # different top class. The ordinary compact-fist validator remains a
        # guard for an already predicted Fist; otherwise it can resemble the
        # clustered Down silhouettes covered by the camera regressions.
        if stop_inverted_valid:
            # Resolve the explicit Stop-inverted shape before applying the
            # retired-output rejection. Strong independent evidence can also
            # recover the camera-domain classifier misses in the reported clip.
            adjusted = np.asarray(model_probabilities if model_probabilities is not None
                                  else probabilities, dtype=np.float64).copy()
            if model_raw == "open_palm" or stop_inverted_geometry_supported:
                adjusted = _set_probability_floor(adjusted, self.config.class_to_idx["stop_inverted"], .96)
            raw = "stop_inverted"
        elif raw == "thumb_down" and rotated_thumb_model_support:
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx["thumb_down"], 0.96
            )
        elif like_geometry_supported:
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx["like"], 0.96
            )
            raw = "like"
        elif fist_foreshortened_relabel:
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx["fist"], 0.96
            )
            raw = "fist"
        elif palm_geometry_supported or (open_palm_valid and raw == "open_palm"):
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx["open_palm"], 0.96
            )
            raw = "open_palm"
        valid = validators.get(raw, True)
        if valid:
            reason = "pose geometry accepted"
        elif raw == "stop_inverted":
            reason = stop_rejection_reason
        elif raw == "fist" and require_fist_depth and fist_depth_alignment is None:
            reason = "fist depth landmarks unavailable"
        elif raw == "fist" and not fist_depth_allowed:
            reason = "fist fingers face away from the camera"
        elif raw == "fist" and not fist_screen_direction_allowed:
            reason = "fist points below the horizontal"
        elif raw == "like" and not like_finger_shape:
            reason = "like requires the thumb to lead and other fingers to stay folded"
        elif raw == "dorsal" and not deliberate_dorsal:
            reason = DORSAL_REJECTION_REASON
        elif raw == "ok" and (closest_other_thumb_gap < .70 or closest_other_thumb_gap - gap < .25):
            reason = "ok requires only the index fingertip to touch the thumb"
        else:
            reason = f"{raw} hand shape failed geometry checks"
        return adjusted, {
            "valid": bool(valid),
            "reason": reason,
            "resolved_gesture": raw,
            "thumb_extension": float(extensions[0]),
            "world_thumb_only": world_thumb_only,
            "index_extension": float(extensions[1]),
            "middle_extension": float(extensions[2]),
            "ring_extension": float(extensions[3]),
            "pinky_extension": float(extensions[4]),
            "thumb_up_score": thumb_up_score,
            "thumb_lead_ratio": thumb_lead,
            "raised_non_thumb_count": raised_non_thumb_count,
            "thumb_down_score": thumb_down_score,
            "thumb_down_lead_ratio": thumb_down_lead,
            "thumb_index_gap_ratio": gap,
            "thumb_middle_gap_ratio": thumb_middle_gap,
            "closest_other_thumb_gap_ratio": closest_other_thumb_gap,
            "extended_ok_other_finger_count": extended_other_count,
            "extended_non_thumb_count": extended_non_thumb_count,
            "open_palm_upward": open_palm_upward,
            "open_palm_allowed_direction": open_palm_allowed_direction,
            "open_hand_xy_degrees": stop_shape["xy_degrees"],
            "open_hand_lower_arc": bool(stop_shape["extended_fingers"]
                                         and not stop_shape["allowed_direction"]),
            "palm_up_alignment": palm_up_alignment,
            "palm_axis_dominance": palm_axis_dominance,
            "dorsal_score": float(dorsal["score"]),
            "dorsal_deliberate_shape": deliberate_dorsal,
            # This shape is rejection evidence at every screen angle. Do not
            # recycle its probability mass into a different command.
            "dorsal_hard_negative": bool(
                (deliberate_dorsal and not palm_visible
                 or (surface["dorsal_visible"] and float(non_thumb.min()) >= .45
                     and float(non_thumb.mean()) >= .68))
                and not world_thumb_only and not world_folded_fingers
                and not stop_inverted_valid
            ),
            "stop_inverted_valid": stop_inverted_valid,
            "stop_inverted_geometry_supported": stop_inverted_geometry_supported,
            "stop_inverted_geometry": stop_shape,
            "stop_inverted_depth_geometry": stop_depth,
            "stop_inverted_rejection_reason": stop_rejection_reason,
            "dorsal_thumb_index_gap_ratio": dorsal_thumb_gap_ratio,
            "dorsal_geometry_supported": False,
            "peace_geometry_supported": peace_geometry_supported,
            "peace_pair_recovery": peace_pair_recovery,
            "dorsal_range_recovery": False,
            "dorsal_xy_degrees": dorsal_xy,
            "dorsal_allowed_direction": dorsal_allowed_direction,
            "peace_model_geometry": peace_model_geometry,
            "peace_image_geometry": peace_image,
            "peace_world_geometry": peace_world,
            "palm_geometry_supported": palm_geometry_supported,
            "sideways_palm_recovery": sideways_palm_recovery,
            "like_compact_fingers": compact_like_fingers,
            "like_world_folded_fingers": world_folded_fingers,
            "like_geometry_supported": like_geometry_supported,
            "fist_geometry_supported": fist_geometry_supported,
            "fist_world_compact_geometry": world_compact_fist,
            "fist_toward_camera_pitch_degrees": (
                None if fist_depth_alignment is None else float(
                    np.rad2deg(np.arcsin(np.clip(-fist_depth_alignment, -1., 1.))))),
            "fist_allowed_direction": fist_allowed_direction,
            "fist_screen_up_alignment": fist_screen_up_alignment,
            "fist_screen_direction_allowed": fist_screen_direction_allowed,
            "fist_depth_allowed": fist_depth_allowed,
            "fist_dorsal_depth_recovery": dorsal_fist_depth_recovery,
            "fist_depth_alignment": fist_depth_alignment,
            "fist_side_view_alignment": fist_side_view_alignment,
            "fist_side_view_geometry": fist_side_view_geometry,
            "fist_reported_reverse_alignment": reported_reverse_alignment,
            "fist_reported_side_alignment": reported_side_alignment,
            "fist_reported_side_geometry": reported_side_fist_geometry,
            "fist_tilted_side_alignment": tilted_side_alignment,
            "fist_rejected_view_alignment": fist_depth_alignment,
            "fist_rejected_view": bool(
                fist_depth_alignment is not None and not fist_depth_allowed
            ),
            "reverse_closed_fist": reverse_closed_fist,
            "downward_closed_fist": downward_closed_fist,
            "fist_foreshortened_relabel": fist_foreshortened_relabel,
            "fist_retracted_finger_count": int(
                foreshortened_fist["retracted_finger_count"]
            ),
            "fist_mean_curlback_ratio": float(
                foreshortened_fist["mean_curlback_ratio"]
            ),
            "fist_maximum_tip_radius_ratio": float(
                foreshortened_fist["maximum_tip_radius_ratio"]
            ),
            "fist_thumb_radius_ratio": float(
                foreshortened_fist["thumb_radius_ratio"]
            ),
            "thumb_down_geometry_supported": thumb_down_geometry_supported,
            "hand_surface": surface["surface"],
            "hand_surface_score": float(surface["surface_score"]),
            "handedness": surface["handedness"],
            "handedness_confidence": float(surface["handedness_confidence"]),
            "downward_finger_count": int(dorsal["downward_finger_count"]),
        }

    def resolve(
        self,
        probabilities: np.ndarray,
        landmarks: np.ndarray,
        *,
        handedness: str | None = None,
        handedness_confidence: float | None = None,
        world_landmarks: np.ndarray | None = None,
        require_fist_depth: bool = False,
        preferred_direction: str | None = None,
        image_shape=None,
    ) -> tuple[np.ndarray, dict[str, dict | None]]:
        adjusted, directional = self._directional(
            probabilities, landmarks, world_landmarks, preferred_direction, image_shape)
        adjusted, shape = self._hand_shape(
            adjusted,
            landmarks,
            handedness=handedness,
            handedness_confidence=handedness_confidence,
            world_landmarks=world_landmarks,
            require_fist_depth=require_fist_depth,
            model_probabilities=probabilities,
            image_shape=image_shape,
        )
        # This tightly bounded single-index silhouette takes precedence over a
        # Dorsal relabel. At this angle the three folded chains can appear
        # parallel and downward even though only the index reaches past them.
        if directional.get("prominent_index_down", False):
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx["down"], .98
            )
        if directional.get("expanded_thumb_pointing", False):
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx[directional["gesture"]], .98)
        if directional.get("continued_direction", False):
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx[directional["gesture"]], .98)
        if directional.get("horizontal_pointing", False):
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx[directional["gesture"]], .98)
        resolved = self.config.class_names[int(np.argmax(adjusted))]
        valid = bool(shape.get("valid", True))
        reason = str(shape.get("reason", "pose geometry accepted"))
        if resolved in {"left", "right", "up", "down"}:
            valid = bool(directional.get("valid", False))
            reason = str(directional.get("reason", "direction geometry accepted"))
        if shape.get("reverse_closed_fist", False):
            valid = False
            reason = "fist fingers face away from the camera"
        elif shape.get("downward_closed_fist", False):
            valid = False
            reason = "fist points below the horizontal"
        if shape.get("open_hand_lower_arc", False):
            # Four extended fingers cannot become a different command simply
            # because the classifier changes its top label during rotation.
            valid = False
            reason = OPEN_HAND_LOWER_ARC_REJECTION_REASON
        # A single long index and three short folded chains are a pointing
        # hand, even when their projected joint angles look locally straight.
        single_index = bool(directional.get("prominent_index_down", False)
                            or directional.get("reverse_palm_edge_down", False)
                            or directional.get("horizontal_pointing", False))
        hard_negative = bool((shape.get("dorsal_hard_negative", False) and not single_index)
                             or resolved in DISABLED_GESTURES)
        if hard_negative:
            valid = False
            reason = DORSAL_REJECTION_REASON
            resolved = self.config.reject_label
            # Preserve the retired output as an explicit rejection sentinel.
            adjusted = np.zeros_like(adjusted)
            adjusted[self.config.class_to_idx["dorsal"]] = 1.0
        return adjusted, {
            "directional": directional,
            "hand_shape": shape,
            "pose_validation": {
                "valid": valid,
                "reason": reason,
                "gesture": resolved,
                "geometry_supported": bool(
                    not hard_negative and ((resolved == "open_palm" and shape.get("palm_geometry_supported", False))
                    or (resolved == "stop_inverted" and shape.get("stop_inverted_geometry_supported", False))
                    or (resolved == "like" and shape.get("like_geometry_supported", False))
                    or (resolved == "fist" and shape.get("fist_geometry_supported", False))
                    or (resolved == "thumb_down" and shape.get("thumb_down_geometry_supported", False))
                    or (resolved == "peace" and shape.get("peace_geometry_supported", False))
                    or (resolved in {"left", "right", "up", "down"} and directional.get("strong_geometry", False)))),
                "sideways_palm_recovery": bool(
                    resolved == "open_palm"
                    and shape.get("sideways_palm_recovery", False)
                ),
                "stop_inverted_recovery": bool(
                    resolved == "stop_inverted" and shape.get("stop_inverted_geometry_supported", False)
                ),
                "hard_negative": hard_negative,
                "dorsal_range_recovery": False,
                "expanded_thumb_direction_recovery": bool(
                    resolved in {"left", "right"} and directional.get("expanded_thumb_pointing", False)),
                "horizontal_direction_recovery": bool(
                    resolved in {"left", "right"} and directional.get("horizontal_pointing", False)),
            },
        }
