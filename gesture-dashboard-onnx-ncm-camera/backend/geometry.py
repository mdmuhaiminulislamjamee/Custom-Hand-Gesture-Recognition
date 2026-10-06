from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RuntimeConfig


MINIMUM_PEACE_TIP_SEPARATION_RATIO = 0.04


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
        and downward_score >= 0.72 and together >= 0.18
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
        and geometry.get("together_score", 0) >= 0.65
        and geometry.get("parallel_score", 0) >= 0.90
    )


def hand_surface_orientation(
    landmarks_xy: np.ndarray,
    handedness: str | None,
    handedness_confidence: float | None = None,
    *,
    minimum_handedness_confidence: float = 0.75,
    surface_winding_margin: float = 0.25,
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
        self, probabilities: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, dict | None]:
        points = np.asarray(landmarks, dtype=np.float32).reshape(21, 2)
        raw = self.config.class_names[int(np.argmax(probabilities))]
        # Peace and Rock also extend the index finger. The one-finger
        # directional recovery below must not rewrite a confidently classified
        # multi-finger command into Left/Right/Up/Down.
        if raw in {"peace", "rock"}:
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
            return probabilities, {
                "valid": raw not in {"left", "right", "up", "down"},
                "reason": "degenerate index-finger direction",
            }
        non_index_extensions = np.asarray([
            finger_extension_score(points, 9, 10, 11, 12),
            finger_extension_score(points, 13, 14, 15, 16),
            finger_extension_score(points, 17, 18, 19, 20),
        ], dtype=np.float32)
        if raw in {"left", "right", "up", "down"} and "peace" in self.config.class_to_idx:
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
        )
        settings = (self.config.raw.get("directional_resolution") or {})
        minimum_axis_dominance = float(settings.get("minimum_axis_dominance", 0.73))
        directional = bool(
            dominance >= minimum_axis_dominance
            and (
                (index_only and (strict_pose or supported_pose))
                or edge_on_down
                or curled_edge_down
                or clustered_edge_down
                or reverse_palm_edge_down
                or prominent_index_down
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
    ) -> tuple[np.ndarray, dict[str, float | int | bool | str | None]]:
        points = np.asarray(landmarks, dtype=np.float32).reshape(21, 2)
        adjusted = np.asarray(probabilities, dtype=np.float64).copy()
        raw = self.config.class_names[int(np.argmax(adjusted))]
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
        # The strongly verified pair can correct even a confident Up score.
        # Ordinary/foreshortened shapes still need classifier support.
        if peace_geometry_supported and (
            raw == "peace"
            or peace_pair_recovery
            or peace_support >= (0.05 if peace_world is not None else 0.15)
        ):
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx["peace"], 0.92
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
        dorsal_supported = dorsal_geometry_support(dorsal)
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
        dorsal_allowed_direction = bool(-1e-4 <= dorsal_xy <= 150.0 + 1e-4)
        # Check finger shape in a palm-aligned frame so sideways/downward
        # rotations retain the same extension/togetherness requirements.
        dorsal_rotation = np.pi / 2 - np.arctan2(palm_axis[1], palm_axis[0])
        c, s = np.cos(dorsal_rotation), np.sin(dorsal_rotation)
        aligned_dorsal = return_main_pose_geometry(
            (points - points[0]) @ np.asarray([[c, -s], [s, c]]).T
        )
        palm_axis_dominance = float(np.max(np.abs(palm_axis)) / palm_axis_norm)
        open_palm_orientation_settings = (
            self.config.raw.get("open_palm_orientation") or {}
        )
        open_palm_upward = palm_up_alignment >= .70
        # Horizontal palms are commands too. Permit a small downward component
        # for landmark jitter at the horizon, but reject clearly descending
        # wrist-to-palm axes (including diagonal-down poses).
        open_palm_allowed_direction = bool(
            palm_up_alignment >= float(
                open_palm_orientation_settings.get(
                    "minimum_non_down_alignment",
                    open_palm_orientation_settings.get("minimum_up_alignment", -.10),
                )
            )
        )
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
                surface_settings.get("surface_winding_margin", 0.25)
            ),
        )
        palm_visible = bool(surface["palm_visible"])
        dorsal_range_recovery = bool(
            dorsal_allowed_direction and surface["dorsal_visible"]
            and dorsal_geometry_support(aligned_dorsal)
            and float(np.linalg.norm(
                points[[8, 12, 16, 20]] - points[[5, 9, 13, 17]], axis=1
            ).min()) / palm_axis_norm >= .35
        )
        dorsal_supported = bool(
            (dorsal_supported or dorsal_range_recovery)
            and dorsal_allowed_direction
            and (not surface_validation_active or bool(surface["dorsal_visible"]))
        )
        dorsal_valid = bool(
            dorsal_range_recovery or (
                dorsal_allowed_direction
                and dorsal["score"] >= 0.76
                and dorsal["downward_finger_count"] >= 4
                and dorsal["minimum_extension_score"] >= 0.50
                and (not surface_validation_active or bool(surface["dorsal_visible"]))
            )
        )
        open_palm_valid = bool(
            extended_non_thumb_count == 4
            and float(non_thumb.mean()) >= 0.78
            and not dorsal_valid
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
        # A side-on thumbs-up can be outside the classifier's learned camera
        # angles even though its silhouette is unambiguous. Keep this recovery
        # deliberately stricter than ordinary Like validation: the thumb must
        # be strongly extended/upward, protrude beyond the compact palm, lead
        # every fingertip, and the other four fingers must all remain folded.
        like_geometry_supported = bool(
            raw in {"like", "fist"}
            and extensions[0] >= .70
            and float(foreshortened_fist["thumb_radius_ratio"]) >= 1.10
            and float(non_thumb.max()) <= .50
            and float(non_thumb.mean()) <= .38
            and thumb_up_score >= .70
            and thumb_lead >= .25
        )
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
        thumb_down_geometry_supported = bool(
            thumb_down_valid
            and extensions[0] >= .70
            and float(non_thumb.mean()) <= .45
            and thumb_down_score >= .70
            and thumb_down_lead >= .25
        )
        # A high model score alone is not evidence of a closed hand. In
        # particular, four partly bent but raised fingers can look like Fist
        # to the MLP. Keep a geometry requirement for every Fist decision.
        fist_compact = bool(
            float(non_thumb.max()) <= .55
            and float(non_thumb.mean()) <= .38
            and extensions[0] <= .85
        )
        # A closed fist can show either its knuckles or the opposite side to
        # the camera. The two observed NCM views below have different signed
        # depth, so depth alone cannot veto a fist. Keep narrow 3-D cones for
        # these views and still require the fingers and thumb to be tucked.
        fist_depth_alignment = None
        fist_side_view_alignment = None
        reported_reverse_alignment = None
        reported_side_alignment = None
        tilted_side_alignment = None
        rejected_view_alignment = None
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
                    # Explicit negative examples take precedence over every
                    # fist path, including a confident model or curl recovery.
                    rejected_view_alignment = max(
                        _mirrored_view_alignment(world_axis_unit, xy, zx)
                        for xy, zx in (
                            (-64.0, 126.0), (-68.0, 50.0), (-20.0, 102.0),
                            # Latest webcam negatives. The first otherwise
                            # overlaps the reverse-facing Fist exception.
                            (-61.0, 142.0), (-64.0, 40.0),
                            # The second HUD's XY/YZ also permits a +140 ZX
                            # reading; retain rejection around that axis too.
                            (-64.0, 140.0),
                        )
                    )
        fist_orientation_settings = self.config.raw.get("fist_orientation") or {}
        maximum_depth_alignment = float(
            fist_orientation_settings.get("maximum_depth_alignment", .45)
        )
        # Permit the 0-20 degree transition just below either horizontal
        # boundary while retaining the downward-fist exclusion.
        horizontal_tolerance_degrees = float(
            fist_orientation_settings.get("horizontal_tolerance_degrees", 20.0)
        )
        fist_xy_allowed = bool(
            palm_up_alignment >=
            -float(np.sin(np.deg2rad(horizontal_tolerance_degrees))) - 1e-6
        )
        reported_reverse_view = bool(
            reported_reverse_alignment is not None
            and reported_reverse_alignment >= float(np.cos(np.deg2rad(12.0)))
        )
        reported_side_view = bool(
            reported_side_alignment is not None
            and reported_side_alignment >= float(np.cos(np.deg2rad(18.0)))
        )
        tilted_side_view = bool(
            tilted_side_alignment is not None
            and tilted_side_alignment >= float(np.cos(np.deg2rad(10.0)))
        )
        rejected_fist_view = bool(
            rejected_view_alignment is not None
            # Keep this narrow: the second negative is close to an earlier
            # accepted raised fist. A six-degree cone tolerates camera jitter
            # without rejecting that positive reference.
            and rejected_view_alignment >= float(np.cos(np.deg2rad(6.0)))
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
        fist_depth_allowed = bool(
            (fist_depth_alignment is not None or not require_fist_depth)
            and (fist_depth_alignment is None
                 or fist_depth_alignment >= -maximum_depth_alignment
                 or reported_reverse_view)
        )
        fist_allowed_direction = bool(
            (fist_xy_allowed or reported_reverse_view or fist_side_view_geometry)
            and fist_depth_allowed
            and not rejected_fist_view
        )
        fist_valid = bool(
            fist_allowed_direction
            and (fist_compact or foreshortened_fist["strong_geometry"]
                 or fist_side_view_geometry or reported_side_fist_geometry)
        )
        fist_geometry_supported = bool(
            fist_allowed_direction
            and (reported_side_fist_geometry or fist_side_view_geometry
                 or foreshortened_fist["strong_geometry"]
                 or (
                     fist_valid
                     and float(non_thumb.max()) <= .42
                     and float(non_thumb.mean()) <= .28
                 ))
        )
        fist_foreshortened_relabel = bool(
            fist_allowed_direction
            and (foreshortened_fist["strong_geometry"] or fist_side_view_geometry
                 or reported_side_fist_geometry)
            # A projected Fist and the two thumb commands can share the same
            # four curled fingertips. Preserve a thumb command only when its
            # thumb geometry actually passes validation.
            and (raw != "like" or not like_valid)
            and (raw != "thumb_down" or not thumb_down_valid)
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
            "open_palm": open_palm_valid,
            "like": like_valid,
            "dorsal": dorsal_valid,
            "ok": ok_valid,
            "fist": fist_valid,
            "thumb_down": thumb_down_valid,
            "peace": peace_geometry_supported,
        }
        # Geometry may safely resolve the two open-hand orientations even when
        # perspective causes the MLP to swap them.
        allow_dorsal_resolution = not (
            raw == "open_palm"
            and not open_palm_allowed_direction
            and not bool(surface["dorsal_visible"])
        )
        # Only the perspective-specific curl-back signature may relabel a
        # different top class. The ordinary compact-fist validator remains a
        # guard for an already predicted Fist; otherwise it can resemble the
        # clustered Down silhouettes covered by the camera regressions.
        if like_geometry_supported:
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx["like"], 0.96
            )
            raw = "like"
        elif fist_foreshortened_relabel:
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx["fist"], 0.96
            )
            raw = "fist"
        elif allow_dorsal_resolution and (
            dorsal_supported
            or (dorsal_valid and raw in {"open_palm", "dorsal", "down"})
        ):
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx["dorsal"], 0.96
            )
            raw = "dorsal"
        elif palm_geometry_supported or (open_palm_valid and raw in {"open_palm", "dorsal"}):
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx["open_palm"], 0.96
            )
            raw = "open_palm"
        valid = validators.get(raw, True)
        if valid:
            reason = "pose geometry accepted"
        elif raw == "dorsal" and not dorsal_allowed_direction:
            reason = "dorsal requires an XY angle from 0 to 150 degrees downward"
        elif raw == "open_palm" and not open_palm_allowed_direction:
            reason = "open_palm must not point downward"
        elif raw == "fist" and require_fist_depth and fist_depth_alignment is None:
            reason = "fist depth landmarks unavailable"
        elif raw == "fist" and rejected_fist_view:
            reason = "fist orientation matches a no-gesture pose"
        elif raw == "fist" and not fist_depth_allowed:
            reason = "fist points away from the camera"
        elif raw == "fist" and not (fist_xy_allowed or reported_reverse_view
                                    or fist_side_view_geometry):
            reason = "fist points downward"
        elif raw == "like" and not like_finger_shape:
            reason = "like requires the thumb to lead and other fingers to stay folded"
        elif raw == "ok" and (closest_other_thumb_gap < .70 or closest_other_thumb_gap - gap < .25):
            reason = "ok requires only the index fingertip to touch the thumb"
        else:
            reason = f"{raw} hand shape failed geometry checks"
        return adjusted, {
            "valid": bool(valid),
            "reason": reason,
            "resolved_gesture": raw,
            "thumb_extension": float(extensions[0]),
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
            "palm_up_alignment": palm_up_alignment,
            "palm_axis_dominance": palm_axis_dominance,
            "dorsal_score": float(dorsal["score"]),
            "dorsal_geometry_supported": dorsal_supported,
            "peace_geometry_supported": peace_geometry_supported,
            "peace_pair_recovery": peace_pair_recovery,
            "dorsal_range_recovery": dorsal_range_recovery,
            "dorsal_xy_degrees": dorsal_xy,
            "dorsal_allowed_direction": dorsal_allowed_direction,
            "peace_model_geometry": peace_model_geometry,
            "peace_image_geometry": peace_image,
            "peace_world_geometry": peace_world,
            "palm_geometry_supported": palm_geometry_supported,
            "sideways_palm_recovery": sideways_palm_recovery,
            "like_geometry_supported": like_geometry_supported,
            "fist_geometry_supported": fist_geometry_supported,
            "fist_allowed_direction": fist_allowed_direction,
            "fist_depth_alignment": fist_depth_alignment,
            "fist_side_view_alignment": fist_side_view_alignment,
            "fist_side_view_geometry": fist_side_view_geometry,
            "fist_reported_reverse_alignment": reported_reverse_alignment,
            "fist_reported_side_alignment": reported_side_alignment,
            "fist_reported_side_geometry": reported_side_fist_geometry,
            "fist_tilted_side_alignment": tilted_side_alignment,
            "fist_rejected_view_alignment": rejected_view_alignment,
            "fist_rejected_view": rejected_fist_view,
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
    ) -> tuple[np.ndarray, dict[str, dict | None]]:
        adjusted, directional = self._directional(probabilities, landmarks)
        adjusted, shape = self._hand_shape(
            adjusted,
            landmarks,
            handedness=handedness,
            handedness_confidence=handedness_confidence,
            world_landmarks=world_landmarks,
            require_fist_depth=require_fist_depth,
        )
        # This tightly bounded single-index silhouette takes precedence over a
        # Dorsal relabel. At this angle the three folded chains can appear
        # parallel and downward even though only the index reaches past them.
        if directional.get("prominent_index_down", False):
            adjusted = _set_probability_floor(
                adjusted, self.config.class_to_idx["down"], .98
            )
        resolved = self.config.class_names[int(np.argmax(adjusted))]
        valid = bool(shape.get("valid", True))
        reason = str(shape.get("reason", "pose geometry accepted"))
        if resolved in {"left", "right", "up", "down"}:
            valid = bool(directional.get("valid", False))
            reason = str(directional.get("reason", "direction geometry accepted"))
        return adjusted, {
            "directional": directional,
            "hand_shape": shape,
            "pose_validation": {
                "valid": valid,
                "reason": reason,
                "gesture": resolved,
                "geometry_supported": bool(
                    (resolved == "dorsal" and shape.get("dorsal_geometry_supported", False))
                    or (resolved == "open_palm" and shape.get("palm_geometry_supported", False))
                    or (resolved == "like" and shape.get("like_geometry_supported", False))
                    or (resolved == "fist" and shape.get("fist_geometry_supported", False))
                    or (resolved == "thumb_down" and shape.get("thumb_down_geometry_supported", False))
                    or (resolved == "peace" and shape.get("peace_geometry_supported", False))
                    or (resolved in {"left", "right", "up", "down"} and directional.get("strong_geometry", False))),
                "sideways_palm_recovery": bool(
                    resolved == "open_palm"
                    and shape.get("sideways_palm_recovery", False)
                ),
                "dorsal_range_recovery": bool(
                    resolved == "dorsal" and shape.get("dorsal_range_recovery", False)
                ),
            },
        }
