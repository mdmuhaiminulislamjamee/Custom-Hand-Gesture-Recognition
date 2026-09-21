from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RuntimeConfig


def _unit_direction(vector: np.ndarray, description: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError(f"Degenerate {description} direction.")
    return np.asarray(vector, dtype=np.float32) / norm


def thumb_index_gap_ratio(landmarks_xy: np.ndarray) -> float:
    points = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)
    references = np.asarray([
        np.linalg.norm(points[5] - points[0]),
        np.linalg.norm(points[9] - points[0]),
        np.linalg.norm(points[17] - points[0]),
        np.linalg.norm(points[17] - points[5]),
    ], dtype=np.float32)
    positive = references[references > 1e-8]
    if not len(positive):
        raise ValueError("Degenerate palm scale for thumb-index gap.")
    return float(np.linalg.norm(points[4] - points[8]) / np.median(positive))


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
            "palm_down_alignment": palm_down_alignment,
            "minimum_finger_down_alignment": float(finger_units[:, 1].min()),
            "finger_reach_ratios": finger_reaches.tolist(),
            "fingertip_descent_ratios": fingertip_descent.tolist(),
            "inner_tip_spread_ratio": inner_tip_spread,
            "outer_tip_drop_ratio": outer_tip_drop,
            "outer_reach_advantage": outer_reach_advantage,
            "strong_geometry": bool(
                directional
                and (
                    edge_on_down
                    or curled_edge_down
                    or clustered_edge_down
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
        thumb_vector = points[4] - points[2]
        thumb_norm = max(float(np.linalg.norm(thumb_vector)), 1e-8)
        thumb_up_score = float(-thumb_vector[1] / thumb_norm)
        non_thumb = extensions[1:]
        extended_non_thumb_count = int((non_thumb >= 0.58).sum())
        extended_other_count = int((extensions[2:] >= 0.58).sum())
        gap = thumb_index_gap_ratio(points)
        dorsal = return_main_pose_geometry(points)
        dorsal_supported = dorsal_geometry_support(dorsal)
        foreshortened_fist = foreshortened_fist_geometry(points)
        palm_axis = points[[5, 9, 13, 17]].mean(axis=0) - points[0]
        palm_axis_norm = max(float(np.linalg.norm(palm_axis)), 1e-8)
        palm_up_alignment = float(-palm_axis[1] / palm_axis_norm)
        palm_axis_dominance = float(np.max(np.abs(palm_axis)) / palm_axis_norm)
        open_palm_orientation_settings = (
            self.config.raw.get("open_palm_orientation") or {}
        )
        open_palm_upward = bool(
            palm_up_alignment >= float(
                open_palm_orientation_settings.get("minimum_up_alignment", .70)
            )
            and palm_axis_dominance >= float(
                open_palm_orientation_settings.get("minimum_axis_dominance", .70)
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
        dorsal_supported = bool(
            dorsal_supported
            and (not surface_validation_active or bool(surface["dorsal_visible"]))
        )
        dorsal_valid = bool(
            dorsal["score"] >= 0.76
            and dorsal["downward_finger_count"] >= 4
            and dorsal["minimum_extension_score"] >= 0.50
            and (not surface_validation_active or bool(surface["dorsal_visible"]))
        )
        open_palm_valid = bool(
            extended_non_thumb_count == 4
            and float(non_thumb.mean()) >= 0.78
            and not dorsal_valid
            and open_palm_upward
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
        like_valid = bool(
            extensions[0] >= 0.60
            and float(non_thumb.mean()) <= 0.72
            and thumb_up_score >= 0.55
        )
        # Side-view thumb-up poses have the same projected-finger ambiguity as
        # Left/Right. Require a confident model and a thumb above all fingertips.
        palm_scale = max(float(np.linalg.norm(points[9] - points[0])), 1e-6)
        thumb_lead = float(min(points[tip, 1] - points[4, 1] for tip in (8, 12, 16, 20)) / palm_scale)
        like_valid = like_valid or bool(
            raw == "like" and float(adjusted.max()) >= 0.90
            and extensions[0] >= 0.40 and thumb_up_score >= 0.20
            and thumb_lead >= 0.10
        )
        # A side-on thumbs-up can be outside the classifier's learned camera
        # angles even though its silhouette is unambiguous. Keep this recovery
        # deliberately stricter than ordinary Like validation: the thumb must
        # be strongly extended/upward, lead every fingertip, and the other four
        # fingers must all remain folded.
        like_geometry_supported = bool(
            raw == "like"
            and extensions[0] >= .70
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
        fist_valid = bool(
            float(non_thumb.max()) <= .55
            and float(non_thumb.mean()) <= .38
            and extensions[0] <= .85
        )
        fist_valid = fist_valid or bool(
            raw == "fist" and float(adjusted.max()) >= .85
            and float(non_thumb.mean()) <= .50
        )
        fist_valid = bool(
            fist_valid or foreshortened_fist["strong_geometry"]
        )
        fist_geometry_supported = bool(
            foreshortened_fist["strong_geometry"]
            or (
                fist_valid
                and float(non_thumb.max()) <= .42
                and float(non_thumb.mean()) <= .28
            )
        )
        fist_foreshortened_relabel = bool(
            foreshortened_fist["strong_geometry"]
            # A projected Fist and the two thumb commands can share the same
            # four curled fingertips. Preserve explicit Like/Thumb Down model
            # evidence instead of overriding the only discriminating digit.
            and raw not in {"like", "thumb_down"}
        )
        ok_valid = bool(
            gap <= 0.58
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
        }
        # Geometry may safely resolve the two open-hand orientations even when
        # perspective causes the MLP to swap them.
        allow_dorsal_resolution = not (
            raw == "open_palm"
            and not open_palm_upward
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
        elif raw == "open_palm" and not open_palm_upward:
            reason = "open_palm must point upward"
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
            "thumb_down_score": thumb_down_score,
            "thumb_down_lead_ratio": thumb_down_lead,
            "thumb_index_gap_ratio": gap,
            "extended_non_thumb_count": extended_non_thumb_count,
            "open_palm_upward": open_palm_upward,
            "palm_up_alignment": palm_up_alignment,
            "palm_axis_dominance": palm_axis_dominance,
            "dorsal_score": float(dorsal["score"]),
            "dorsal_geometry_supported": dorsal_supported,
            "palm_geometry_supported": palm_geometry_supported,
            "like_geometry_supported": like_geometry_supported,
            "fist_geometry_supported": fist_geometry_supported,
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
    ) -> tuple[np.ndarray, dict[str, dict | None]]:
        adjusted, directional = self._directional(probabilities, landmarks)
        adjusted, shape = self._hand_shape(
            adjusted,
            landmarks,
            handedness=handedness,
            handedness_confidence=handedness_confidence,
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
                    or (resolved in {"left", "right", "up", "down"} and directional.get("strong_geometry", False))),
            },
        }
