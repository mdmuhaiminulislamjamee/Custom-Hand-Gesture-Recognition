"""Conservative source-pixel checks before a candidate enters gesture tracking."""
import numpy as np


NON_HAND_REASONS = frozenset({
    "insufficient_hand_pixels", "incomplete_hand", "invalid_hand_geometry",
    "face_detected_as_hand", "face_context_unavailable",
})


def collapsed_finger_chains(points):
    """Reject collapsed joints, without assuming a gesture or finger direction.

    A coherent 3-D observation can rescue an edge-on 2-D projection. This is
    deliberately only a degeneracy check, not proof that the image is a hand.
    """
    points = np.asarray(points, dtype=float)
    palm = np.median(np.linalg.norm(points[[5, 9, 13, 17]] - points[0], axis=1))
    if not np.isfinite(palm) or palm < 1e-6:
        return True
    lengths = [np.linalg.norm(np.diff(points[start:start + 4], axis=0), axis=1).sum()
               for start in (5, 9, 13, 17)]
    return np.count_nonzero(np.asarray(lengths) < .12 * palm) >= 3


def squashed_knuckle_span(points):
    """Reject collapsed metacarpal span in 3D (e.g. ear ridges, neck creases).

    In real human hands, the 3D metric distance between index knuckle (5) and
    pinky knuckle (17) is rigidly bounded by the metacarpal bones: >= 0.35x palm length.
    When a hand detector hallucinates on an ear or skin fold, all knuckles are
    regressed along a narrow vertical curve (< 0.28x palm length).
    """
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 21:
        return False
    palm = float(np.linalg.norm(points[9] - points[0]))
    knuckles = float(np.linalg.norm(points[5] - points[17]))
    if not np.isfinite(palm) or palm < 1e-6:
        return True
    return (knuckles / palm) < 0.28


def invalid_hand_geometry(points, image_shape, world=None):
    pixels = np.asarray(points) * [image_shape[1], image_shape[0]]
    if world is not None:
        world = np.asarray(world, dtype=float)
        if world.shape == (21, 3) and np.isfinite(world).all():
            if squashed_knuckle_span(world):
                return True
            if collapsed_finger_chains(pixels):
                return collapsed_finger_chains(world)
            return False
    return collapsed_finger_chains(pixels)
