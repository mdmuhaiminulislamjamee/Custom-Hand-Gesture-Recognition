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


def invalid_hand_geometry(points, image_shape, world=None):
    pixels = np.asarray(points) * [image_shape[1], image_shape[0]]
    if not collapsed_finger_chains(pixels):
        return False
    if world is not None:
        world = np.asarray(world)
        if world.shape == (21, 3) and np.isfinite(world).all():
            return collapsed_finger_chains(world)
    return True
