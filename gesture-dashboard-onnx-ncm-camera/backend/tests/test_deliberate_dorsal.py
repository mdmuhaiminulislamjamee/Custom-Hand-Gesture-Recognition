"""Explicit Dorsal shape, separate from face position and resting-arm context.

The chin-rest fixture traces the visible screenshot joints; it is not a pixel
replay and cannot establish camera accuracy.
"""
import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.geometry import GeometryResolver, hand_surface_orientation
from backend.runtime import TemporalGate
from backend.hand_selection import gate_arguments
from backend.tests.test_reported_peace_dorsal import reported_dorsal
from backend.tests.test_gesture_orientation import rotate_screen


def chin_rest():
    return np.asarray([
        [574,555], [557,495], [572,435], [584,393], [589,361],
        [649,429], [696,391], [716,374], [734,361],
        [673,452], [721,418], [742,403], [755,387],
        [685,483], [719,462], [735,451], [747,439],
        [699,518], [726,497], [740,479], [748,463],
    ], dtype=float)


@pytest.mark.parametrize("angle", range(-180, 180, 45))
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("scale", [.5, 1., 1.5])
@pytest.mark.parametrize("intentional", [False, True])
def test_deliberate_dorsal_and_chin_rest_across_views(angle, mirror, scale, intentional):
    points = rotate_screen(reported_dorsal() if intentional else chin_rest(), angle, mirror=mirror) * scale
    side = next(side for side in ("Left", "Right")
                if hand_surface_orientation(points, side, .99)["dorsal_visible"])
    config = RuntimeConfig()
    probabilities = np.eye(len(config.class_names))[config.class_to_idx["dorsal"]]
    resolved, details = GeometryResolver(config).resolve(
        probabilities, points, handedness=side, handedness_confidence=.99)
    assert details["hand_shape"]["dorsal_deliberate_shape"] is intentional
    assert details["pose_validation"]["valid"] is intentional
    gate = TemporalGate(config)
    for timestamp in (1., 1.1, 1.2, 1.3, 1.6):
        decision = gate.update(resolved, now=timestamp,
            **gate_arguments(details, dict(known_gesture_mass=1.)))
        if not intentional:
            assert decision.predicted_gesture == "no_gesture" and not decision.execute
    if intentional:
        assert decision.predicted_gesture == "dorsal"


@pytest.mark.parametrize("fault", ["thumb_spread", "fingers_spread", "finger_curled"])
def test_confident_model_cannot_override_loose_dorsal_shape(fault):
    points = reported_dorsal().copy()
    if fault == "thumb_spread":
        points[4] += [100, -30]
    elif fault == "fingers_spread":
        for finger, offset in zip((5, 9, 13, 17), (100, 30, -30, -100)):
            points[finger + 1:finger + 4, 0] += np.linspace(offset / 3, offset, 3)
    else:
        points[12] = points[9]
    config = RuntimeConfig()
    probabilities = np.eye(len(config.class_names))[config.class_to_idx["dorsal"]]
    _, details = GeometryResolver(config).resolve(
        probabilities, points, handedness="Right", handedness_confidence=.99)
    assert not details["hand_shape"]["dorsal_deliberate_shape"]
    assert not details["pose_validation"]["valid"]
