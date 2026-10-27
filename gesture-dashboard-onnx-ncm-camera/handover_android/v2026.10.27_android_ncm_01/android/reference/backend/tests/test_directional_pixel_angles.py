"""Direction sectors describe the unmirrored camera image in pixel units."""
import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.geometry import GeometryResolver
from backend.hand_selection import gate_arguments
from backend.runtime import TemporalGate
from backend.tests.test_multiview_training import directional_points


@pytest.mark.parametrize("label", ["left", "right", "up", "down"])
@pytest.mark.parametrize("tilt", [-38., -30., 0., 30., 38.])
@pytest.mark.parametrize("shape", [(600, 600, 3), (540, 960, 3), (960, 540, 3)])
def test_pixel_direction_is_independent_of_camera_aspect(label, tilt, shape):
    config = RuntimeConfig()
    source = directional_points(label)
    angle = np.deg2rad(tilt)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pixels = (source - source[0]) @ rotation.T * 180 + [shape[1] / 2, shape[0] / 2]
    points = pixels / [shape[1], shape[0]]
    probability = np.eye(len(config.class_names))[config.class_to_idx[label]]
    resolved, details = GeometryResolver(config).resolve(probability, points, image_shape=shape)
    assert details["directional"]["gesture"] == label
    assert details["pose_validation"]["valid"]
    assert details["directional"]["direction_angle_space"] == "source image pixels"
    gate = TemporalGate(config)
    for tick in range(4):
        decision = gate.update(resolved, now=1. + tick * .1,
                               **gate_arguments(details, {"known_gesture_mass": .99}))
    assert decision.execute and decision.predicted_gesture == label
