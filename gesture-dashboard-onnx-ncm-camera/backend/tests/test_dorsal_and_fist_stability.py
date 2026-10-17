"""Tests verifying stable detection of Dorsal and Fist gestures across views and both hands."""

import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.geometry import GeometryResolver, hand_surface_orientation
from backend.tests.test_geometry import fist_at_xy_angle
from backend.tests.test_reported_peace_dorsal import reported_dorsal


def image_1_dorsal_pose(mirror: bool = False):
    """Synthetic landmark trace modeled after Image 1.

    In Image 1:
    - Wrist is at top.
    - Four fingers (index, middle, ring, pinky) point downward together.
    - Thumb is resting naturally to the side (gap ratio ~0.6-0.7).
    - Back of hand faces the camera.
    """
    points = reported_dorsal().copy()
    # Move thumb outward slightly to match the relaxed thumb in Image 1
    points[1:5, 0] += np.linspace(10, 35, 4)
    if mirror:
        points[:, 0] = 1000.0 - points[:, 0]
    return points


@pytest.mark.parametrize("handedness,mirror", [
    ("Right", False),
    ("Left", True),
    (None, False),
])
def test_image_1_dorsal_is_always_negative(handedness, mirror):
    config = RuntimeConfig()
    points = image_1_dorsal_pose(mirror=mirror)
    row = np.zeros(len(config.class_names))
    row[config.class_to_idx["dorsal"]] = 0.98

    resolver = GeometryResolver(config)
    resolved, details = resolver.resolve(
        row, points, handedness=handedness, handedness_confidence=0.99
    )

    pose = details["pose_validation"]
    assert details["hand_shape"]["dorsal_deliberate_shape"] is True
    assert pose["valid"] is False
    assert pose["gesture"] == "no_gesture"
    assert pose["reason"] == "dorsal is a hard negative"



@pytest.mark.parametrize("handedness", ["Right", "Left", None])
@pytest.mark.parametrize("mirror", [False, True])
def test_image_2_fist_detected_for_both_hands(handedness, mirror):
    """Image 2 HUD angles: XY: -80deg, YZ: -174deg, ZX: +123deg.

    Fist alignment is ~ -0.11. Must be accepted for both Right and Left hands.
    """
    config = RuntimeConfig()
    depth_ratio = float(np.cos(np.deg2rad(-80)) / np.tan(np.deg2rad(123)))
    points, world = fist_at_xy_angle(-80, depth_ratio, mirror=mirror)

    row = np.zeros(len(config.class_names))
    row[config.class_to_idx["fist"]] = 0.995
    row[config.class_to_idx["left"]] = 0.005

    resolver = GeometryResolver(config)
    resolved, details = resolver.resolve(
        row, points, handedness=handedness, handedness_confidence=0.99,
        world_landmarks=world, require_fist_depth=True,
    )

    pose = details["pose_validation"]
    assert details["hand_shape"]["fist_depth_allowed"] is True
    assert details["hand_shape"]["fist_allowed_direction"] is True
    assert pose["valid"] is True
    assert pose["gesture"] == "fist"
    assert pose["reason"] == "pose geometry accepted"


@pytest.mark.parametrize("handedness", ["Right", "Left"])
@pytest.mark.parametrize("angle_deg,angle_desc", [
    (-90, "up"),
    (-135, "up to left"),
    (-45, "up to right"),
    (0, "horizontal right"),
    (180, "horizontal left"),
])
def test_fist_accepted_angles_for_both_hands(handedness, angle_deg, angle_desc):
    config = RuntimeConfig()
    points, world = fist_at_xy_angle(angle_deg, 0.17)
    row = np.zeros(len(config.class_names))
    row[config.class_to_idx["fist"]] = 0.995

    resolver = GeometryResolver(config)
    _, details = resolver.resolve(
        row, points, handedness=handedness, handedness_confidence=0.99,
        world_landmarks=world, require_fist_depth=True,
    )

    pose = details["pose_validation"]
    assert details["hand_shape"]["fist_allowed_direction"] is True, f"Failed for {angle_desc} ({angle_deg} deg)"
    assert pose["valid"] is True
    assert pose["gesture"] == "fist"


@pytest.mark.parametrize("handedness", ["Right", "Left"])
@pytest.mark.parametrize("angle_deg,angle_desc", [
    (28, "right to down"),
    (45, "right to down diagonal"),
    (135, "left to down diagonal"),
    (152, "left to down"),
])
def test_fist_rejects_diagonal_down_angles_for_both_hands(handedness, angle_deg, angle_desc):
    config = RuntimeConfig()
    points, world = fist_at_xy_angle(angle_deg, 0.17)
    row = np.zeros(len(config.class_names))
    row[config.class_to_idx["fist"]] = 0.995

    resolver = GeometryResolver(config)
    _, details = resolver.resolve(
        row, points, handedness=handedness, handedness_confidence=0.99,
        world_landmarks=world, require_fist_depth=True,
    )

    pose = details["pose_validation"]
    assert details["hand_shape"]["fist_allowed_direction"] is False, f"Failed for {angle_desc} ({angle_deg} deg)"
    assert pose["valid"] is False
    assert pose["gesture"] == "fist"

