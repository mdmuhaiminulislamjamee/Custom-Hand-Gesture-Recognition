"""Approximate landmark trace of the reported arched-thumb Left screenshot.

This fixture traces the displayed joints; it is not a raw camera observation.
The tests reproduce rejection at the displayed ~0.15 vocabulary mass and check
the narrow recovery without lowering the global unknown-gesture threshold.
"""
import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.geometry import GeometryResolver, arched_thumb_pointing_geometry
from backend.hand_selection import gate_arguments
from backend.runtime import TemporalGate


def reported_pointing(aspect_height=501, mirror=False):
    points = np.asarray([
        [171, 390], [203, 341], [256, 309], [303, 314], [319, 340],
        [279, 355], [351, 352], [392, 351], [425, 351],
        [300, 357], [333, 371], [310, 382], [283, 385],
        [280, 390], [319, 392], [304, 403], [281, 404],
        [275, 435], [301, 422], [291, 411], [279, 416],
    ], dtype=np.float32) / [950, aspect_height]
    if mirror:
        points[:, 0] = 1 - points[:, 0]
    return points


@pytest.mark.parametrize('mirror', [False, True])
@pytest.mark.parametrize('height', [501, 712, 950])
@pytest.mark.parametrize('mass', [.14, .1499, .15, .25, 1.])
def test_compact_arched_pointing_recovers_the_calibrated_direction(height, mirror, mass):
    config = RuntimeConfig()
    points = reported_pointing(height, mirror)
    assert arched_thumb_pointing_geometry(points)
    # A camera-domain classifier miss must still resolve from this strictly
    # single-index silhouette, even if the initial predicted class is Like.
    raw = np.eye(len(config.class_names))[config.class_to_idx['like']]
    p, details = GeometryResolver(config).resolve(raw, points)
    args = gate_arguments(details, dict(known_gesture_mass=mass))
    gate = TemporalGate(config)
    first = gate.update(p, now=1., **args)
    assert not first.execute
    for frame in range(1, 9):
        decision = gate.update(p, now=1 + frame * .1, **args)
    assert decision.execute
    assert decision.predicted_gesture == ('right' if mirror else 'left')


@pytest.mark.parametrize('other', [9, 13, 17])
def test_a_second_extended_finger_cannot_use_arched_thumb_recovery(other):
    points = reported_pointing()
    vector = points[8] - points[5]
    points[other + 1:other + 4] = [points[other] + vector * fraction
                                 for fraction in (.4, .7, 1.)]
    assert not arched_thumb_pointing_geometry(points)


def test_folded_thumb_cannot_use_the_special_low_mass_recovery():
    config = RuntimeConfig()
    points = reported_pointing()
    points[4] = points[5]
    assert not arched_thumb_pointing_geometry(points)
    raw = np.eye(len(config.class_names))[config.class_to_idx['left']]
    p, details = GeometryResolver(config).resolve(raw, points)
    gate = TemporalGate(config)
    for frame in range(12):
        decision = gate.update(p, now=1 + frame * .1,
            **gate_arguments(details, dict(known_gesture_mass=.14)))
        assert not decision.execute
