"""Regression for removal of the former trouser-area restriction."""
import pytest

from backend.body_guard import (RESTING_HAND_REASON, DORSAL_BODY_REQUIRED_REASON,
                                body_hand_context, dorsal_body_rejection)
from backend.config import RuntimeConfig
from backend.tests.test_body_context_policy import SHAPE, standing_case


@pytest.mark.parametrize("label", RuntimeConfig().class_names)
def test_trouser_position_never_blocks_a_non_dorsal_gesture(label):
    hand, pose = standing_case()
    context = body_hand_context(hand, SHAPE, [pose])
    assert context["below_waist"] and context["relaxed"]
    assert dorsal_body_rejection(context, label) == ("dorsal is a hard negative" if label == "dorsal" else None)


@pytest.mark.parametrize("height", [240, 300, 380])
def test_raised_dorsal_has_no_expanded_trouser_buffer(height):
    hand, pose = standing_case(hand_wrist_y=height)
    context = body_hand_context(hand, SHAPE, [pose])
    assert not context["relaxed"]
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"


def test_legacy_trouser_flags_do_not_prove_dorsal_presentation_or_reject_other_gestures():
    old = dict(near_trouser_region=True, hand_points_downward=True,
               below_waist=False, relaxed=False)
    # Under the new policy, below_waist=False and not relaxed is allowed for all gestures.
    for label in RuntimeConfig().class_names:
        assert dorsal_body_rejection(old, label) == ("dorsal is a hard negative" if label == "dorsal" else None)
