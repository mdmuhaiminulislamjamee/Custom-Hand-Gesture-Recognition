"""Continuation is bounded by real observations and cannot create commands."""
import pytest
from backend.direction_stability import DirectionStability, DIRECTIONS


def update(state, label="no_gesture", *, confirmed=False, qualified=False,
           now=1.1, alignment=.70, safe=True, mass=.99, hard=False, raw="left"):
    return state.update(label=label, qualified=qualified, confirmed=confirmed,
        now=now, directional={"continuation_safe": safe, "axis_dominance": .71,
                              "axis_alignment": {name: alignment for name in DIRECTIONS}},
        confidence=.94, margin=.80, known_mass=mass, hard_rejection=hard, raw_label=raw)


@pytest.mark.parametrize("label", sorted(DIRECTIONS))
def test_single_dropout_is_bridged_but_two_consecutive_release(label):
    state = DirectionStability()
    assert update(state, label, confirmed=True, qualified=True, now=1.) is None
    assert update(state, now=1.1, raw=label) == label
    assert update(state, now=1.2, raw=label) is None
    assert state.label is None


@pytest.mark.parametrize("label", sorted(DIRECTIONS))
def test_grace_does_not_refresh_deadline_or_seed_an_unconfirmed_pose(label):
    state = DirectionStability()
    update(state, label, qualified=True, now=1.)
    assert update(state, now=1.1) is None
    update(state, label, qualified=True, confirmed=True, now=2.)
    assert update(state, now=2.201) is None
    assert state.label is None


@pytest.mark.parametrize("label", sorted(DIRECTIONS))
@pytest.mark.parametrize("kwargs", [dict(alignment=-1.), dict(safe=False),
    dict(mass=.1), dict(hard=True), dict(raw="peace")])
def test_opposite_axes_multi_finger_unknown_and_hard_negatives_never_bridge(label, kwargs):
    state = DirectionStability()
    update(state, label, qualified=True, confirmed=True, now=1.)
    assert update(state, **kwargs) is None


@pytest.mark.parametrize("gesture", ["peace", "rock", "like", "fist", "thumb_down",
                                     "open_palm", "stop_inverted", "ok"])
def test_other_qualified_gestures_take_priority_without_added_hold(gesture):
    state = DirectionStability()
    update(state, "left", qualified=True, confirmed=True, now=1.)
    assert update(state, gesture, qualified=True) is None
    assert state.label is None


def test_diagonal_boundary_is_bounded_and_clear_opposite_transition_is_immediate():
    state = DirectionStability()
    update(state, "left", qualified=True, confirmed=True, now=1.)
    assert update(state, "up", qualified=True, confirmed=True, now=1.1) == "left"
    assert update(state, "up", qualified=True, confirmed=True, now=1.2) == "no_gesture"
    assert state.label is None
    update(state, "left", qualified=True, confirmed=True, now=2.)
    assert update(state, "right", qualified=True, confirmed=True, now=2.1, alignment=-1.) is None
    assert state.label == "right"
