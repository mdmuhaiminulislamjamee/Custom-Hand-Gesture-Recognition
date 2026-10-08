from backend.follow_object import FOLLOW_SEQUENCE, FollowObjectStateMachine


def test_object_tracking_is_opt_in_and_uses_only_open_palm():
    machine = FollowObjectStateMachine(timeout_seconds=4.0, hold_seconds=0.2)
    assert machine.observe("open_palm", stable=True, now=0.0).state == "inactive"
    started = machine.start(now=0.0)
    assert started.expected_gesture == "open_palm"
    assert FOLLOW_SEQUENCE == ("open_palm",)

    machine.observe("open_palm", stable=True, now=0.0)
    complete = machine.observe("open_palm", stable=True, now=0.21)
    assert complete.completed is True
    assert complete.message == "Object tracking enabled"
    assert machine.state == "inactive"


def test_wrong_pose_waits_for_open_palm():
    machine = FollowObjectStateMachine(timeout_seconds=4.0, hold_seconds=0.0)
    machine.start(now=0.0)
    update = machine.observe("down", stable=True, now=0.4)
    assert update.reset is False
    assert update.state == "wait_open_palm"
    assert update.expected_gesture == "open_palm"


def test_session_timeout_closes_tracking_procedure():
    machine = FollowObjectStateMachine(
        timeout_seconds=2.0, hold_seconds=1.0, session_timeout_seconds=2.0
    )
    machine.start(now=0.0)
    update = machine.observe("open_palm", stable=False, now=3.0)
    assert update.reset is True
    assert update.active is False
