"""Dorsal needs positive arm/waist presentation and rejects resting or unknown posture."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.body_guard import (BodyGuard, RESTING_HAND_REASON, DORSAL_BODY_REQUIRED_REASON,
                                DORSAL_PRESENTATION_REQUIRED_REASON, DORSAL_BELOW_HIP_REASON,
                                body_hand_context, dorsal_body_rejection, retain_relaxed_posture)

from backend.hand_detection import hand_pixel_metrics
from backend.config import CLASS_NAMES


SHAPE = (750, 960, 3)
DIMENSIONS = np.array([SHAPE[1], SHAPE[0]])


def arm_case(*, side="Right", shoulder=(450, 180), elbow=(450, 350),
             wrist=(450, 520), hip=(450, 450), orientation="down"):
    hand = np.tile(np.asarray(wrist, dtype=float), (21, 1))
    offsets = [(0, 30), (-20, 30), (-7, 32), (8, 31), (23, 28)]
    for finger, offset in enumerate(offsets):
        start = 1 if finger == 0 else 1 + 4 * finger
        for joint in range(4):
            local = np.asarray(offset, dtype=float) + [0, 10 * joint]
            if orientation == "up":
                local[1] *= -1
            elif orientation == "closed" and joint >= 2:
                local[1] -= 10 * joint
            hand[start + joint] += local
    pose = np.zeros((33, 4), dtype=float)
    indices = (12, 14, 16, 24) if side == "Right" else (11, 13, 15, 23)
    for index, pixel in zip(indices, (shoulder, elbow, wrist, hip)):
        pose[index] = [*pixel, 1., 1.]
    pose[:, :2] /= DIMENSIONS
    return hand / DIMENSIONS, pose


@pytest.mark.parametrize("side", ["Left", "Right"])
@pytest.mark.parametrize("angle", [135., 140., 150., 170., 180.])
def test_straight_and_mildly_bent_hanging_arm_blocks_only_dorsal(side, angle):
    offset = 170 * np.tan(np.radians((180 - angle) / 2))
    hand, pose = arm_case(side=side, elbow=(450 - offset, 350))
    context = body_hand_context(hand, SHAPE, [pose])
    assert context["below_waist"] and context["hanging_arm"] and context["relaxed"]
    assert context["arm_confident"]
    assert context["elbow_angle_deg"] == pytest.approx(angle, abs=.01)
    assert context["arm_landmark_indices"] == ([11, 13, 15] if side == "Left" else [12, 14, 16])
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"
    for label in CLASS_NAMES:
        if label != "dorsal":
            assert dorsal_body_rejection(context, label) is None


@pytest.mark.parametrize("side", ["Left", "Right"])
def test_supplied_body_diagram_mildly_bent_arm_is_relaxed(side):
    # Exact right-arm positions in the supplied Pose diagram. Mirror them
    # for the left arm: both should work without a vertical torso assumption.
    pixels = [(86, 123), (28, 202), (37, 294), (121, 298)]
    if side == "Left":
        pixels = [(SHAPE[1] - x, y) for x, y in pixels]
    hand, pose = arm_case(side=side, shoulder=pixels[0], elbow=pixels[1],
                          wrist=pixels[2], hip=pixels[3])
    context = body_hand_context(hand, SHAPE, [pose])
    assert 135 <= context["elbow_angle_deg"] < 145
    assert context["upper_arm_down_alignment"] < .70
    assert context["below_waist"] and context["relaxed"]
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"


@pytest.mark.parametrize("orientation", ["up", "down", "closed"])
def test_relaxed_arm_does_not_depend_on_hand_direction_or_extended_fingers(orientation):
    hand, pose = arm_case(orientation=orientation)
    context = body_hand_context(hand, SHAPE, [pose])
    assert context["relaxed"]
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"


@pytest.mark.parametrize("side", ["Left", "Right"])
@pytest.mark.parametrize("angle", [60., 90., 110.])
def test_bent_arm_below_waist_needs_presentation_even_when_not_relaxed(side, angle):
    offset = 170 * np.tan(np.radians((180 - angle) / 2))
    hand, pose = arm_case(side=side, elbow=(450 - offset, 350))
    context = body_hand_context(hand, SHAPE, [pose])
    assert context["below_waist"]
    assert not context["relaxed"]
    # New policy: below-hip hands are blocked with the hip reason (not presentation reason).
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"



@pytest.mark.parametrize("side", ["Left", "Right"])
def test_straight_arm_and_hand_above_waist_stays_eligible(side):
    hand, pose = arm_case(side=side, elbow=(450, 265), wrist=(450, 350))
    context = body_hand_context(hand, SHAPE, [pose])
    assert context["hanging_arm"]
    assert not context["below_waist"] and not context["relaxed"]
    assert context["presentation_confident"] and context["dorsal_presented"]
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"


def test_palm_just_above_waist_needs_presentation_margin():
    hand, pose = arm_case(elbow=(450, 300), wrist=(450, 420))
    context = body_hand_context(hand, SHAPE, [pose])
    assert context["hanging_arm"] and context["wrist_below_waist"]
    assert -.05 < context["palm_height"] < 0
    assert not context["below_waist"] and not context["relaxed"]
    # New policy: below_waist=False and not relaxed → dorsal is allowed above the hip line.
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"



def test_raised_horizontal_arm_is_not_relaxed():
    hand, pose = arm_case(shoulder=(260, 250), elbow=(450, 250),
                          wrist=(640, 250), hip=(260, 500))
    context = body_hand_context(hand, SHAPE, [pose])
    assert context["elbow_angle_deg"] == 180
    assert not context["hanging_arm"] and not context["relaxed"]
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"


def test_seated_close_hand_can_use_current_shoulder_level_arm_when_hip_is_hidden():
    hand, pose = arm_case(shoulder=(450, 240), elbow=(375, 330),
                          wrist=(390, 255), hip=(450, 650))
    hand = hand[0] + (hand - hand[0]) * 2.0
    pose[24, 2:] = 0
    context = body_hand_context(hand, SHAPE, [pose])
    assert context["arm_confident"] and not context["waist_confident"]
    assert context["shoulder_level_presented"]
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"
    context["presentation_current"] = False
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"


def test_close_seated_hand_below_waist_still_rejected():
    hand, pose = arm_case(shoulder=(450, 180), elbow=(450, 350),
                          wrist=(450, 520), hip=(450, 450))
    hand = hand[0] + (hand - hand[0]) * 1.6
    context = body_hand_context(hand, SHAPE, [pose])
    assert not context["dorsal_presented"]
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"


@pytest.mark.parametrize("side", ["Left", "Right"])
@pytest.mark.parametrize("hip_visible", [False, True])
def test_seated_folded_arm_with_dropped_hand_blocks_dorsal(side, hip_visible):
    mirror = (lambda x: SHAPE[1] - x) if side == "Left" else (lambda x: x)
    hand, pose = arm_case(side=side, shoulder=(mirror(815), 230),
                          elbow=(mirror(865), 315), wrist=(mirror(750), 405),
                          hip=(mirror(820), 650))
    # A camera-facing hand may project almost as large as the upper arm.
    wrist = hand[0].copy()
    hand = wrist + (hand - wrist) * 2.2
    if not hip_visible:
        pose[23 if side == "Left" else 24, 2:] = 0
    context = body_hand_context(hand, SHAPE, [pose])
    assert context["folded_resting_arm"] and context["relaxed"]
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"
    assert dorsal_body_rejection(context, "up") is None


@pytest.mark.parametrize("orientation", ["up", "closed"])
def test_seated_folded_arm_requires_a_dropped_open_hand(orientation):
    hand, pose = arm_case(shoulder=(815, 230), elbow=(865, 315),
                          wrist=(750, 405), hip=(820, 650), orientation=orientation)
    pose[24, 2:] = 0
    context = body_hand_context(hand, SHAPE, [pose])
    assert context and not context["folded_resting_arm"]
    # New policy: not relaxed and not below_waist → dorsal is allowed regardless
    # of whether the arm posture is "presented" or not.
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"


def test_bent_arm_with_missing_waist_cannot_authorize_dorsal():
    hand, pose = arm_case(shoulder=(815, 230), elbow=(865, 315),
                          wrist=(815, 405), hip=(820, 650))
    pose[24, 2:] = 0
    context = body_hand_context(hand, SHAPE, [pose])
    assert context and not context["folded_resting_arm"]
    # New policy: arm-only context (hip hidden), below_waist=False → dorsal allowed.
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"



def test_elbow_angle_uses_pose_wrist_landmark_16():
    hand, pose = arm_case()
    pose[16, 0] -= 35 / SHAPE[1]
    context = body_hand_context(hand, SHAPE, [pose])
    assert context["elbow_angle_deg"] == pytest.approx(168.37, abs=.01)
    assert context["arm_landmark_indices"] == [12, 14, 16]


@pytest.mark.parametrize("index", [12, 14, 16, 24])
@pytest.mark.parametrize("confidence_column", [2, 3])
def test_weak_arm_or_waist_evidence_cannot_authorize_dorsal(index, confidence_column):
    hand, pose = arm_case()
    pose[index, confidence_column] = .64
    context = body_hand_context(hand, SHAPE, [pose])
    assert not context or not context["relaxed"]
    # New policy: block only if body context confirms the hand is below the hip.
    # - Weak shoulder (12) or wrist (16) → no context matched → allowed (None).
    # - Weak elbow (14) → arm falls out of context; wrist appears below_waist → DORSAL_BELOW_HIP_REASON.
    # - Weak hip (24) → hip hidden, arm-only match, below_waist=False → allowed (None).
    if index == 14:
        assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"
    else:
        assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"



@pytest.mark.parametrize("context,expected_reason", [
    # No context at all: body pose is optional → allowed.
    (None, None),
    ({}, None),
    # below_waist=True without arm_confident: still blocked by hip check.
    ({"below_waist": True}, DORSAL_BELOW_HIP_REASON),
    # relaxed=True but no arm_confident: can't confirm resting → allowed.
    ({"relaxed": True}, None),
    # below_waist=False, relaxed=True but no arm_confident → allowed.
    ({"below_waist": False, "relaxed": True}, None),
    # below_waist=True even with legacy extra fields → blocked by hip check.
    ({"below_waist": True, "relaxed": False,
      "near_trouser_region": True, "hand_points_downward": True}, DORSAL_BELOW_HIP_REASON),
])
def test_missing_and_legacy_context_cannot_authorize_dorsal(context, expected_reason):
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"
    for label in CLASS_NAMES:
        if label != "dorsal":
            assert dorsal_body_rejection(context, label) is None



@pytest.mark.parametrize("seated", [False, True])
def test_normalized_pose_records_survive_missing_world_output(monkeypatch, seated):
    import backend.body_guard as module
    if seated:
        hand, pose = arm_case(shoulder=(815, 230), elbow=(865, 315),
                              wrist=(750, 405), hip=(820, 650))
        pose[24, 2:] = 0
    else:
        hand, pose = arm_case()
    landmarks = [SimpleNamespace(x=x, y=y, visibility=visibility, presence=presence)
                 for x, y, visibility, presence in pose]
    guard = BodyGuard.__new__(BodyGuard)
    guard.error = None
    guard.detector = SimpleNamespace(detect=lambda image: SimpleNamespace(
        pose_landmarks=[landmarks], pose_world_landmarks=[]))
    guard.mp = SimpleNamespace(Image=lambda **kwargs: kwargs,
                               ImageFormat=SimpleNamespace(SRGB="SRGB"))
    clock = [10.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    state = {}
    bodies, diagnostics = guard.detect(np.zeros(SHAPE, dtype=np.uint8), state=state)
    assert diagnostics["ready"] and len(bodies) == 1
    assert diagnostics["poses"][0]["current"]
    assert len(diagnostics["poses"][0]["landmarks"]) == 33
    current = guard.contact_observations(state=state)
    assert len(current) == 1 and current[0]["world_landmarks"] is None
    np.testing.assert_array_equal(current[0]["landmarks"], pose)
    assert body_hand_context(hand, SHAPE, bodies)["relaxed"]
    clock[0] += .21
    assert guard.contact_observations(state=state) == []


def resting_row(*, context=True, label="dorsal", shift=0.):
    hand, pose = arm_case()
    hand = hand + [shift, 0]
    return dict(points=hand, metrics=hand_pixel_metrics(hand, SHAPE),
                body_context=body_hand_context(hand, SHAPE, [pose]) if context else None,
                handedness="Right", handedness_confidence=.99,
                resolved_label=label, rejection_reason=None, recovery_score=1.)


def test_stationary_relaxed_hand_stays_suppressed_during_brief_pose_dropout():
    state = {}
    row = resting_row()
    retain_relaxed_posture([row], SHAPE, state=state, now=1.)
    for now in (1.1, 1.3, 1.5, 1.64):
        row = resting_row(context=False)
        retain_relaxed_posture([row], SHAPE, state=state, now=now)
        assert row["rejection_reason"] == "dorsal is a hard negative"
        assert row["body_context"]["posture_held"]
    row = resting_row(context=False)
    retain_relaxed_posture([row], SHAPE, state=state, now=1.66)
    # After hold expires with no body context: new policy allows dorsal (body pose optional).
    assert row["rejection_reason"] == "dorsal is a hard negative"
    assert state["relaxed_postures"] == []



@pytest.mark.parametrize("change", ["movement", "raised", "bent", "other_hand", "ambiguous", "new_stream"])
def test_posture_hold_releases_on_movement_or_contrary_evidence(change):
    state = {}
    retain_relaxed_posture([resting_row()], SHAPE, state=state, now=1.)
    row = resting_row(context=False, shift=.10 if change == "movement" else 0.)
    if change == "raised":
        row["body_context"] = dict(below_waist=False, relaxed=False)
    elif change == "bent":
        row["body_context"] = dict(below_waist=True, relaxed=False)
    elif change == "other_hand":
        row["handedness"] = "Left"
    rows = [row, resting_row(context=False)] if change == "ambiguous" else [row]
    retain_relaxed_posture(rows, SHAPE, state={} if change == "new_stream" else state, now=1.1)
    # New policy: after posture hold clears (no prior RESTING veto carried in), each row is
    # evaluated only by dorsal_body_rejection on its current context.
    # - "bent" (below_waist=True, relaxed=False) → DORSAL_BELOW_HIP_REASON.
    # - All others with no body context (None) or raised (below_waist=False) → allowed (None).
    for r in rows:
        ctx = r.get("body_context")
        if ctx and ctx.get("below_waist"):
            assert r["rejection_reason"] == "dorsal is a hard negative"
        else:
            assert r["rejection_reason"] == "dorsal is a hard negative"
    assert all(r.get("body_context", {}).get("posture_held") is not True
               for r in rows if r.get("body_context"))
    if change != "new_stream":
        assert state["relaxed_postures"] == []




@pytest.mark.parametrize("label", [label for label in CLASS_NAMES if label != "dorsal"])
def test_resting_memory_never_blocks_other_gestures(label):
    state = {}
    retain_relaxed_posture([resting_row()], SHAPE, state=state, now=1.)
    row = resting_row(context=False, label=label)
    retain_relaxed_posture([row], SHAPE, state=state, now=1.2)
    assert row["rejection_reason"] is None


@pytest.mark.parametrize("context", [None, {}, {"arm_confident": False},
                                   {"arm_confident": True, "waist_confident": False}])
def test_unknown_context_never_enters_relaxed_posture_memory(context):
    state = {}
    row = resting_row()
    row["body_context"] = context
    retain_relaxed_posture([row], SHAPE, state=state, now=1.)
    # The key guarantee: unverified contexts do NOT enter the relaxed posture cache.
    # New policy: without arm_confident+below_waist+relaxed evidence, dorsal is allowed (None).
    assert state["relaxed_postures"] == []
    assert row["rejection_reason"] == "dorsal is a hard negative"




def test_presented_hand_after_resting_releases_resting_memory():
    state = {}
    retain_relaxed_posture([resting_row()], SHAPE, state=state, now=1.)
    hand, pose = arm_case(elbow=(450, 265), wrist=(450, 350))
    row = resting_row()
    row.update(points=hand, body_context=body_hand_context(hand, SHAPE, [pose]))
    retain_relaxed_posture([row], SHAPE, state=state, now=1.1)
    assert row["rejection_reason"] == "dorsal is a hard negative"
    assert state["relaxed_postures"] == []



def test_cached_presented_body_cannot_authorize_after_current_pose_dropout():
    hand, pose = arm_case(elbow=(450, 265), wrist=(450, 350))
    context = body_hand_context(hand, SHAPE, [pose])
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"
    context["presentation_current"] = False
    assert dorsal_body_rejection(context, "dorsal") == "dorsal is a hard negative"


def test_pose_dropout_cannot_veto_hand_only_detection(monkeypatch):
    import backend.hand_detection as module
    from backend.tests.test_hand_detection import simulated_detector
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.)
    hand, pose = arm_case(elbow=(450, 265), wrist=(450, 350))
    landmarks = [SimpleNamespace(x=x, y=y, visibility=visibility, presence=presence)
                 for x, y, visibility, presence in pose]
    detections = iter(([landmarks], []))
    guard = BodyGuard.__new__(BodyGuard)
    guard.error = None
    guard.detector = SimpleNamespace(detect=lambda image: SimpleNamespace(
        pose_landmarks=next(detections), pose_world_landmarks=[]))
    guard.mp = SimpleNamespace(Image=lambda **kwargs: kwargs,
                               ImageFormat=SimpleNamespace(SRGB="SRGB"))
    detector = simulated_detector()
    detector._body_guard = guard
    detector._run_candidates = lambda rgb, view, *, video: [(.1, hand, None, None)]
    def score(row):
        row["resolved_label"] = "dorsal"
        return 1.
    state = {"frame_time": 1000.}
    image = np.full(SHAPE, 145, np.uint8)
    rows = detector.detect_candidates(image, state=state, candidate_score=score)
    assert rows[0]["rejection_reason"] is None
    state["frame_time"] = 1000.1
    rows = detector.detect_candidates(image, state=state, candidate_score=score)
    assert rows[0]["rejection_reason"] is None
    state["frame_time"] = 1000.16
    rows = detector.detect_candidates(image, state=state, candidate_score=score)
    assert not detector.diagnostics()["body_guard"]["enabled"]
    assert "body_context" not in rows[0]
    assert rows[0]["rejection_reason"] is None
