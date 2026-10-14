"""Synthetic anatomy cases, including approximate screenshot joints.

These isolate face/torso association; they are not recorded camera replays.
"""
from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.face_contact import (FACE_HAND_REASON, FACE_OVERLAP_REASON, FACE_OVAL,
                                  apply_contact_policy, box_face_observations,
                                  contact_evidence, face_rejection_reason)
from backend.tests.test_face_contact import row_for, scene
from backend.tests.test_hand_detection import simulated_detector

cv2.setNumThreads(1)
SHAPE = (500, 960, 3)


def raised_chest_hand(now=1., mirror=False, fallback=False):
    """Approximate the compact chest-level pointing joints in screenshot 2."""
    hand = np.asarray([
        [420, 162], [438, 155], [462, 146], [483, 137], [493, 137],
        [471, 150], [510, 146], [570, 151], [589, 151],
        [480, 155], [512, 158], [532, 166], [532, 174],
        [480, 160], [510, 170], [502, 182], [485, 186],
        [469, 165], [485, 181], [469, 184], [444, 181],
    ], dtype=float) / [960, 500]
    pose = np.zeros((33, 4))
    side_ids = [12, 14, 16, 24] if mirror else [11, 13, 15, 23]
    pose[side_ids, :2] = np.asarray([[345, 20], [340, 160], [420, 162], [400, 430]]) / [960, 500]
    pose[side_ids, 2:] = 1.
    if fallback:
        faces = box_face_observations([((385, 95, 240, 130), now)], SHAPE)
    else:
        angle = np.linspace(-np.pi / 2, 3 * np.pi / 2, len(FACE_OVAL), endpoint=False)
        mesh = np.tile([505, 160], (478, 1)).astype(float)
        mesh[list(FACE_OVAL)] += np.column_stack((120 * np.cos(angle), 65 * np.sin(angle)))
        faces = [dict(points=mesh / [960, 500], observed_at=now)]
    if mirror:
        hand[:, 0] = 1 - hand[:, 0]
        pose[:, 0] = 1 - pose[:, 0]
        faces[0]["points"][:, 0] = 1 - faces[0]["points"][:, 0]
    bodies = [dict(landmarks=pose, world_landmarks=None, observed_at=now)]
    return hand, faces, bodies


@pytest.mark.parametrize("scale", [.5, 1., 1.5])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_raised_real_hand_clears_false_face_on_shirt_with_cropped_head(scale, mirror, fallback):
    hand, faces, bodies = raised_chest_hand(mirror=mirror, fallback=fallback)
    shape = (round(500 * scale), round(960 * scale), 3)
    # Establish that this would be treated as a face hallucination when the
    # reliable arm/torso evidence is unavailable.
    unsupported, _ = contact_evidence(hand, shape, faces, [], now=1.)
    assert unsupported["face_alias"]
    result, _ = contact_evidence(hand, shape, faces, bodies, now=1.)
    assert result["arm_supported"]
    assert result["ignored_torso_face_count"] == 1
    assert not result["candidate"]
    assert not result["near_face"]
    assert face_rejection_reason(result) is None


@pytest.mark.parametrize("fault", ["stale", "wrist_hidden", "wrong_wrist", "ambiguous", "no_hip"])
def test_false_face_is_not_exempted_without_current_unambiguous_arm(fault):
    hand, faces, bodies = raised_chest_hand()
    if fault == "stale":
        bodies[0]["observed_at"] = .7
    elif fault == "wrist_hidden":
        bodies[0]["landmarks"][15, 2:] = .1
    elif fault == "wrong_wrist":
        bodies[0]["landmarks"][15, 0] += .2
    elif fault == "no_hip":
        bodies[0]["landmarks"][23, 2:] = .1
    else:
        bodies += [deepcopy(bodies[0])]
    result, _ = contact_evidence(hand, SHAPE, faces, bodies, now=1.)
    assert result["candidate"]
    assert face_rejection_reason(result) == FACE_HAND_REASON


def test_arm_support_never_exempts_actual_face_when_head_is_hidden_from_pose():
    hand, faces, bodies = raised_chest_hand()
    pose = bodies[0]["landmarks"]
    # The same touching hand now lies in the real head position, above its
    # shoulder plane; the head landmarks/depth themselves remain unavailable.
    pose[[11, 13, 23], :2] = np.asarray([[450, 280], [350, 310], [470, 490]]) / [960, 500]
    result, _ = contact_evidence(hand, SHAPE, faces, bodies, now=1.)
    assert result["arm_supported"]
    assert result["candidate"]
    assert result["near_face"]
    assert not result.get("ignored_torso_face_count")


def test_background_pose_head_protects_real_face_over_foreground_torso():
    hand, faces, bodies = raised_chest_hand()
    background = np.zeros((33, 4))
    background[:11, :2] = np.asarray([
        [505, 160], [470, 140], [465, 140], [460, 140],
        [540, 140], [545, 140], [550, 140], [400, 160], [610, 160],
        [485, 185], [525, 185],
    ]) / [960, 500]
    background[:11, 2:] = 1.
    bodies.append(dict(landmarks=background, world_landmarks=None, observed_at=1.))
    result, _ = contact_evidence(hand, SHAPE, faces, bodies, now=1.)
    assert result["arm_supported"]
    assert result["candidate"]
    assert not result.get("ignored_torso_face_count")


@pytest.mark.parametrize("label", ["dorsal", "left", "fist", "like"])
def test_policy_leaves_chest_hand_eligible_but_real_face_overlap_blocked(label):
    state = {}
    for index in range(5):
        now = 1 + .1 * index
        hand, faces, bodies = raised_chest_hand(now)
        row = row_for(hand, SHAPE, label)
        apply_contact_policy([row], SHAPE, faces, bodies, state=state, now=now)
        assert row["rejection_reason"] is None
        assert not row["face_context"]["suppressed"]
    hand, faces, bodies = scene(now=2.)
    bodies[0]["world_landmarks"] = None
    row = row_for(hand, label=label)
    apply_contact_policy([row], (750, 960, 3), faces, bodies, state={}, now=2.)
    assert row["face_context"]["suppressed"]
    assert row["rejection_reason"] in {FACE_HAND_REASON, FACE_OVERLAP_REASON}


def test_torso_exception_does_not_hide_another_actual_face_region():
    hand, false_faces, bodies = raised_chest_hand()
    actual_face = deepcopy(false_faces[0])
    # Add a separate face whose nose remains above the reliable shoulder
    # plane but whose lower surface overlaps the fingertips.
    actual_face["points"][:, 1] -= .25
    actual_face["points"][list(FACE_OVAL), 1] += np.linspace(0., .30, len(FACE_OVAL))
    result, _ = contact_evidence(hand, SHAPE, false_faces + [actual_face], bodies, now=1.)
    assert result["ignored_torso_face_count"] == 1
    assert result["candidate"]


def test_live_detector_uses_current_pose_without_face_mesh(monkeypatch):
    import backend.body_guard as body_module
    import backend.hand_detection as hand_module

    clock = [1.]
    monkeypatch.setattr(body_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(hand_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(hand_module.time, "perf_counter", lambda: 10.)
    hand, _, bodies = raised_chest_hand()
    pose_result = SimpleNamespace(
        pose_landmarks=[[SimpleNamespace(x=x, y=y, visibility=v, presence=p)
                         for x, y, v, p in bodies[0]["landmarks"]]],
        pose_world_landmarks=[])
    body_guard = body_module.BodyGuard.__new__(body_module.BodyGuard)
    body_guard.error = None
    body_guard.mp = SimpleNamespace(Image=lambda **kwargs: None,
                                    ImageFormat=SimpleNamespace(SRGB=1))
    body_guard.detector = SimpleNamespace(detect=lambda image: pose_result)
    detector = simulated_detector()
    detector._body_guard = body_guard
    def unexpected_face_detection(rgb, *, state):
        raise AssertionError("live detection must not load face mesh")

    detector._face_mesh = SimpleNamespace(detector=True, detect=unexpected_face_detection)
    detector._run_candidates = lambda rgb, view, *, video: [(.1, hand, None, None)]
    frame, state, scored = np.full(SHAPE, 145, np.uint8), {}, []

    def score(row):
        scored.append(True)
        row["resolved_label"] = "dorsal"
        return 1.

    for index in range(4):
        clock[0] = 1 + .2 * index
        rows = detector.detect_candidates(frame, state=state, candidate_score=score)
        assert len(rows) == 1
        assert rows[0]["rejection_reason"] is None
        assert not rows[0]["face_context"]["enabled"]
        assert not rows[0]["face_context"]["contact_confirmed"]
        assert rows[0]["body_context"] is not None
        assert not rows[0]["face_context"]["suppressed"]
        current = body_guard.contact_observations(state=state)
        assert len(current) == 1
        assert current[0]["world_landmarks"] is None
        assert current[0]["observed_at"] == clock[0]
        assert detector.diagnostics()["face_contact_rejected_count"] == 0
    assert scored
