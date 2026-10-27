"""Synthetic spatial/temporal cases; these do not establish camera accuracy."""
from copy import deepcopy
from types import SimpleNamespace
import threading

import cv2
import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.face_contact import (CONTACT_REASON, CONTACT_PENDING_REASON, FACE_HAND_REASON,
                                  FACE_OVERLAP_REASON, FACE_HOLES, FACE_OVAL, FaceMeshGuard,
                                  apply_contact_policy, box_face_observations, contact_evidence,
                                  face_rejection_reason)
from backend.hand_detection import hand_pixel_metrics
from backend.model_runtime import InferenceEngine, RuntimeSession
from backend.tests.test_camera_stability import representative_hand
from backend.tests.test_hand_detection import simulated_detector

SHAPE = (750, 960, 3)


def scene(now=1., depth=0., offset=0., right=False):
    hand = (representative_hand() - .5) * .45 + [.5 + offset, .4]
    mesh = np.tile([.5, .35], (478, 1))
    angle = np.linspace(-np.pi / 2, 3 * np.pi / 2, len(FACE_OVAL), endpoint=False)
    mesh[list(FACE_OVAL)] = np.column_stack((.5 + .22 * np.cos(angle), .35 + .30 * np.sin(angle)))
    for ids, center in zip(FACE_HOLES, ([.42, .24], [.58, .24], [.5, .46])):
        angle = np.linspace(0, 2 * np.pi, len(ids), endpoint=False)
        mesh[list(ids)] = np.asarray(center) + np.column_stack((.025 * np.cos(angle), .01 * np.sin(angle)))
    mesh[1] = [.5, .35]
    pose = np.zeros((33, 4))
    world = np.zeros((33, 3))
    pose[:11, :2] = [[.5, .35], [.45, .28], [.44, .28], [.43, .28],
                    [.55, .28], [.56, .28], [.57, .28], [.3, .35], [.7, .35],
                    [.46, .45], [.54, .45]]
    pose[:11, 2:] = 1
    world[:11, :2] = (pose[:11, :2] - [.5, .35]) * .4
    ids = [16, 20, 18, 22] if right else [15, 19, 17, 21]
    pose[ids, :2] = hand[[0, 8, 20, 4]]
    pose[ids, 2:] = 1
    world[ids, :2] = (pose[ids, :2] - [.5, .35]) * .4
    world[ids, 2] = depth
    return hand, [dict(points=mesh, observed_at=now)], [dict(
        landmarks=pose, world_landmarks=world, observed_at=now)]


def row_for(hand, shape=SHAPE, label="like"):
    config = RuntimeConfig()
    return dict(points=hand, metrics=hand_pixel_metrics(hand, shape), handedness="Left",
                handedness_confidence=.99, rejection_reason=None, recovery_score=1.,
                probabilities=np.eye(len(config.class_names))[config.class_to_idx[label]],
                gate_args=dict(known_gesture_mass=1.))


@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("scale", [.5, 1., 1.5])
@pytest.mark.parametrize("depth,expected", [(0., True), (.12, False), (-.12, False)])
def test_mesh_overlap_requires_common_pose_depth(depth, expected, scale, right):
    hand, faces, bodies = scene(depth=depth, right=right)
    # Global translations do not change agreement within the one world frame.
    bodies[0]["world_landmarks"] += [1.5, -.7, 2.]
    result, _ = contact_evidence(hand, (750 * scale, 960 * scale, 3), faces, bodies, now=1.)
    assert result["near_face"]
    assert result["depth_available"]
    assert result["candidate"] is expected


@pytest.mark.parametrize("fault", ["no_mesh", "no_pose", "stale_mesh", "stale_pose", "hidden_wrist",
                                   "wrong_person", "nan", "ambiguous"])
def test_incomplete_or_ambiguous_evidence_never_confirms_contact(fault):
    hand, faces, bodies = scene()
    if fault == "no_mesh":
        faces = []
    elif fault == "no_pose":
        bodies = []
    elif fault == "stale_mesh":
        faces[0]["observed_at"] = .4
    elif fault == "stale_pose":
        bodies[0]["observed_at"] = .7
    elif fault == "hidden_wrist":
        bodies[0]["landmarks"][15, 2] = .1
    elif fault == "wrong_person":
        bodies[0]["landmarks"][0, 0] += .4
    elif fault == "nan":
        bodies[0]["world_landmarks"][19] = np.nan
    else:
        bodies.append(deepcopy(bodies[0]))
    result, _ = contact_evidence(hand, SHAPE, faces, bodies, now=1.)
    assert not result["depth_agrees"]
    assert result["candidate"] == (fault not in {"no_mesh", "stale_mesh"})


def test_hand_beside_face_is_clear_even_with_matching_depth():
    hand, faces, bodies = scene(offset=-.38)
    result, _ = contact_evidence(hand, SHAPE, faces, bodies, now=1.)
    assert not result["near_face"]
    assert not result["candidate"]


@pytest.mark.parametrize("depth,blocked", [(0., True), (-.12, False), (.12, False)])
def test_confident_coarse_pose_matches_spread_palm_and_turned_head(depth, blocked):
    # Live Open Palm capture: Pose's nose was 0.412 face widths from the
    # mesh nose, and its median finger error was 1.03 palm lengths. These
    # coarse, confident anchors must still provide shared-world evidence.
    hand, faces, bodies = scene(depth=depth)
    pose = bodies[0]["landmarks"]
    dimensions = np.asarray([SHAPE[1], SHAPE[0]])
    pixels = hand * dimensions
    palm_scale = np.median([np.linalg.norm(pixels[a] - pixels[b])
                           for a, b in ((0, 5), (0, 9), (0, 17), (5, 17))])
    face_width = np.linalg.norm((faces[0]["points"][234] - faces[0]["points"][454]) * dimensions)
    pose[:11, 0] += .412 * face_width / dimensions[0]
    pose[[19, 17, 21], 1] += 1.10 * palm_scale / dimensions[1]
    row = row_for(hand, label="open_palm")
    apply_contact_policy([row], SHAPE, faces, bodies, state={}, now=1.)
    assert row["face_context"]["depth_available"]
    assert row["face_context"]["suppressed"] is blocked
    assert row["rejection_reason"] == (CONTACT_PENDING_REASON if blocked else None)


@pytest.mark.parametrize("fault", ["weak_anchors", "distant_wrist", "distant_nose", "incoherent_head"])
def test_coarse_pose_tolerance_does_not_certify_unreliable_separation(fault):
    hand, faces, bodies = scene(depth=-.12)
    pose = bodies[0]["landmarks"]
    dimensions = np.asarray([SHAPE[1], SHAPE[0]])
    face_width = np.linalg.norm((faces[0]["points"][234] - faces[0]["points"][454]) * dimensions)
    pose[:11, 0] += .412 * face_width / dimensions[0]
    if fault == "weak_anchors":
        pose[[15, 19, 17, 21], 2:] = .65
    elif fault == "distant_wrist":
        pose[15, 0] += .4
    elif fault == "distant_nose":
        pose[0, 0] += .20 * face_width / dimensions[0]
    else:
        pose[1:11, 0] += .8
    evidence, _ = contact_evidence(hand, SHAPE, faces, bodies, now=1.)
    assert not evidence["depth_available"]
    # The small synthetic hand can also trigger the independent face-alias
    # veto; neither reason may certify separation from unreliable Pose data.
    assert face_rejection_reason(evidence) in {FACE_OVERLAP_REASON, FACE_HAND_REASON}


def test_eye_and_mouth_occlusion_do_not_open_a_hole_in_the_face_guard():
    hand, faces, bodies = scene()
    # Make the only overlapping point the index tip, first on cheek then in
    # an eye and the inner mouth. Other joints are beside the face.
    hand[:, 0] -= .4
    for position, expected in (([.39, .36], True), ([.42, .24], True), ([.5, .46], True)):
        hand[8] = position
        result, _ = contact_evidence(hand, SHAPE, faces, bodies, now=1.)
        assert result["near_face"] is expected


def run_policy(state, now, *, depth=0., offset=0., faces_available=True, shape=SHAPE, label="like"):
    hand, faces, bodies = scene(now, depth, offset)
    row = row_for(hand, shape, label)
    apply_contact_policy([row], shape, faces if faces_available else [], bodies, state=state, now=now)
    return row


def test_confirmation_release_and_reacquisition_are_elapsed_time_based():
    state = {}
    assert run_policy(state, 1.)["rejection_reason"] == CONTACT_PENDING_REASON
    assert run_policy(state, 1.1)["rejection_reason"] == CONTACT_PENDING_REASON
    assert run_policy(state, 1.2)["rejection_reason"] == CONTACT_REASON
    held = run_policy(state, 1.3, depth=.12)
    assert held["rejection_reason"] == CONTACT_REASON
    assert held["face_context"]["release_hold"]
    assert run_policy(state, 1.36, depth=.12)["rejection_reason"] is None
    assert run_policy(state, 1.4)["rejection_reason"] == CONTACT_PENDING_REASON
    assert run_policy(state, 1.5)["rejection_reason"] == CONTACT_PENDING_REASON
    assert run_policy(state, 1.6)["rejection_reason"] == CONTACT_REASON


def test_quick_pass_and_moving_hand_pause_without_confirming_contact():
    state = {}
    for index in range(7):
        row = run_policy(state, 1 + index * .05, offset=index * .02)
        assert not row["face_context"]["contact_confirmed"]
    state = {}
    for now in (1., 1.05, 1.1):
        assert not run_policy(state, now)["face_context"]["contact_confirmed"]
    assert run_policy(state, 1.15, depth=.12)["face_context"]["release_hold"]
    assert run_policy(state, 1.26, depth=.12)["rejection_reason"] is None


def test_cached_model_result_and_crop_retries_do_not_accumulate_confirmation():
    state = {}
    hand, faces, bodies = scene()
    for now in (1., 1., 1.05, 1.1, 1.19, 1.3):
        row = row_for(hand)
        apply_contact_policy([row], SHAPE, faces, bodies, state=state, now=now)
        assert not row["face_context"]["contact_confirmed"]


def test_pose_refresh_cadence_can_confirm_after_200ms_with_two_observations():
    state = {}
    for now, observed_at in ((1., 1.), (1.08, 1.), (1.16, 1.16), (1.24, 1.16)):
        hand, faces, bodies = scene(observed_at)
        row = row_for(hand)
        apply_contact_policy([row], SHAPE, faces, bodies, state=state, now=now)
        assert row["face_context"]["contact_confirmed"] == (now >= 1.2)
        assert row["face_context"]["suppressed"]


def test_brief_missing_hand_retains_contact_and_missing_mesh_releases_it():
    state = {}
    for now in (1., 1.1, 1.2):
        run_policy(state, now)
    apply_contact_policy([], SHAPE, [], [], state=state, now=1.25)
    assert run_policy(state, 1.3)["rejection_reason"] == CONTACT_REASON
    assert run_policy(state, 1.4, faces_available=False)["rejection_reason"] == CONTACT_REASON
    assert run_policy(state, 1.46, faces_available=False)["rejection_reason"] is None


def test_session_resolution_and_long_gap_reset_contact():
    state = {}
    for now in (1., 1.1, 1.2):
        row = run_policy(state, now)
    assert row["rejection_reason"] == CONTACT_REASON
    assert not run_policy({}, 1.3)["face_context"]["contact_confirmed"]
    assert not run_policy(state, 1.3, shape=(375, 480, 3))["face_context"]["contact_confirmed"]
    for now in (2., 2.1, 2.2):
        run_policy(state, now)
    assert not run_policy(state, 3.)["face_context"]["contact_confirmed"]


@pytest.mark.parametrize("label", RuntimeConfig().class_names)
def test_contact_stops_all_gestures_and_withdrawal_requires_new_hold(label):
    config = RuntimeConfig()
    session = RuntimeSession(config)
    state = {}
    # First qualify an overlapping hand that is separated in depth.
    for i in range(8):
        row = run_policy(state, 1 + i * .1, depth=.12, label=label)
        chosen, _, _ = session.hand_selector.update([row], SHAPE, 1 + i * .1)
    assert chosen is not None
    for now in (1.8, 1.9, 2.):
        row = run_policy(state, now, label=label)
        chosen, selection, previews = session.hand_selector.update([row], SHAPE, now)
    assert chosen is None
    assert selection["reason"] == CONTACT_REASON
    assert previews[0]["gesture"] == "no_gesture"
    for i in range(6):
        now = 2.1 + i * .1
        row = run_policy(state, now, depth=.12, label=label)
        chosen, _, previews = session.hand_selector.update([row], SHAPE, now)
        if i < 3:
            assert not previews[0]["eligible"]
    assert chosen is not None


def test_other_hand_does_not_inherit_contact():
    state = {}
    for now in (1., 1.1, 1.2):
        hand, faces, bodies = scene(now)
        touch = row_for(hand)
        clear = row_for(hand + [.35, 0])
        clear["handedness"] = "Right"
        rows = [touch, clear] if now != 1.1 else [clear, touch]
        apply_contact_policy(rows, SHAPE, faces, bodies, state=state, now=now)
    assert touch["rejection_reason"] == CONTACT_REASON
    assert clear["rejection_reason"] is None


def test_live_detector_ignores_legacy_face_contact_objects(monkeypatch):
    import backend.hand_detection as detection
    clock = [1.]
    monkeypatch.setattr(detection.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(detection.time, "perf_counter", lambda: 10.)
    detector = simulated_detector()
    detector._face_mesh = SimpleNamespace(detector=True, detect=lambda rgb, *, state: (
        scene(clock[0])[1], dict(ready=True, face_count=1)))
    detector._body_guard = SimpleNamespace(detector=True,
        detect=lambda rgb, *, state: ([], dict(ready=True, body_count=0)),
        contact_observations=lambda *, state: scene(clock[0])[2])
    detector._run_candidates = lambda rgb, view, *, video: [(.1, scene(clock[0])[0], None, None)]
    config = RuntimeConfig()
    session = RuntimeSession(config)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config, engine._lock, engine.detector = config, threading.Lock(), detector
    probabilities = row_for(scene()[0])["probabilities"]
    engine.model_manager = SimpleNamespace(ready=True,
        models={"ONNX": object()}, resolve_name=lambda name: "ONNX",
        predict_many=lambda *args: {"ONNX": (probabilities, .01, dict(known_gesture_mass=1.))},
        predict_detailed=lambda *args: (probabilities, None, None, dict(known_gesture_mass=1.)))
    engine.resolver = SimpleNamespace(resolve=lambda *args, **kwargs: (
        probabilities, {"pose_validation": {"valid": True}}))
    encoded = cv2.imencode(".jpg", np.full(SHAPE, 145, np.uint8))[1].tobytes()
    # Seed spatial confirmation without requiring a fully initialized ONNX
    # session; the final call goes through the real camera result path.
    for now in (1., 1.1):
        clock[0] = now
        detector.detect_candidates(np.full(SHAPE, 145, np.uint8),
            state=session.hand_selector.detector_state, candidate_score=lambda row: 1.)
    clock[0] = 1.2
    output = engine.process_frame(encoded, session)
    assert output["runtime_prediction"] == "like"
    assert detector.diagnostics()["face_contact_rejected_count"] == 0
    assert detector.diagnostics()["face_contact"]["enabled"] is False


def test_face_mesh_cache_is_per_session_and_drops_missing_or_failed_results(monkeypatch):
    import backend.face_contact as module
    clock = [1.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    mesh = scene()[1][0]["points"]
    result = SimpleNamespace(face_landmarks=[[SimpleNamespace(x=x, y=y) for x, y in mesh]])
    guard = FaceMeshGuard.__new__(FaceMeshGuard)
    guard.error = None
    guard.mp = SimpleNamespace(Image=lambda **kwargs: None, ImageFormat=SimpleNamespace(SRGB=1))
    guard.detector = SimpleNamespace(detect=lambda image: result)
    frame, state = np.zeros(SHAPE, np.uint8), {}
    assert len(guard.detect(frame, state=state)[0]) == 1
    result.face_landmarks = []
    assert guard.detect(frame, state={})[0] == []
    clock[0] = 1.15
    held, _ = guard.detect(frame, state=state)
    assert held[0]["observed_at"] == 1.
    def failed(image):
        raise RuntimeError("test failure")
    guard.detector.detect = failed
    clock[0] = 1.3
    faces, diagnostics = guard.detect(frame, state=state)
    assert faces[0]["observed_at"] == 1.
    assert not diagnostics["ready"]
    clock[0] = 1.51
    assert guard.detect(frame, state=state)[0] == []


def test_body_contact_world_observations_ignore_hip_visibility_and_expire(monkeypatch):
    import backend.body_guard as module
    clock = [1.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    record = scene()[2][0]
    result = SimpleNamespace(
        pose_landmarks=[[SimpleNamespace(x=x, y=y, visibility=v, presence=p)
                         for x, y, v, p in record["landmarks"]]],
        pose_world_landmarks=[[SimpleNamespace(x=x, y=y, z=z)
                               for x, y, z in record["world_landmarks"]]])
    guard = module.BodyGuard.__new__(module.BodyGuard)
    guard.error = None
    guard.mp = SimpleNamespace(Image=lambda **kwargs: None, ImageFormat=SimpleNamespace(SRGB=1))
    guard.detector = SimpleNamespace(detect=lambda image: result)
    state, frame = {}, np.zeros(SHAPE, np.uint8)
    assert guard.detect(frame, state=state)[0] == []  # Hips out of frame.
    assert len(guard.contact_observations(state=state)) == 1
    assert guard.contact_observations(state={}) == []
    clock[0] = 1.21
    assert guard.contact_observations(state=state) == []
    result.pose_landmarks = []
    guard.detect(frame, state=state)
    assert guard.contact_observations(state=state) == []


def reported_contact_scene(now=1., mirror=False, scale=1.):
    """Approximate displayed joints in the mouth-touch screenshot, not a replay."""
    shape = (round(500 * scale), round(960 * scale), 3)
    pixels = np.asarray([
        [391, 595], [440, 566], [461, 531], [479, 500], [489, 534],
        [421, 490], [461, 447], [495, 442], [518, 445],
        [419, 502], [465, 493], [482, 482], [507, 519],
        [416, 512], [461, 532], [484, 500], [504, 535],
        [413, 524], [440, 566], [461, 532], [489, 534],
    ], dtype=float)
    pixels[:, 1] -= 158  # Remove the viewer header.
    hand = pixels / [960, 500]
    mesh = np.tile([.52, .52], (478, 1))
    angle = np.linspace(-np.pi / 2, 3 * np.pi / 2, len(FACE_OVAL), endpoint=False)
    mesh[list(FACE_OVAL)] = [.52, .52] + np.column_stack((.13 * np.cos(angle), .32 * np.sin(angle)))
    mesh[1] = [.52, .52]
    if mirror:
        hand[:, 0] = 1 - hand[:, 0]
        mesh[:, 0] = 1 - mesh[:, 0]
    return hand, [dict(points=mesh, observed_at=now)], shape


@pytest.mark.parametrize("label", ["fist", "like", "left"])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("scale", [.5, 1., 1.5])
def test_reported_mouth_touch_blocks_high_confidence_without_pose(label, mirror, scale):
    session, state = RuntimeSession(RuntimeConfig()), {}
    for index in range(10):
        now = 1 + index * .08
        hand, faces, shape = reported_contact_scene(now, mirror, scale)
        row = row_for(hand, shape, label)
        apply_contact_policy([row], shape, faces, [], state=state, now=now)
        assert row["rejection_reason"] == FACE_OVERLAP_REASON
        assert not row["face_context"]["contact_confirmed"]  # Uncertain depth remains explicit.
        chosen, selection, previews = session.hand_selector.update([row], shape, now)
        assert chosen is None
        assert previews[0]["gesture"] == "no_gesture"
        assert not previews[0]["eligible"]


def test_face_hallucination_is_rejected_on_first_frame_at_998_percent():
    hand, faces, _ = scene()
    # A small complete landmark cloud over the eye/nose area as in screenshot 3.
    hand = (representative_hand() - .5) * .16 + [.56, .35]
    row = row_for(hand, label="open_palm")
    state = {}
    apply_contact_policy([row], SHAPE, faces, [], state=state, now=1.)
    assert row["rejection_reason"] == FACE_HAND_REASON
    assert row["face_context"]["face_alias"]
    chosen, _, previews = RuntimeSession(RuntimeConfig()).hand_selector.update([row], SHAPE, 1.)
    assert chosen is None
    assert previews[0]["gesture"] == "no_gesture"


@pytest.mark.parametrize("depth", [.045, .065])
def test_noisy_contact_depth_is_uncertain_and_paused(depth):
    hand, faces, bodies = scene(depth=depth)
    row = row_for(hand)
    apply_contact_policy([row], SHAPE, faces, bodies, state={}, now=1.)
    assert row["rejection_reason"] == CONTACT_PENDING_REASON
    assert row["face_context"]["status"] == "depth_uncertain"


def test_hidden_pose_thumb_cannot_turn_a_touch_into_a_command():
    hand, faces, bodies = scene()
    bodies[0]["landmarks"][21, 2:] = .1
    row = row_for(hand, label="like")
    apply_contact_policy([row], SHAPE, faces, bodies, state={}, now=1.)
    assert row["rejection_reason"] == FACE_OVERLAP_REASON
    assert row["face_context"]["arm_supported"]


def test_face_overlap_does_not_prevent_scoring_or_recovery(monkeypatch):
    import backend.hand_detection as module
    clock = [1.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.)
    hand, faces, shape = reported_contact_scene()
    detector = simulated_detector()
    detector._face_mesh = SimpleNamespace(detector=True, detect=lambda rgb, *, state: (
        reported_contact_scene(clock[0])[1], dict(ready=True, face_count=1)))
    frame, state, scored = np.full(shape, 145, np.uint8), {}, []
    # Establish the actual face before the hand arrives. A face-shaped fit
    # appearing for the first time on a hand is no longer contact evidence.
    detector._run_candidates = lambda rgb, view, *, video: []
    detector.detect_candidates(frame, state=state, candidate_score=lambda row: 1.)
    detector._run_candidates = lambda rgb, view, *, video: [
        (.1, hand + ([.02, 0] if view[4] else [0, 0]), None, None)]
    def score(row):
        scored.append(True)
        return 1.
    for index in range(7):
        clock[0] = 1 + index * .1
        state["pose_recovery_requested"] = True
        rows = detector.detect_candidates(frame, state=state, candidate_score=score)
        assert rows[0]["rejection_reason"] is None
    assert scored


def test_occluded_mesh_uses_unexpanded_face_box_and_never_confirms_from_it():
    faces = box_face_observations([((300, 70, 350, 430), 1.)], SHAPE)
    state = {}
    for now in (1., 1.1, 1.2):
        hand, _, bodies = scene(now)
        row = row_for(hand)
        apply_contact_policy([row], SHAPE, faces, bodies, state=state, now=now)
        assert row["face_context"]["suppressed"]
        assert not row["face_context"]["contact_confirmed"]
        assert row["face_context"]["region_source"] == "face_box_fallback"


def test_face_box_fallback_cache_never_crosses_sessions(monkeypatch):
    import backend.face_guard as module
    from backend.face_guard import FaceGuard
    clock = [1.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    result = SimpleNamespace(detections=[SimpleNamespace(bounding_box=SimpleNamespace(
        origin_x=100, origin_y=40, width=70, height=90))])
    guard = FaceGuard.__new__(FaceGuard)
    guard.mp = SimpleNamespace(Image=lambda **kwargs: None, ImageFormat=SimpleNamespace(SRGB=1))
    guard.detector = SimpleNamespace(detect=lambda image: result)
    frame, state = np.zeros((240, 320, 3), np.uint8), {}
    assert guard.detect(frame, state=state)
    result.detections = []
    assert guard.detect(frame, state={}) == []
    clock[0] = 1.4
    assert guard.detect(frame, state=state)
    clock[0] = 1.51
    assert guard.detect(frame, state=state) == []


def test_unused_face_check_cannot_pause_live_classification(monkeypatch):
    import backend.hand_detection as module
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.)
    detector = simulated_detector()
    detector._face_mesh = SimpleNamespace(detector=True, detect=lambda rgb, *, state: (
        [], dict(ready=False, face_count=0, error="model inference failed")))
    detector._run_candidates = lambda rgb, view, *, video: [(.1, scene()[0], None, None)]
    def score(row):
        return 1.
    rows = detector.detect_candidates(np.full(SHAPE, 145, np.uint8), state={}, candidate_score=score)
    assert rows[0]["rejection_reason"] is None
