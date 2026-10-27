"""Face/ear regressions use approximate screenshot joints, not pixel replays."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.config import RuntimeConfig
from backend.face_guard import FaceGuard, face_context_observation, overlaps_face_as_small_hand
from backend.model_runtime import RuntimeSession
from backend.tests.test_hand_detection import simulated_detector
from backend.tests.test_camera_stability import representative_hand


def reported_resting_hand(index):
    # Approximate visible wrist, thumb, and four finger chains in HUD pixels.
    # Boxes are approximate face extents; these are spatial-policy fixtures,
    # not claims about BlazeFace outputs on the original images.
    poses = [
        ([[547,440], [501,404], [478,353], [480,307], [501,293],
          [513,312], [520,280], [520,295], [513,312],
          [543,320], [548,288], [546,302], [543,320],
          [568,335], [582,299], [577,315], [568,335],
          [597,353], [609,326], [601,337], [597,353]], (625,260,175,245)),
        ([[577,467], [580,426], [579,378], [548,350], [533,353],
          [586,351], [599,322], [613,324], [616,342],
          [590,371], [618,329], [629,343], [605,374],
          [617,378], [634,339], [638,351], [619,378],
          [631,381], [642,362], [639,370], [631,381]], (631,266,165,245)),
        ([[462,438], [509,409], [529,376], [542,353], [563,332],
          [485,322], [531,280], [541,292], [559,302],
          [496,327], [528,294], [560,303], [568,312],
          [505,331], [560,302], [579,315], [582,329],
          [514,338], [575,310], [590,324], [584,329]], (538,235,177,245)),
        ([[469,437], [429,390], [416,333], [433,312], [456,310],
          [435,317], [451,277], [468,300], [456,310],
          [477,316], [483,281], [491,300], [481,314],
          [508,330], [511,294], [510,310], [508,330],
          [525,344], [531,316], [527,330], [525,344]], (550,235,180,250)),
    ]
    points, box = poses[index]
    return np.asarray(points, dtype=np.float32), box


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("scale", [.5, 1., 1.5])
def test_reported_face_ear_poses_observed_at_each_scale_and_mirror(index, mirror, scale):
    points, (x, y, w, h) = reported_resting_hand(index)
    if mirror:
        points[:, 0] = 960 - points[:, 0]
        x = 960 - x - w
    shape = (round(750 * scale), round(960 * scale), 3)
    boxes = [tuple(value * scale for value in (x, y, w, h))]
    normalized = points / [960, 750]
    assert face_context_observation(normalized, shape, boxes)["near_face"] is True


def test_fingers_at_ear_are_observed_even_when_palm_center_is_outside_face():
    points, box = reported_resting_hand(3)
    normalized = points / [960, 750]
    assert not overlaps_face_as_small_hand(normalized, (750, 960, 3), [box])
    assert face_context_observation(normalized, (750, 960, 3), [box])["near_face"] is True


def test_separate_command_and_small_background_face_remain_eligible():
    points, box = reported_resting_hand(3)
    assert not face_context_observation((points - [260, 0]) / [960, 750], (750, 960, 3), [box])["near_face"]
    assert face_context_observation(representative_hand(), (400, 400, 3), [(150,60,90,110)])["near_face"]
    assert not face_context_observation(points / [960, 750], (750, 960, 3), [])["near_face"]
    assert not face_context_observation(points / [960, 750], (750, 960, 3), [(460,300,30,40)])["near_face"]


def test_one_fingertip_on_face_counts_without_palm_overlap():
    assert face_context_observation(
        representative_hand(), (400, 400, 3), [(180, 100, 80, 105)]
    )["near_face"]


@pytest.mark.parametrize("gesture", [label for label in RuntimeConfig().class_names if label != "dorsal"])
@pytest.mark.parametrize("blocked", [True, False])
def test_legacy_face_box_alone_never_blocks_a_gesture(monkeypatch, gesture, blocked):
    import backend.hand_detection as detection_module
    monkeypatch.setattr(detection_module.time, "perf_counter", lambda: 10.)
    points, box = reported_resting_hand(3)
    detector = simulated_detector()
    detector._face_guard = SimpleNamespace(detector=True, detect=lambda rgb: [box] if blocked else [])
    detector._run_candidates = lambda rgb, view, *, video: [(.1, points / [960,750], None, None)]
    config = RuntimeConfig()
    session = RuntimeSession(config)
    detector._multi_context = session.hand_selector.detector_state
    probabilities = np.eye(len(config.class_names))[config.class_to_idx[gesture]]
    scored = []

    def score(row):
        scored.append(True)
        row["resolved_label"] = gesture
        row["probabilities"] = probabilities
        row["gate_args"] = dict(known_gesture_mass=1.0)
        return 1.

    decisions = []
    for frame in range(8):
        rows = detector.detect_candidates(
            np.full((750,960,3), 145, dtype=np.uint8),
            state=session.hand_selector.detector_state, candidate_score=score)
        assert rows[0]["rejection_reason"] is None
        assert not rows[0]["face_context"]["contact_confirmed"]
        chosen, selection, previews = session.hand_selector.update(
            rows, (750,960,3), 1 + frame * .1)
        decisions.append(previews[0]["gesture"])
    assert scored
    assert decisions == [gesture] * 8
    assert chosen is not None


def test_face_occlusion_hold_expires_and_resets_on_resolution_change(monkeypatch):
    import backend.face_guard as module
    guard = FaceGuard.__new__(FaceGuard)
    guard.boxes = []
    guard._recent_boxes = []
    guard._image_shape = None
    guard.mp = SimpleNamespace(Image=lambda **kwargs: None, ImageFormat=SimpleNamespace(SRGB=1))
    face = SimpleNamespace(bounding_box=SimpleNamespace(origin_x=100, origin_y=40, width=70, height=90))
    detection = SimpleNamespace(detections=[face])
    guard.detector = SimpleNamespace(detect=lambda image: detection)
    frame = np.zeros((240,320,3), dtype=np.uint8)
    now = 1.
    monkeypatch.setattr(module.time, 'monotonic', lambda: now)
    assert guard.detect(frame) == [(100,40,70,90)]
    detection.detections = []
    now = 1.4
    assert guard.detect(frame) == [(100,40,70,90)]
    background = SimpleNamespace(bounding_box=SimpleNamespace(origin_x=10, origin_y=10, width=20, height=30))
    detection.detections = [background]
    now = 1.45
    assert guard.detect(frame) == [(100,40,70,90), (10,10,20,30)]
    detection.detections = []
    now = 1.51
    assert guard.detect(frame) == [(10,10,20,30)]
    now = 1.96
    assert guard.detect(frame) == []
    detection.detections = [face]
    now = 2.
    assert guard.detect(frame)
    detection.detections = []
    assert guard.detect(np.zeros((480,640,3), dtype=np.uint8)) == []
