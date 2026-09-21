import numpy as np
import pytest

from backend.hand_detection import HandDetector, hand_pixel_metrics, normalized_hand_shape, restore_points, restore_world_points
from backend.model_runtime import RuntimeSession, hand_plane_angles
from backend.config import RuntimeConfig
from backend.tests.test_camera_stability import representative_hand


@pytest.mark.parametrize("turns", range(4))
def test_rotated_crop_returns_original_camera_axes(turns):
    original = np.array([[.2, .3], [.8, .7]], dtype=np.float32)
    rotated = original.copy()
    for _ in range(turns):
        rotated = np.column_stack([rotated[:, 1], 1 - rotated[:, 0]])
    restored = restore_points(rotated, (100, 50, 200, 150, turns, 0), (480, 640, 3))
    np.testing.assert_allclose(restored * [640, 480], original * [200, 150] + [100, 50], atol=1e-4)


@pytest.mark.parametrize("turns", range(4))
def test_rotated_world_landmarks_return_to_original_camera_axes(turns):
    original = np.zeros((21, 3), dtype=np.float32)
    original[:, 0] = np.linspace(-.2, .2, 21)
    original[:, 1] = np.linspace(.3, -.3, 21)
    original[:, 2] = np.linspace(-.1, .1, 21)
    rotated = original.copy()
    for _ in range(turns):
        x = rotated[:, 0].copy()
        rotated[:, 0] = rotated[:, 1]
        rotated[:, 1] = -x
    np.testing.assert_allclose(restore_world_points(rotated, turns), original, atol=1e-6)


def test_live_plane_angles_use_real_three_dimensional_palm_axis():
    points = np.zeros((21, 3), dtype=np.float32)
    points[[5, 9, 13, 17]] = [1.0, 1.0, 1.0]
    angles = hand_plane_angles(points)
    assert angles == pytest.approx({"xy": 45.0, "yz": 45.0, "xz": 45.0, "zx": 45.0})


def test_hand_pixel_measurement_is_not_crop_magnification():
    local = representative_hand()
    restored = restore_points(local, (100, 100, 80, 80, 0, 0), (640, 640, 3))
    metrics = hand_pixel_metrics(restored, (640, 640, 3))
    expected = hand_pixel_metrics(local, (80, 80, 3))
    assert metrics["palm_scale_px"] == pytest.approx(expected["palm_scale_px"], rel=1e-6)
    assert metrics["hand_span_px"] == pytest.approx(expected["hand_span_px"], rel=1e-6)


def test_distant_complete_landmarks_are_not_rejected_by_old_area_gate():
    points = (representative_hand() - .5) * .16 + .5
    assert np.prod(np.ptp(points, axis=0)) < .018
    accepted, diagnostics = RuntimeSession(RuntimeConfig()).stabilize_landmarks(points, 1.0)
    assert accepted is not None
    assert diagnostics["accepted"]


def test_degenerate_and_clipped_distant_landmarks_still_rejected():
    session = RuntimeSession(RuntimeConfig())
    assert session.stabilize_landmarks(np.full((21, 2), .5), 1.0)[0] is None
    assert session.stabilize_landmarks(representative_hand() + 2, 1.0)[0] is None


def simulated_detector():
    detector = HandDetector.__new__(HandDetector)
    detector.ready = True
    detector._shape = detector._view = None
    detector._search_index = detector._misses = detector._unqualified_frames = 0
    detector._recovery_tracking = detector._pose_recovery_requested = False
    detector._pending_recovery = detector._accepted_points = detector._accepted_view = None
    detector._pending_recovery_count = 0
    return detector


def test_recovery_can_replace_bad_tracked_fingers_with_new_image_evidence(monkeypatch):
    detector = simulated_detector()
    initial = representative_hand()
    recovered = initial - [.1, 0]
    calls = []
    def run(rgb, view, *, video):
        calls.append(video)
        return (.10, initial if video else recovered, None)
    monkeypatch.setattr(detector, '_run_view', run)
    scorer = lambda points: float(points[0, 0] < .45)
    result = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8),
                             candidate_score=scorer)
    assert result is None  # New recovery observations require confirmation.
    result = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8),
                             candidate_score=scorer)
    np.testing.assert_array_equal(result, recovered)
    assert calls == [True, False, True, False]


def test_recovery_confirmation_allows_whole_hand_motion(monkeypatch):
    detector = simulated_detector()
    initial = representative_hand()
    positions = iter([initial, initial + [.08, .03], initial + [.15, .06]])
    monkeypatch.setattr(detector, '_run_view', lambda rgb, view, *, video: (.1, next(positions), None))
    detector._shape = (640, 640)
    detector._recovery_tracking = True
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    assert detector.detect(blank) is None
    assert detector.detect(blank) is not None
    np.testing.assert_allclose(normalized_hand_shape(initial), normalized_hand_shape(initial + [.15, .06]), atol=1e-5)


def test_recovery_is_skipped_when_first_pass_uses_its_time_allowance(monkeypatch):
    import backend.hand_detection as module
    detector = simulated_detector()
    times = iter([1., 1.056])
    monkeypatch.setattr(module.time, 'perf_counter', lambda: next(times))
    calls = []
    def run(rgb, view, *, video):
        calls.append(video)
        return None
    monkeypatch.setattr(detector, '_run_view', run)
    assert detector.detect(np.zeros((640, 640, 3), dtype=np.uint8)) is None
    assert calls == [True]


def test_transient_miss_retries_last_successful_rotation_first(monkeypatch):
    detector = simulated_detector()
    detector._shape = (480, 640)
    detector._view = (0, 0, 640, 480, 3, 0)
    calls = []
    recovered = representative_hand()

    def run(_rgb, view, *, video):
        calls.append((view, video))
        return None if video else (.1, recovered, None)

    monkeypatch.setattr(detector, '_run_view', run)
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) is None
    assert calls == [
        ((0, 0, 640, 480, 3, 0), True),
        ((0, 0, 640, 480, 3, 1), False),
    ]
    assert detector.diagnostics()['rejection_reason'] == 'confirming_recovered_hand'


def test_recovery_matching_last_accepted_hand_resumes_without_confirmation_gap(monkeypatch):
    detector = simulated_detector()
    hand = representative_hand()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    calls = iter([
        (.1, hand, None),
        None,
        (.1, hand + [.03, -.02], None),
    ])
    monkeypatch.setattr(detector, '_run_view', lambda *_args, **_kwargs: next(calls))

    assert detector.detect(blank) is not None
    recovered = detector.detect(blank)

    assert recovered is not None
    assert detector.diagnostics()['rejection_reason'] is None
