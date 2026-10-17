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


def test_mediapipe_video_receives_recorded_clock_with_strict_ordering():
    from types import SimpleNamespace
    import threading
    detector = HandDetector.__new__(HandDetector)
    timestamps = []
    detector._timestamp_lock = threading.Lock()
    detector._last_timestamp_ms = -1
    detector._last_diagnostics = {}
    detector._mp = SimpleNamespace(Image=lambda **kwargs: kwargs,
                                   ImageFormat=SimpleNamespace(SRGB="RGB"))
    detector._landmarker = SimpleNamespace(detect_for_video=lambda image, timestamp:
        (timestamps.append(timestamp) or SimpleNamespace(hand_landmarks=[])))
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    for frame_time in (1000., 1000.2, 1000.2):
        detector._frame_time = frame_time
        detector._run_candidates(rgb, (0, 0, 4, 4, 0, 0), video=True)
    assert timestamps == [1000000, 1000200, 1000201]


def test_initialization_loads_only_hand_models(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    vision = SimpleNamespace(
        RunningMode=SimpleNamespace(VIDEO="video", IMAGE="image"),
        HandLandmarkerOptions=lambda **kwargs: kwargs,
        HandLandmarker=SimpleNamespace(create_from_options=lambda options:
                                      SimpleNamespace(close=lambda: None)),
    )
    fake_mp = SimpleNamespace(tasks=SimpleNamespace(
        BaseOptions=lambda **kwargs: kwargs, vision=vision))
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    model_path = tmp_path / "hand_landmarker.task"
    model_path.write_bytes(b"hand-model-fixture")
    detector = HandDetector(model_path)
    try:
        assert detector.ready
        assert detector.error is None
        assert not hasattr(detector, "_body_guard")
        assert detector._face_guard is None
    finally:
        detector.close()


def test_recovery_prioritizes_central_side_views_without_expanding_search_set():
    detector = simulated_detector()
    height, width = 480, 640
    side = int(min(width, height) * .60)
    views = detector._search_views((height, width, 3))

    original_full_views = {
        (0, 0, width, height, turn, enhance)
        for turn, enhance in ((3, 0), (3, 1), (1, 0), (1, 1), (2, 0), (0, 1))
    }
    original_crop_views = {
        (round((width - side) * fx), round((height - side) * fy), side, side, turn, 1)
        for turn in (0, 3, 1)
        for fx, fy in ((.5, .5), (0, .5), (1, .5), (0, 0), (1, 0), (0, 1), (1, 1))
    }

    assert len(views) == 27
    assert len(set(views)) == 27
    assert set(views) == original_full_views | original_crop_views
    assert views[:4] == [
        (0, 0, width, height, 3, 0),
        (0, 0, width, height, 1, 0),
        (round((width - side) * .5), round((height - side) * .5), side, side, 3, 1),
        (round((width - side) * .5), round((height - side) * .5), side, side, 1, 1),
    ]


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


def test_live_miss_retries_upright_crop_before_rotation(monkeypatch):
    import backend.hand_detection as module
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 10.)
    detector = simulated_detector()
    calls = []
    hand = representative_hand()

    def run(rgb, view, *, video):
        calls.append((view, video))
        return [(.1, hand, None, None)] if view[5] else []

    monkeypatch.setattr(detector, '_run_candidates', run)
    rows = detector.detect_candidates(np.full((480, 640, 3), 140, np.uint8),
                                      state={}, candidate_score=lambda row: 1.)
    assert calls[:2] == [((0, 0, 640, 480, 0, 0), False),
                         (detector._startup_recovery_view((480, 640, 3)), False)]
    assert len(calls) <= 3
    assert len(rows) == 1
    np.testing.assert_allclose(rows[0]['points'], hand)



def test_live_tracking_loss_uses_recent_hand_crop_and_fresh_pixels(monkeypatch):
    import backend.hand_detection as module
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 10.)
    detector = simulated_detector()
    image = np.full((480, 640, 3), 140, np.uint8)
    hand = (representative_hand() - .5) * .35 + .5
    state = {'frame_time': 1.}
    monkeypatch.setattr(detector, '_run_candidates', lambda *args, **kwargs: [(.1, hand, None, None)])
    assert len(detector.detect_candidates(image, state=state, candidate_score=lambda row: 1.)) == 1
    calls = []

    def run(rgb, view, *, video):
        calls.append(view)
        return [(.1, hand, None, None)] if view[2] < 640 else []

    monkeypatch.setattr(detector, '_run_candidates', run)
    state['frame_time'] = 1.1
    rows = detector.detect_candidates(image, state=state, candidate_score=lambda row: 1.)
    assert 2 <= len(calls) <= 3 and len(rows) == 1
    assert calls[0] == (0, 0, 640, 480, 0, 0)
    assert calls[1][2] < 640 and calls[1][3] < 480 and calls[1][5] == 1
    x, y, width, height = calls[1][:4]
    pixels = hand * [640, 480]
    assert (pixels >= [x, y]).all() and (pixels <= [x + width, y + height]).all()
    # Recent coordinates are a search hint, never a substitute for detection.
    monkeypatch.setattr(detector, '_run_candidates', lambda *args, **kwargs: [])
    state['frame_time'] = 1.2
    assert detector.detect_candidates(image, state=state, candidate_score=lambda row: 1.) == []


def test_local_recovery_expires_and_does_not_cross_sessions():
    detector = simulated_detector()
    shape = (480, 640, 3)
    hand = (representative_hand() - .5) * .35 + .5
    state = {'frame_time': 1.1, 'recent_hands': [dict(points=hand, seen=1., turns=3)]}
    assert detector._live_recovery_view(shape, state, [])[2] < 640
    other = {'frame_time': 1.1, 'ever_qualified': True, 'recovery_attempts': 1}
    assert detector._live_recovery_view(shape, other, []) == (0, 0, 640, 480, 0, 1)
    state['frame_time'], state['recovery_attempts'], state['search_index'] = 2., 0, 0
    assert detector._live_recovery_view(shape, state, []) == detector._startup_recovery_view(shape)


def test_local_recovery_leaves_global_search_running():
    detector = simulated_detector()
    hand = (representative_hand() - .5) * .35 + .5
    local = detector._hand_crop(hand, (480, 640, 3))
    state = {'frame_time': 1.1, 'ever_qualified': True, 'view': local,
             'recent_hands': [dict(points=hand, seen=1., turns=0)]}
    views = []
    for frame in range(64):
        state['frame_time'] = 1.1 + frame * .1
        views.append(detector._live_recovery_view((480, 640, 3), state, []))
    assert local in views
    assert (0, 0, 640, 480, 0, 1) in views
    assert (0, 0, 640, 480, 3, 0) in views
    assert (0, 0, 640, 480, 1, 0) in views



def test_recovery_targets_missing_peer_before_already_visible_hand():
    detector = simulated_detector()
    hand = (representative_hand() - .5) * .3 + [.25, .5]
    peer = hand + [.5, 0]
    state = {'frame_time': 1.1, 'ever_qualified': True, 'recent_hands': [dict(points=peer, seen=1., turns=0)]}
    view = detector._live_recovery_view((480, 640, 3), state,
                                       [dict(points=hand, view=(0, 0, 640, 480, 0, 0))])
    assert view == detector._hand_crop(peer, (480, 640, 3))


def test_contrast_recovery_enhances_low_contrast_crop_without_mutating_camera_frame():
    from types import SimpleNamespace
    detector = simulated_detector()
    captured = []
    detector._mp = SimpleNamespace(Image=lambda **kwargs: kwargs,
                                   ImageFormat=SimpleNamespace(SRGB='RGB'))
    detector._recovery = SimpleNamespace(detect=lambda image: (
        captured.append(image['data']) or SimpleNamespace(hand_landmarks=[])))
    image = np.repeat(np.tile(np.linspace(90, 115, 128, dtype=np.uint8), (128, 1))[:, :, None], 3, axis=2)
    original = image.copy()
    detector._run_candidates(image, (16, 16, 96, 96, 0, 1), video=False)
    assert captured[0].std() > image[16:112, 16:112].std()
    assert captured[0].flags.c_contiguous
    np.testing.assert_array_equal(image, original)


def test_live_recovery_respects_frame_budget(monkeypatch):
    import backend.hand_detection as module
    times = iter([1., 1.056])
    monkeypatch.setattr(module.time, 'perf_counter', lambda: next(times))
    detector = simulated_detector()
    calls = []
    monkeypatch.setattr(detector, '_run_candidates', lambda *args, **kwargs: calls.append(True) or [])
    assert detector.detect_candidates(np.full((480, 640, 3), 140, np.uint8),
                                      state={}, candidate_score=lambda row: 1.) == []
    assert len(calls) == 1
    assert not detector.diagnostics()['recovery_attempted']
