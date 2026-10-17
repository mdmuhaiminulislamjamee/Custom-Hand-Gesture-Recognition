"""Fresh recovery observations must work before a hand leaves the clothing."""
import numpy as np

from backend.tests.test_hand_detection import simulated_detector


def test_cold_start_revisits_upright_crop_while_global_search_progresses():
    detector = simulated_detector()
    shape = (480, 640, 3)
    state = {"frame_time": 1.}
    views = [detector._live_recovery_view(shape, state, []) for _ in range(16)]
    upright = detector._startup_recovery_view(shape)
    assert views[:3] == [upright] * 3
    assert views.count(upright) >= 12
    assert len(set(views)) > 2


def test_productive_crop_is_retried_every_frame_and_expires_without_fresh_evidence():
    detector = simulated_detector()
    shape = (480, 640, 3)
    view = (160, 90, 220, 220, 0, 1)
    state = {"frame_time": 1.2, "view": view, "view_last_seen": 1.,
             "ever_qualified": True}
    for _ in range(10):
        assert detector._live_recovery_view(shape, state, []) == view
    state.update(frame_time=2., view_misses=2)
    assert detector._live_recovery_view(shape, state, []) != view


def test_finding_a_peer_does_not_abandon_the_existing_hands_productive_view():
    from backend.tests.test_hand_selection import observation

    detector = simulated_detector()
    hand = observation(scale=.5)
    view = detector._hand_crop(hand['points'], (480, 640, 3))
    state = dict(frame_time=1.2, view=view, view_last_seen=1.1, ever_qualified=True)
    first = detector._live_recovery_view((480, 640, 3), state, [hand])
    assert first == view
    state['visited_views'] = {view}
    second = detector._live_recovery_view((480, 640, 3), state, [hand])
    assert second != view


def test_single_visible_hand_searches_the_opposite_torso_side():
    from backend.tests.test_camera_stability import representative_hand

    detector = simulated_detector()
    shape = (480, 640, 3)
    left = (representative_hand() - .5) * .28 + [.32, .52]
    right = (representative_hand() - .5) * .28 + [.68, .52]

    left_view = detector._live_recovery_view(
        shape, {"frame_time": 1., "ever_qualified": True},
        [dict(points=left, view=(0, 0, 640, 480, 0, 0))],
    )
    right_view = detector._live_recovery_view(
        shape, {"frame_time": 1., "ever_qualified": True},
        [dict(points=right, view=(0, 0, 640, 480, 0, 0))],
    )

    assert left_view[0] + left_view[2] / 2 > 640 * .5
    assert right_view[0] + right_view[2] / 2 < 640 * .5
    assert left_view[2:4] == right_view[2:4] == (326, 326)
    assert left_view[5] == right_view[5] == 0


def test_missing_peer_search_alternates_upright_and_side_views():
    from backend.tests.test_camera_stability import representative_hand

    detector = simulated_detector()
    shape = (480, 640, 3)
    hand = (representative_hand() - .5) * .28 + [.32, .52]
    row = dict(points=hand, view=(0, 0, 640, 480, 0, 0))
    state = {"frame_time": 1., "ever_qualified": True}

    views = [detector._live_recovery_view(shape, state, [row]) for _ in range(12)]

    assert {view[4] for view in views} == {0, 3, 1}
    assert {view[5] for view in views} == {0, 1}
    # Each orientation/contrast pair searches both directions. The person may
    # be off-centre; screen centre cannot predict which side holds the peer.
    assert all(views[i][0] != views[i + 1][0] for i in range(0, 12, 2))


def test_cached_crop_never_creates_a_hand_or_skips_the_full_frame(monkeypatch):
    detector = simulated_detector()
    image = np.full((480, 640, 3), 120, np.uint8)
    view = (160, 90, 220, 220, 0, 1)
    state = {"shape": image.shape[:2], "frame_time": 1.2,
             "view": view, "view_last_seen": 1., "ever_qualified": True}
    calls = []
    monkeypatch.setattr(detector, "_run_candidates",
                        lambda rgb, view, **kwargs: calls.append(view) or [])
    result = detector.detect_candidates(image, state=state, candidate_score=lambda row: 1.)
    assert result == []
    assert calls[:2] == [(0, 0, 640, 480, 0, 0), view]
    assert len(calls) <= 3


def test_startup_recovery_uses_video_with_monotonic_time_and_isolates_clients():
    import threading
    from pathlib import Path
    from types import SimpleNamespace
    detector = simulated_detector()
    trackers, timestamps, closed = [], [], []

    def make_tracker(options):
        tracker = SimpleNamespace(
            detect_for_video=lambda image, timestamp: (
                timestamps.append(timestamp) or SimpleNamespace(hand_landmarks=[])),
            close=lambda: closed.append(True))
        trackers.append(tracker)
        return tracker

    detector._mp = SimpleNamespace(
        Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB="RGB"),
        tasks=SimpleNamespace(BaseOptions=lambda **kwargs: kwargs, vision=SimpleNamespace(
            RunningMode=SimpleNamespace(VIDEO="video"), HandLandmarkerOptions=lambda **kwargs: kwargs,
            HandLandmarker=SimpleNamespace(create_from_options=make_tracker))))
    detector._model_path = Path("fixture.task")
    detector._landmarker = None
    detector._timestamp_lock = threading.Lock()
    detector._last_timestamp_ms = -1
    detector._frame_time = 1.
    detector._last_diagnostics = {}
    rgb = np.full((32, 32, 3), 120, np.uint8)
    view = detector._startup_recovery_view(rgb.shape)
    first = {}
    for state in (first, first, {}):
        detector._live_recovery_context = state
        assert detector._run_candidates(rgb, view, video=False) == []
    assert timestamps == [1000, 1001, 1002]
    assert len(trackers) == 2 and len(closed) == 1


def test_two_recovery_streams_retain_independent_video_history_and_close():
    import threading
    from pathlib import Path
    from types import SimpleNamespace

    detector = simulated_detector()
    streams, closed = [], []

    def make_tracker(options):
        timestamps = []
        tracker = SimpleNamespace(
            times=timestamps,
            detect_for_video=lambda image, timestamp: (
                timestamps.append(timestamp) or SimpleNamespace(hand_landmarks=[])),
            close=lambda: closed.append(id(timestamps)))
        streams.append(tracker)
        return tracker

    detector._mp = SimpleNamespace(
        Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB='RGB'),
        tasks=SimpleNamespace(BaseOptions=lambda **kwargs: kwargs, vision=SimpleNamespace(
            RunningMode=SimpleNamespace(VIDEO='video'), HandLandmarkerOptions=lambda **kwargs: kwargs,
            HandLandmarker=SimpleNamespace(create_from_options=make_tracker))))
    detector._model_path = Path('fixture.task')
    detector._landmarker = detector._recovery = None
    detector._timestamp_lock = threading.Lock()
    detector._last_timestamp_ms = -1
    detector._frame_time = 1.
    detector._last_diagnostics = {}
    image = np.full((160, 320, 3), 120, np.uint8)
    first, second = (0, 0, 160, 160, 0, 0), (160, 0, 160, 160, 3, 0)
    state = {'recent_hands': [{'view': first}, {'view': second}]}
    detector._live_recovery_context = state
    for _ in range(8):
        for view in (first, second):
            detector._run_candidates(image, view, video=False)
    assert len(streams) == 2 and not closed
    assert all(len(stream.times) == 8 for stream in streams)
    assert all(np.diff(stream.times).min() > 0 for stream in streams)
    detector._live_recovery_context = {'view': first}
    detector._run_candidates(image, first, video=False)
    assert len(closed) == 2  # Another client cannot inherit either track.
    detector.close()
    assert len(closed) == 3 and len(set(closed)) == 3


def test_both_hands_remain_fresh_when_full_frame_alternately_misses_one_or_both(monkeypatch):
    import backend.hand_detection as module
    from backend.tests.test_hand_selection import observation

    monkeypatch.setattr(module.time, 'perf_counter', lambda: 1.)
    detector = simulated_detector()
    image = np.full((480, 640, 3), 120, np.uint8)
    full = (0, 0, 640, 480, 0, 0)
    owner, peer = observation(scale=.5), observation(x=.75, scale=.5, hand='Left')
    peer_view = detector._missing_peer_crop(owner['points'], image.shape, enhance=0)
    owner_view = detector._hand_crop(owner['points'], image.shape)
    clock = [0]
    peer_present = [True]
    calls = []

    def run(rgb, view, **kwargs):
        calls.append(view)
        if view == full:
            rows = [owner] if clock[0] % 2 == 0 else []
        elif view == peer_view:
            rows = [peer] if peer_present[0] else []
        elif view == owner_view:
            rows = [owner]
        else:
            rows = []
        return [(row['area'], row['points'], None, None) for row in rows]

    monkeypatch.setattr(detector, '_run_candidates', run)
    state = {}
    for frame in range(30):
        clock[0] = frame
        state['frame_time'] = 1 + frame * .1
        calls.clear()
        rows = detector.detect_candidates(image, state=state, candidate_score=lambda row: 1.)
        assert len(rows) == 2
        assert calls[0] == full and len(calls) <= 3
        assert any(row['view'] == peer_view for row in rows)
    # When the real peer is absent, its retained stream cannot manufacture it.
    peer_present[0] = False
    for frame in range(30, 40):
        clock[0] = frame
        state['frame_time'] = 1 + frame * .1
        rows = detector.detect_candidates(image, state=state, candidate_score=lambda row: 1.)
        assert all(row['points'][0, 0] < .5 for row in rows)

