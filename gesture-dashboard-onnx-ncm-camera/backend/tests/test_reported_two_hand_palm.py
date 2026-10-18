"""Approximate screenshot skeletons; image detection and depth are not replayed."""
from types import SimpleNamespace
import threading

import cv2
import numpy as np
import pytest

from backend.config import load_runtime_config
from backend.geometry import GeometryResolver
from backend.model_runtime import InferenceEngine, ModelManager, RuntimeSession
from backend.tests.test_hand_detection import simulated_detector


SHAPE = (540, 960, 3)


def reported_hands(mirror=False, scene=1):
    # Manually transcribed from the user's screenshot, in MediaPipe order.
    # The video occupies 956 x 720 pixels below a 23-pixel window title bar.
    skeletons = {
        "Right": [[276,550], [248,512], [233,471], [219,432], [208,406],
                  [257,442], [260,381], [265,348], [269,320],
                  [279,444], [288,376], [294,340], [303,313],
                  [300,455], [310,398], [315,369], [322,335],
                  [314,468], [334,429], [345,404], [351,385]],
        "Left": [[507,536], [480,500], [449,468], [434,447], [425,433],
                 [473,434], [468,394], [465,365], [465,345],
                 [489,433], [488,389], [487,354], [486,327],
                 [506,441], [506,399], [506,367], [506,336],
                 [523,450], [529,420], [532,394], [535,373]],
    }
    dimensions = [956, 720]
    if scene == 2:
        # Follow-up screenshot: right hand bent forward, left palm upright.
        # Coordinates are approximate and do not reconstruct real world depth.
        dimensions = [1917, 1011]
        skeletons = {
            "Right": [[742,663], [806,680], [859,700], [878,727], [864,748],
                      [822,655], [815,760], [807,755], [807,743],
                      [764,647], [761,766], [767,764], [766,742],
                      [710,647], [704,754], [716,758], [725,732],
                      [663,656], [655,733], [668,737], [684,718]],
            "Left": [[1169,827], [1100,803], [1049,751], [1014,707], [974,678],
                     [1073,669], [1061,600], [1051,556], [1043,518],
                     [1112,664], [1104,583], [1097,533], [1091,487],
                     [1150,675], [1148,600], [1143,550], [1135,507],
                     [1188,698], [1199,643], [1205,603], [1209,563]],
        }
    hands = {}
    for side, coordinates in skeletons.items():
        points = ((np.asarray(coordinates, dtype=np.float32) - [0, 23]) / dimensions).astype(np.float32)
        if mirror:
            points[:, 0] = 1 - points[:, 0]
            side = "Left" if side == "Right" else "Right"
        hands[side] = points
    return hands


def world_for_scale(points, scale):
    # Synthetic world XY tests arbitration under reconstruction mismatch.
    # A screenshot cannot supply measured world landmarks or camera distance.
    pixels = points * [SHAPE[1], SHAPE[0]]
    return np.column_stack(((pixels - pixels[0]) / scale, np.zeros(21))).astype(np.float32)


@pytest.fixture(scope="module")
def shipped_manager():
    manager = ModelManager(load_runtime_config())
    assert manager.ready, manager.errors
    return manager


@pytest.mark.parametrize("projected_ratio", [1.07, 1.4])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("scene", [1, 2])
def test_reported_palm_replaces_rejected_owner_using_shipped_classifier(shipped_manager, projected_ratio, mirror, scene):
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config = shipped_manager.config
    engine.model_manager = shipped_manager
    engine.resolver = GeometryResolver(engine.config)
    engine._lock = threading.Lock()
    engine.detector = simulated_detector()
    hands = reported_hands(mirror, scene)
    idle_side, palm_side = ("Left", "Right") if mirror else ("Right", "Left")
    scales = {idle_side: 1344., palm_side: 900.}
    palm_confirmed = False

    def candidates(rgb, view, *, video):
        return [(.1, points.copy(), SimpleNamespace(category_name=side,
                 score=.99 if side == idle_side or palm_confirmed else .1),
                 world_for_scale(points, scales[side])) for side, points in hands.items()]

    engine.detector._run_candidates = candidates
    frame = cv2.imencode(".jpg", np.full(SHAPE, 140, np.uint8))[1].tobytes()
    session = RuntimeSession(engine.config)
    # Start with both gestures rejected (palm handedness is not confirmed yet),
    # so the physical foreground hand owns the track without executing an action.
    for tick in range(10):
        result = engine.process_frame(frame, session, now=1 + tick * .1)
    idle_id = result["hand_selection"]["selected_track_id"]
    assert idle_id is not None
    assert result["quality"]["detector"]["selected_handedness"] == idle_side
    assert result["runtime_prediction"] == "no_gesture"

    # The other hand now qualifies as Palm. Its distance must not block fallback.
    palm_confirmed = True
    scales.update({idle_side: 1200. * projected_ratio, palm_side: 1200.})
    results = [engine.process_frame(frame, session, now=1 + tick * .1) for tick in range(10, 30)]
    handoff = next(result for result in results if result["hand_selection"]["ownership_reset"])
    assert handoff["runtime_action"] == "Wait / No Action"
    assert handoff["stable_frames"] == 1
    assert handoff["quality"]["detector"]["selected_handedness"] == palm_side
    final = results[-1]
    assert final["hand_selection"]["selected_track_id"] != idle_id
    assert final["hand_selection"]["gesture_priority"]
    assert not final["hand_selection"]["switch_pending"]
    assert final["runtime_prediction"] == "open_palm"
    np.testing.assert_allclose(final["landmarks"], hands[palm_side], atol=1e-6)
    assert final["diagnostics"]["ONNX"]["execute"]
    assert final["resolvers"]["pose_validation"]["valid"]
    selected = next(hand for hand in final["hands"] if hand["selected"])
    assert selected["handedness"] == palm_side
    assert selected["gesture"] == "open_palm" and selected["eligible"]
    assert all(hand["gesture"] == "no_gesture" for hand in final["hands"] if not hand["selected"])
