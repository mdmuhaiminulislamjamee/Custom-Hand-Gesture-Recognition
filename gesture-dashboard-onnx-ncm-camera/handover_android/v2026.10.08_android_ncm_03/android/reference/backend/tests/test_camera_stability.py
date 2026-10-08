from __future__ import annotations

from types import SimpleNamespace
import sys

import numpy as np

from backend.config import RuntimeConfig
from backend.model_runtime import (
    HandDetector,
    InferenceEngine,
    RuntimeSession,
    adaptive_low_light_preprocess,
)
import backend.model_runtime as model_runtime


def representative_hand() -> np.ndarray:
    points = np.zeros((21, 2), dtype=np.float32)
    points[0] = [0.5, 0.9]
    bases = [0.32, 0.43, 0.54, 0.65, 0.76]
    for finger, base_x in enumerate(bases):
        start = 1 if finger == 0 else 5 + (finger - 1) * 4
        for joint in range(4):
            points[start + joint] = [
                base_x + 0.015 * joint,
                0.78 - 0.13 * joint,
            ]
    return points


def test_low_light_preprocessing_brightens_without_mirroring() -> None:
    image = np.full((64, 96, 3), 22, dtype=np.uint8)
    image[12:52, 8:30] = 65

    enhanced, quality = adaptive_low_light_preprocess(image)

    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8
    assert quality["low_light"] is True
    assert quality["enhancement_applied"] is True
    assert quality["input_mirrored"] is False
    assert quality["output_luma_mean"] > quality["input_luma_mean"]
    assert enhanced[:, :48].mean() > enhanced[:, 48:].mean()


def test_near_black_frame_is_not_usable_for_detection() -> None:
    _, quality = adaptive_low_light_preprocess(
        np.zeros((48, 48, 3), dtype=np.uint8)
    )
    assert quality["near_blank"] is True
    assert quality["usable_for_detection"] is False


def test_well_lit_frame_is_not_needlessly_modified() -> None:
    image = np.full((48, 64, 3), 145, dtype=np.uint8)
    image[:, 10:20] = 190

    processed, quality = adaptive_low_light_preprocess(image)

    assert quality["low_light"] is False
    assert quality["enhancement_applied"] is False
    assert np.array_equal(processed, image)


def test_near_blank_process_frame_suppresses_detection_and_exposes_quality() -> None:
    import cv2

    class DetectorThatMustNotRun:
        ready = True
        error = None
        running_mode = "VIDEO"

        def detect(self, _image):
            raise AssertionError("MediaPipe should be skipped for a blank frame")

    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config = RuntimeConfig()
    engine.model_manager = SimpleNamespace(ready=True)
    engine.detector = DetectorThatMustNotRun()
    encoded_ok, encoded = cv2.imencode(
        ".jpg", np.zeros((64, 64, 3), dtype=np.uint8)
    )
    assert encoded_ok

    result = engine.process_frame(encoded.tobytes(), RuntimeSession(engine.config))

    assert result["status"] == "no_hand"
    assert result["runtime_prediction"] == engine.config.reject_label
    assert result["quality"]["classification_allowed"] is False
    assert result["quality"]["detector"]["skipped"] is True
    assert "near_blank_frame" in result["quality"]["issues"]
    assert "preprocess_ms" in result["timing"]


def test_landmark_filter_reduces_stationary_jitter_and_is_session_local() -> None:
    config = RuntimeConfig()
    first_session = RuntimeSession(config)
    second_session = RuntimeSession(config)
    base = representative_hand()
    jitter = np.tile(np.asarray([0.006, -0.004], dtype=np.float32), (21, 1))

    initial, initial_quality = first_session.stabilize_landmarks(base, now=1.0)
    smoothed, quality = first_session.stabilize_landmarks(base + jitter, now=1.05)
    independent, independent_quality = second_session.stabilize_landmarks(
        base + jitter, now=1.05
    )

    assert initial is not None and smoothed is not None and independent is not None
    assert initial_quality["reason"] == "filter_initialized"
    assert quality["reason"] == "adaptively_smoothed"
    # At 20 FPS the per-frame weight is lower to retain the 10 FPS time constant.
    assert 0.09 <= quality["smoothing_alpha"] < 0.18
    assert np.linalg.norm(smoothed - initial) < np.linalg.norm(jitter)
    assert np.allclose(independent, base + jitter)
    assert independent_quality["reason"] == "filter_initialized"


def test_landmark_filter_rejects_one_frame_jump_then_reacquires() -> None:
    session = RuntimeSession(RuntimeConfig())
    base = representative_hand() * 0.30 + np.asarray([0.10, 0.10], dtype=np.float32)
    jump = base + np.asarray([0.35, 0.0], dtype=np.float32)

    initial, _ = session.stabilize_landmarks(base, now=2.0)
    rejected, rejected_quality = session.stabilize_landmarks(jump, now=2.05)
    reacquired, reacquired_quality = session.stabilize_landmarks(
        jump + 0.0005, now=2.10
    )

    assert initial is not None
    assert rejected is None
    assert rejected_quality["jump_rejected"] is True
    assert rejected_quality["reason"] == "isolated_landmark_jump"
    assert reacquired is not None
    assert reacquired_quality["reason"] == "persistent_jump_reacquired"
    assert reacquired_quality["filter_reset"] is True


def test_landmark_filter_resets_after_three_missing_frames() -> None:
    session = RuntimeSession(RuntimeConfig())
    session.stabilize_landmarks(representative_hand(), now=3.0)

    session.observe_missing_landmarks(now=3.05)
    session.observe_missing_landmarks(now=3.10)
    quality = session.observe_missing_landmarks(now=3.15)
    reacquired, reacquired_quality = session.stabilize_landmarks(
        representative_hand(), now=3.20
    )

    assert quality["filter_reset"] is True
    assert reacquired is not None
    assert reacquired_quality["reason"] == "filter_initialized"


def test_hand_detector_uses_video_mode_and_strict_monotonic_timestamps(
    monkeypatch, tmp_path
) -> None:
    timestamps: list[int] = []
    captured_options: dict[str, object] = {}
    video_mode = object()

    class FakeLandmarker:
        spacing = 0.10

        def detect_for_video(self, _image, timestamp_ms):
            timestamps.append(timestamp_ms)
            points = [
                SimpleNamespace(
                    x=float(index % 5) * self.spacing + 0.2,
                    y=float(index // 5) * self.spacing + 0.2,
                )
                for index in range(21)
            ]
            handedness = [[SimpleNamespace(category_name="Right", score=0.91)]]
            return SimpleNamespace(hand_landmarks=[points], handedness=handedness)

        def detect(self, _image):
            return SimpleNamespace(hand_landmarks=[], handedness=[])

        def close(self):
            return None

    landmarker = FakeLandmarker()

    class FakeOptions:
        def __init__(self, **kwargs):
            if kwargs.get("running_mode") is video_mode:
                captured_options.update(kwargs)

    fake_mp = SimpleNamespace(
        tasks=SimpleNamespace(
            BaseOptions=lambda **kwargs: SimpleNamespace(**kwargs),
            vision=SimpleNamespace(
                RunningMode=SimpleNamespace(VIDEO=video_mode, IMAGE=object()),
                HandLandmarkerOptions=FakeOptions,
                HandLandmarker=SimpleNamespace(
                    create_from_options=lambda _options: landmarker
                ),
            ),
        ),
        Image=lambda **kwargs: SimpleNamespace(**kwargs),
        ImageFormat=SimpleNamespace(SRGB="srgb"),
    )
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setattr(model_runtime.time, "monotonic_ns", lambda: 1_000_000)
    model_path = tmp_path / "hand_landmarker.task"
    model_path.write_bytes(b"fake")

    detector = HandDetector(model_path)
    first = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    second = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    landmarker.spacing = 0.01
    rejected_small_hand = detector.detect(
        np.zeros((320, 320, 3), dtype=np.uint8)
    )

    assert detector.ready is True
    assert captured_options["running_mode"] is video_mode
    assert captured_options["min_hand_detection_confidence"] == 0.55
    assert captured_options["min_tracking_confidence"] == 0.60
    assert first is not None and second is not None
    assert rejected_small_hand is None
    assert timestamps == [1, 2, 3]
    assert detector.diagnostics()["timestamp_ms"] == 3
    assert detector.diagnostics()["selected_handedness_confidence"] == 0.91
    assert detector.diagnostics()["rejection_reason"] == "insufficient_hand_pixels"
