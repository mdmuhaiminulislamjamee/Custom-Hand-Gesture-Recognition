"""The camera entry point must tolerate reserved, inactive model outputs."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from backend.config import load_runtime_config
from scripts import run_live_camera


def test_responsive_profile_changes_runtime_timing_without_changing_gestures():
    config = load_runtime_config()
    actions = dict(config.gesture_to_action)
    classes = list(config.class_names)

    run_live_camera._apply_runtime_profile(config, "responsive")

    assert config.gesture_to_action == actions
    assert config.class_names == classes
    assert config.responsive_confirmation is True
    assert config.stable_frames_required == 2
    assert config.release_frames_required == 2
    assert config.minimum_hold_seconds == 0.0
    assert config.action_cooldown_seconds == 0.0


def test_webcam_reader_retains_the_latest_frame_only():
    class Capture:
        def __init__(self):
            self.frames = [
                np.full((2, 2, 3), value, dtype=np.uint8)
                for value in (1, 2, 3)
            ]
            self.released = False
            self.unblock = threading.Event()

        def read(self):
            if self.frames:
                return True, self.frames.pop(0)
            self.unblock.wait(0.2)
            return False, None

        def release(self):
            self.released = True
            self.unblock.set()

    capture = Capture()
    source = run_live_camera.LatestWebcamFrame(capture)
    source.start()
    frame_id, frame, captured_at = source.wait_for_frame(-1, timeout_seconds=0.5)
    while frame_id < 3:
        frame_id, frame, captured_at = source.wait_for_frame(
            frame_id, timeout_seconds=0.5
        )
    source.stop()

    assert frame_id == 3
    assert frame is not None and np.all(frame == 3)
    assert captured_at > 0.0
    assert capture.released is True


@pytest.mark.parametrize("source", ["webcam", "ncm"])
def test_startup_prints_active_commands_without_looking_up_retired_dorsal(monkeypatch, capsys, source):
    config = load_runtime_config()
    assert "dorsal" in config.class_names
    assert "dorsal" not in config.gesture_to_action
    monkeypatch.setattr(run_live_camera.sys, "argv", ["run_live_camera.py", "--source", source])
    monkeypatch.setattr(run_live_camera, "load_runtime_config", lambda: config)
    # Stop after the shared startup path, before accessing camera hardware.
    monkeypatch.setattr(run_live_camera, "InferenceEngine",
                        lambda config: SimpleNamespace(status=lambda: {"ready": False}))
    run_live_camera.main()
    output = capsys.readouterr().out
    assert "Commands (12):" in output
    assert "Stop inverted: Return to main position" in output
    assert "dorsal       ->" not in output
    assert "Return to Default Position" not in output
