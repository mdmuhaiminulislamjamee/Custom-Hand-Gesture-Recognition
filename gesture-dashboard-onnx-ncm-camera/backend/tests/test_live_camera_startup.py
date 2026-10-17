"""The camera entry point must tolerate reserved, inactive model outputs."""
from types import SimpleNamespace

import pytest

from backend.config import load_runtime_config
from scripts import run_live_camera


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
    assert "Commands (11):" in output
    assert "Dorsal is a hard negative" in output
    assert "dorsal       ->" not in output
    assert "Return to Default Position" not in output
