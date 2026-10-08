from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in ("request_id", "method", "path", "status_code", "elapsed_ms"):
            if hasattr(record, name):
                payload[name] = getattr(record, name)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"), default=str)


def configure_logging(directory: Path) -> logging.Logger:
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gesture_dashboard")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = RotatingFileHandler(
            directory / "backend.jsonl",
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


class RuntimeMetrics:
    def __init__(self, window_size: int = 1000):
        self.window_size = int(window_size)
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._timings: dict[str, deque[float]] = {
            name: deque(maxlen=self.window_size)
            for name in (
                "total_ms",
                "mediapipe_ms",
                "feature_ms",
                "classifier_ms",
                "effective_application_fps",
            )
        }
        self._statuses: Counter[str] = Counter()
        self._models: Counter[str] = Counter()
        self._actions: Counter[str] = Counter()
        self.frames_total = 0
        self.errors_total = 0
        self.budget_pass_total = 0
        self.actions_total = 0
        self.active_sessions = 0
        self.peak_sessions = 0

    def session_opened(self) -> None:
        with self._lock:
            self.active_sessions += 1
            self.peak_sessions = max(self.peak_sessions, self.active_sessions)

    def session_closed(self) -> None:
        with self._lock:
            self.active_sessions = max(0, self.active_sessions - 1)

    def record_error(self, status: str = "internal_error") -> None:
        with self._lock:
            self.errors_total += 1
            self._statuses[status] += 1

    def record_frame(self, result: dict[str, Any]) -> None:
        timing = result.get("timing") or {}
        with self._lock:
            self.frames_total += 1
            status = str(result.get("status", "unknown"))
            self._statuses[status] += 1
            if status not in {"predicted", "no_hand"}:
                self.errors_total += 1
            model = result.get("model")
            if model:
                self._models[str(model)] += 1
            action = result.get("runtime_action")
            if action and action != "Wait / No Action":
                self.actions_total += 1
                self._actions[str(action)] += 1
            if timing.get(
                "target_fps_capacity_pass",
                timing.get("ten_fps_capacity_pass"),
            ) is True:
                self.budget_pass_total += 1
            for name, values in self._timings.items():
                value = timing.get(name)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    values.append(float(value))

    @staticmethod
    def _distribution(values: deque[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(array),
            "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            frames = self.frames_total
            uptime = max(0.0, time.time() - self.started_at)
            return {
                "uptime_seconds": uptime,
                "window_size": self.window_size,
                "frames_total": frames,
                "errors_total": self.errors_total,
                "error_rate": self.errors_total / frames if frames else 0.0,
                "budget_pass_total": self.budget_pass_total,
                "target_fps_budget_pass_rate": self.budget_pass_total / frames if frames else 0.0,
                # Backward-compatible API alias for older dashboard clients.
                "ten_fps_budget_pass_rate": self.budget_pass_total / frames if frames else 0.0,
                "actions_total": self.actions_total,
                "active_sessions": self.active_sessions,
                "peak_sessions": self.peak_sessions,
                "statuses": dict(self._statuses),
                "models": dict(self._models),
                "actions": dict(self._actions),
                "timing": {
                    name: self._distribution(values)
                    for name, values in self._timings.items()
                },
            }

    def prometheus(self) -> str:
        values = self.snapshot()
        total = values["timing"]["total_ms"]
        lines = [
            "# HELP gesture_frames_total Frames processed by the inference service.",
            "# TYPE gesture_frames_total counter",
            f"gesture_frames_total {values['frames_total']}",
            "# TYPE gesture_errors_total counter",
            f"gesture_errors_total {values['errors_total']}",
            "# TYPE gesture_actions_total counter",
            f"gesture_actions_total {values['actions_total']}",
            "# TYPE gesture_active_sessions gauge",
            f"gesture_active_sessions {values['active_sessions']}",
            "# TYPE gesture_pipeline_total_ms gauge",
            f"gesture_pipeline_total_ms{{quantile=\"0.50\"}} {total['p50'] or 0.0}",
            f"gesture_pipeline_total_ms{{quantile=\"0.95\"}} {total['p95'] or 0.0}",
            f"gesture_pipeline_total_ms{{quantile=\"0.99\"}} {total['p99'] or 0.0}",
            "# HELP gesture_target_fps_budget_pass_rate Fraction of frames within the configured frame budget.",
            "# TYPE gesture_target_fps_budget_pass_rate gauge",
            f"gesture_target_fps_budget_pass_rate {values['target_fps_budget_pass_rate']}",
        ]
        return "\n".join(lines) + "\n"
