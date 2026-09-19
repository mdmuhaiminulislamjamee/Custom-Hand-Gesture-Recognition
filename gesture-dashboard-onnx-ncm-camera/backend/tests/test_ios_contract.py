from __future__ import annotations

import re
from pathlib import Path

from backend.config import CLASS_NAMES, GESTURE_TO_ACTION, REJECT_LABEL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWIFT_SOURCE = PROJECT_ROOT / "swift" / "GestureFeatureExtractor.swift"


def _quoted_values(block: str) -> list[str]:
    return re.findall(r'"([^"]+)"', block)


def test_swift_command_order_and_actions_match_the_runtime_contract() -> None:
    source = SWIFT_SOURCE.read_text(encoding="utf-8")
    class_block = re.search(
        r"public static let classNames\s*=\s*\[(.*?)\]",
        source,
        re.DOTALL,
    )
    assert class_block is not None
    assert _quoted_values(class_block.group(1)) == CLASS_NAMES

    action_block = re.search(
        r"public static let gestureToAction: \[String: String\]\s*=\s*\[(.*?)\]",
        source,
        re.DOTALL,
    )
    assert action_block is not None
    swift_actions = dict(
        re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', action_block.group(1))
    )
    assert swift_actions == GESTURE_TO_ACTION
    assert f'public static let rejectLabel = "{REJECT_LABEL}"' in source


def test_swift_has_the_cross_platform_pose_guards() -> None:
    source = SWIFT_SOURCE.read_text(encoding="utf-8")
    assert "isOpenPalmPointingUpward" in source
    assert "minimumUpAlignment: Float = 0.70" in source
    assert "minimumAxisDominance: Float = 0.70" in source
    assert "isFistPose" in source
    assert "isForeshortenedFistPose" in source
    assert "meanCurlback >= 0.16" in source
    assert "maximumTipRadius <= 0.85" in source
    assert "thumbRadius <= 0.85" in source
    assert "landmarksXY: [SIMD2<Float>]? = nil" in source
    assert 'raw != "like"' in source
    assert 'raw != "thumb_down"' in source
    assert "isThumbDownPose" in source
    assert "runtimePoseAllowed" in source
    # Keep the confidence-assisted side-view guards aligned with the Python
    # runtime; otherwise iOS silently rejects poses accepted by the dashboard.
    assert source.count("modelConfidence >= 0.85") == 2
    assert "thumbExtension >= 0.40" in source
    assert "downScore >= 0.25" in source
    assert "lead >= 0.15" in source
