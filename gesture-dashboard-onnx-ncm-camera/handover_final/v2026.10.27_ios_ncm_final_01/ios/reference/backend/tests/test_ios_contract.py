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
    assert "isOpenPalmDirectionAllowed" in source
    assert "minimumNonDownAlignment: Float = -0.10" in source
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
    # Model confidence cannot bypass the closed-hand Fist geometry check.
    assert source.count("modelConfidence >= 0.85") == 1
    assert "let allowedDirection = try isOpenPalmDirectionAllowed(points)" in source
    assert "thumbExtension >= 0.40" in source
    assert "downScore >= 0.25" in source
    assert "lead >= 0.15" in source


def test_ios_geometry_and_temporal_sources_follow_the_twelve_class_contract() -> None:
    geometry = (PROJECT_ROOT / "swift" / "GestureGeometryResolver.swift").read_text()
    temporal = (PROJECT_ROOT / "swift" / "GestureTemporalGate.swift").read_text()
    assert 'raw == "peace" || raw == "rock"' in geometry
    assert "minimumPeaceTipSeparationRatio: Float = 0.04" in geometry
    assert "fistAllowedDirection && (fistCompact || fist.strongGeometry || fistSideGeometry || reportedSideFistGeometry)" in geometry
    assert "outside the twelve-gesture vocabulary" in temporal


def test_final_ios_handover_uses_ncm_frames_only() -> None:
    integration = (PROJECT_ROOT / "handover_docs" / "IOS_INTEGRATION.md").read_text()
    plist = (PROJECT_ROOT / "swift" / "ios" / "Info.plist.fragment").read_text()
    ncm = (PROJECT_ROOT / "swift" / "ios" / "GestureNCMClient.swift").read_text()

    assert "only image source is the USB-NCM board camera" in integration
    assert "AVCaptureVideoDataOutput" not in integration
    assert "NSCameraUsageDescription" not in plist
    assert "NSLocalNetworkUsageDescription" in plist
    assert "NWConnection" in ncm
    assert "maximumPayload = 200 * 1024" in ncm
    assert "crc32(payload) == expectedCRC" in ncm
    assert "case 0x03" in ncm
    assert "case 0x10" in ncm
