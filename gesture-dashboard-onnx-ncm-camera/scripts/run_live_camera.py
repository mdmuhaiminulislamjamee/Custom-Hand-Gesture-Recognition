#!/usr/bin/env python3
"""
Standalone eleven-command real-time gesture recognition with ONNX
Runs directly in an OpenCV window on your PC without starting the web server.

Usage:
  python scripts/run_live_camera.py                 # Uses default PC webcam
  python scripts/run_live_camera.py --source ncm    # Connects to USB-NCM Board Camera (192.168.50.2:5000)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure thread limit for OpenBLAS on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from backend.config import load_runtime_config
from backend.model_runtime import InferenceEngine, RuntimeSession
from backend.ncm_camera import NcmCameraClient, NcmCameraConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live eleven-command ONNX gesture recognition"
    )
    parser.add_argument(
        "--source",
        choices=["webcam", "ncm"],
        default="webcam",
        help="Video source: 'webcam' (default) or 'ncm' (development board)",
    )
    parser.add_argument(
        "--webcam-id",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    args = parser.parse_args()

    print("Loading models and runtime configuration...")
    config = load_runtime_config()
    engine = InferenceEngine(config)
    session = RuntimeSession(config)

    print(f"Commands ({len(config.gesture_to_action)}):")
    for gesture, action in config.gesture_to_action.items():
        print(f"  {gesture:<12} -> {action}")
    print("Dorsal is a hard negative: no_gesture, with no action.")
    print("Hand-only detection: body position and clothing do not block gestures.")
    print("A valid gesture takes priority over the other hand's no_gesture, regardless of distance.")
    print("Face filtering removed: gestures can qualify in front of the face.\n")

    status = engine.status()
    if not status.get("ready"):
        print(f"Error: InferenceEngine is not ready: {status}")
        return

    print("Models loaded successfully!")
    print("Press 'q' or 'ESC' to exit; press 'r' to reset hand selection.\n")

    if args.source == "ncm":
        print("Connecting to USB-NCM camera at 192.168.50.2:5000...")
        ncm_config = NcmCameraConfig(host_ip="192.168.50.1", device_ip="192.168.50.2", tcp_port=5000)
        client = NcmCameraClient(ncm_config)
        client.start()
        
        window_title = "ONNX Gesture Recognition - USB-NCM Camera (q: exit, r: reset)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 960, 720)

        last_frame_id = -1
        try:
            while True:
                frame_id, jpeg_bytes = client.wait_for_frame(last_frame_id, timeout_seconds=0.2)
                if jpeg_bytes is None or frame_id == last_frame_id:
                    # Check connection status
                    status = client.status()
                    if not status.get("connected"):
                        display = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(
                            display,
                            f"Connecting to NCM ({status.get('state', 'connecting')})...",
                            (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 165, 255),
                            2,
                        )
                        cv2.imshow(window_title, display)
                        if _handle_key(cv2.waitKey(30), session):
                            break
                    continue

                last_frame_id = frame_id
                result = engine.process_frame(jpeg_bytes, session)
                
                # Decode for display
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    _draw_overlay(frame, result)
                    cv2.imshow(window_title, frame)

                if _handle_key(cv2.waitKey(1), session):
                    break
        finally:
            client.stop()
            cv2.destroyAllWindows()
            engine.close()

    else:
        print(f"Opening PC Webcam (Index {args.webcam_id})...")
        cap = cv2.VideoCapture(args.webcam_id)
        if not cap.isOpened():
            print(f"Error: Could not open webcam at index {args.webcam_id}.")
            return

        # The default driver mode is often 640x480. Request more source detail
        # for fingers at 1 m+; unsupported modes fall back through the driver.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"Webcam capture: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        window_title = "ONNX Gesture Recognition - PC Webcam (q: exit, r: reset)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 960, 720)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Failed to grab frame from webcam.")
                    break

                # Encode frame to JPEG bytes for InferenceEngine
                success, encoded = cv2.imencode(".jpg", frame)
                if not success:
                    continue

                result = engine.process_frame(encoded.tobytes(), session)
                _draw_overlay(frame, result)
                cv2.imshow(window_title, frame)

                if _handle_key(cv2.waitKey(1), session):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            engine.close()


def _handle_key(key: int, session: RuntimeSession) -> bool:
    key &= 0xFF
    if key == ord("r"):
        session.reset()
        print("Hand selection reset; hold a clear gesture to select the visible hand.")
    return key in (ord("q"), 27)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
    (5, 9), (9, 13), (13, 17),             # Knuckle bridge
]


def _draw_axis_arrow(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
    label_offset: tuple[int, int] = (4, -4),
) -> None:
    # Dark outline keeps the axes visible on both bright and dark camera frames.
    cv2.arrowedLine(frame, start, end, (15, 15, 15), 6, cv2.LINE_AA, tipLength=0.24)
    cv2.arrowedLine(frame, start, end, color, 3, cv2.LINE_AA, tipLength=0.24)
    cv2.putText(
        frame,
        label,
        (end[0] + label_offset[0], end[1] + label_offset[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (15, 15, 15),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (end[0] + label_offset[0], end[1] + label_offset[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def _unit_vector(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return None
    return vector / length


def _draw_hand_axes(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    world_landmarks: list | np.ndarray | None,
) -> None:
    """Draw a rotating hand-local XYZ triad with its origin on the wrist."""
    origin = np.asarray(points[0], dtype=np.float32)
    palm_center = np.asarray(
        [points[index] for index in (5, 9, 13, 17)], dtype=np.float32
    ).mean(axis=0)

    # Local +Y follows the wrist toward the fingers. Local +X runs from the
    # index-finger side toward the little-finger side of the palm.
    y_axis = _unit_vector(palm_center - origin)
    x_axis = _unit_vector(
        np.asarray(points[17], dtype=np.float32)
        - np.asarray(points[5], dtype=np.float32)
    )
    if x_axis is None or y_axis is None:
        return

    axis_length = float(np.clip(min(frame.shape[:2]) * 0.10, 36.0, 58.0))

    def endpoint(axis: np.ndarray, scale: float = 1.0) -> tuple[int, int]:
        target = origin + axis * axis_length * scale
        return int(round(float(target[0]))), int(round(float(target[1])))

    origin_px = int(round(float(origin[0]))), int(round(float(origin[1])))
    _draw_axis_arrow(frame, origin_px, endpoint(x_axis), (0, 55, 255), "X")
    _draw_axis_arrow(frame, origin_px, endpoint(y_axis), (40, 235, 40), "Y")

    z_projection = None
    z_depth = -1.0
    if world_landmarks is not None:
        world = np.asarray(world_landmarks, dtype=np.float32)
        if world.shape == (21, 3) and np.isfinite(world).all():
            world_y = _unit_vector(world[[5, 9, 13, 17]].mean(axis=0) - world[0])
            world_x_raw = world[17] - world[5]
            if world_y is not None:
                world_x = _unit_vector(
                    world_x_raw - float(np.dot(world_x_raw, world_y)) * world_y
                )
                if world_x is not None:
                    world_z = _unit_vector(np.cross(world_x, world_y))
                    if world_z is not None:
                        z_projection = world_z[:2]
                        z_depth = float(world_z[2])

    # When the palm normal points mostly into/out of the image, use the usual
    # circle-dot/circle-cross depth symbol rather than an unreadably short arrow.
    z_color = (255, 110, 35)
    if z_projection is None or float(np.linalg.norm(z_projection)) < 0.22:
        cv2.circle(frame, origin_px, 12, (15, 15, 15), 5, cv2.LINE_AA)
        cv2.circle(frame, origin_px, 12, z_color, 2, cv2.LINE_AA)
        if z_depth <= 0.0:
            cv2.circle(frame, origin_px, 3, z_color, -1, cv2.LINE_AA)
        else:
            cv2.line(frame, (origin_px[0] - 5, origin_px[1] - 5),
                     (origin_px[0] + 5, origin_px[1] + 5), z_color, 2, cv2.LINE_AA)
            cv2.line(frame, (origin_px[0] - 5, origin_px[1] + 5),
                     (origin_px[0] + 5, origin_px[1] - 5), z_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "Z", (origin_px[0] + 14, origin_px[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(frame, "Z", (origin_px[0] + 14, origin_px[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, z_color, 1, cv2.LINE_AA)
    else:
        _draw_axis_arrow(frame, origin_px, endpoint(z_projection), z_color, "Z")

    # Keep the requested yellow wrist point visible as the common origin.
    cv2.circle(frame, origin_px, 5, (0, 165, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, origin_px, 6, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_corner_brackets(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int = 2,
    corner_len: int = 24,
) -> None:
    """Draws tech-styled corner brackets around a rectangle."""
    # Ensure points are ordered
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    c_len = min(corner_len, (x_max - x_min) // 3, (y_max - y_min) // 3)

    # Top-Left
    cv2.line(img, (x_min, y_min), (x_min + c_len, y_min), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x_min, y_min), (x_min, y_min + c_len), color, thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(img, (x_max, y_min), (x_max - c_len, y_min), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x_max, y_min), (x_max, y_min + c_len), color, thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(img, (x_min, y_max), (x_min + c_len, y_max), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x_min, y_max), (x_min, y_max - c_len), color, thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(img, (x_max, y_max), (x_max - c_len, y_max), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x_max, y_max), (x_max, y_max - c_len), color, thickness, cv2.LINE_AA)


def _draw_overlay(
    frame: np.ndarray,
    result: dict,
) -> None:
    """Draw targeting guide, hand skeleton, XYZ motion axes, and HUD banners."""
    h, w = frame.shape[:2]
    status = result.get("status", "")
    pred = result.get("runtime_prediction", "no_gesture")
    action = result.get("runtime_action", "Wait / No Action")
    reason = result.get("action_reason", "")
    conf = float(result.get("confidence", 0.0))
    mass = float(result.get("known_gesture_mass", 0.0))
    fps = float(result.get("actual_fps", 0.0))
    selected_preview = next((hand for hand in result.get("hands", []) if hand.get("selected")), {})
    landmarks = (result.get("display_landmarks") or selected_preview.get("display_landmarks")
                 or result.get("landmarks"))
    world_landmarks = result.get("world_landmarks")
    distance_m = result.get("distance_m")
    plane_angles = result.get("plane_angles")
    selection = result.get("hand_selection") or {}
    hands = result.get("hands") or []

    if distance_m is None and landmarks and len(landmarks) == 21:
        pts_arr = np.asarray(landmarks) * np.array([w, h])
        segs = [np.linalg.norm(pts_arr[a] - pts_arr[b]) for a, b in ((0, 5), (0, 9), (0, 17), (5, 17))]
        palm_scale = float(np.median(segs))
        if palm_scale >= 8:
            distance_m = round(float(np.clip((w * 0.85 * 0.075) / palm_scale, 0.15, 5.0)), 2)

    is_recognized = (pred != "no_gesture" and conf >= 0.8)

    # 1. Draw Centered Interaction Guide Frame
    gw = int(w * 0.52)
    gh = int(h * 0.58)
    gx1 = (w - gw) // 2
    gy1 = max(95, (h - gh) // 2)
    gx2 = gx1 + gw
    gy2 = min(h - 45, gy1 + gh)

    if landmarks and len(landmarks) == 21:
        # Faint guide frame when hand is active
        guide_color = (60, 60, 60)
        _draw_corner_brackets(frame, gx1, gy1, gx2, gy2, guide_color, thickness=1, corner_len=18)
    else:
        # Prompting guide frame when waiting for hand
        guide_color = (180, 180, 180)
        _draw_corner_brackets(frame, gx1, gy1, gx2, gy2, guide_color, thickness=2, corner_len=28)
        prompt_text = "Hold a clear gesture" if hands else "Place Hand Inside Frame"
        (tw, th), _ = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        cv2.putText(
            frame,
            prompt_text,
            (gx1 + (gw - tw) // 2, gy2 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # Show every observed hand, including blocked/ambiguous candidates. Only the
    # selected hand receives the existing gesture overlay and motion axes.
    for hand in hands:
        if hand.get("selected") and landmarks:
            continue
        points = hand.get("display_landmarks") or hand.get("landmarks") or []
        if len(points) != 21:
            continue
        pixels = [(int(point[0] * w), int(point[1] * h)) for point in points]
        color = (160, 160, 160)
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, pixels[start], pixels[end], color, 1, cv2.LINE_AA)
        for point in pixels:
            cv2.circle(frame, point, 2, color, -1, cv2.LINE_AA)
        identifier = f" #{hand['track_id']}" if hand.get("track_id") else ""
        state_label = (hand.get("reason") or "observed").replace("_", " ")
        label = f"{hand.get('handedness') or 'Hand'}{identifier} | {state_label}"
        x = max(5, min(w - 230, min(point[0] for point in pixels)))
        y = max(100, min(h - 65, min(point[1] for point in pixels) - 8))
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)

    # 2. Draw selected Hand Skeleton, bounding box, and rotating local axes
    if landmarks and len(landmarks) == 21:
        pts = [(int(p[0] * w), int(p[1] * h)) for p in landmarks]

        # Determine theme color
        if is_recognized:
            box_color = (50, 220, 50)     # Green
            skel_color = (80, 240, 120)
        elif pred != "no_gesture":
            box_color = (0, 200, 255)     # Orange/Amber (pending/low confidence)
            skel_color = (0, 220, 255)
        else:
            box_color = (200, 200, 200)   # Neutral Gray
            skel_color = (200, 210, 220)

        # Draw skeleton bone lines
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            cv2.line(frame, pts[p1_idx], pts[p2_idx], skel_color, 2, cv2.LINE_AA)

        # Draw joint circles
        for idx, (px, py) in enumerate(pts):
            if idx == 0:
                # Wrist
                cv2.circle(frame, (px, py), 6, (0, 165, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)
            elif idx in (4, 8, 12, 16, 20):
                # Fingertips
                cv2.circle(frame, (px, py), 5, box_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 3, (240, 240, 240), -1, cv2.LINE_AA)

        # Calculate bounding box
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 25
        bx1 = max(5, min(xs) - pad)
        by1 = max(90, min(ys) - pad)
        bx2 = min(w - 5, max(xs) + pad)
        by2 = min(h - 5, max(ys) + pad)

        # Draw bounding box rectangle + corner brackets
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 1, cv2.LINE_AA)
        _draw_corner_brackets(frame, bx1, by1, bx2, by2, box_color, thickness=3, corner_len=22)

        # Draw Badge/Tag over the hand box
        dist_suffix = f" | ~{distance_m:.2f}m" if distance_m is not None else ""
        if is_recognized:
            tag_label = f"{pred.upper()} ({conf*100:.0f}%){dist_suffix}"
        elif pred != "no_gesture":
            tag_label = f"DETECTING...{dist_suffix}"
        else:
            tag_label = f"HAND{dist_suffix}"
        if selection.get("selected_track_id") is not None:
            tag_label = f"#{selection['selected_track_id']} SELECTED | {tag_label}"
        (lw, lh), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tag_y1 = max(90, by1 - lh - 10)
        tag_y2 = tag_y1 + lh + 8
        tag_x2 = min(w - 5, bx1 + lw + 16)
        cv2.rectangle(frame, (bx1, tag_y1), (tag_x2, tag_y2), box_color, -1)
        cv2.putText(
            frame,
            tag_label,
            (bx1 + 8, tag_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )

        # Rotating hand-local axes, anchored at landmark 0 (the yellow wrist).
        _draw_hand_axes(frame, pts, world_landmarks)

    # 3. Top HUD Banner
    cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)
    cv2.line(frame, (0, 85), (w, 85), (50, 50, 50), 1)

    # Prediction text
    if is_recognized:
        pred_color = (50, 220, 50)   # Green
        pred_text = f"Gesture: {pred.upper()} ({conf*100:.1f}%)"
    elif pred != "no_gesture":
        pred_color = (0, 200, 255)   # Amber
        pred_text = f"Gesture: {pred.upper()} ({conf*100:.1f}%) [HOLD]"
    else:
        pred_color = (170, 170, 170) # Gray
        pred_text = "Gesture: NO GESTURE"

    cv2.putText(frame, pred_text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, pred_color, 2, cv2.LINE_AA)

    # Action text
    if action != "Wait / No Action":
        action_color = (0, 230, 230)  # Cyan/Yellow
    else:
        action_color = (150, 150, 150)
    action_text = f"Action: {action}"
    if reason:
        action_text += f" ({reason})"
    cv2.putText(frame, action_text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.62, action_color, 2, cv2.LINE_AA)

    # Stats on top right
    dist_hud = f"Dist: ~{distance_m:.2f}m | " if distance_m is not None else ""
    stats_text = f"{dist_hud}Mass: {mass:.2f} | FPS: {fps:.1f}"
    (sw, _), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
    cv2.putText(frame, stats_text, (max(20, w - sw - 20), 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)

    # 4. Bottom HUD Banner: 3D Palm Angles (XY, YZ, ZX) and Distance
    bar_h = 60
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (18, 18, 18), -1)
    cv2.line(frame, (0, h - bar_h), (w, h - bar_h), (45, 45, 45), 1)
    selection_text = selection.get("message", "Automatic hand selection")
    if selection.get("reason") == "active_hand_missing":
        selection_text += " Press R to reset now."
    label = f"{len(hands)} hand(s) | {selection_text}"
    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]
    text_scale = min(.45, .45 * max(100, w - 24) / max(text_width, 1))
    cv2.putText(frame, label, (12, h - 39), cv2.FONT_HERSHEY_SIMPLEX,
                text_scale, (220, 200, 140), 1, cv2.LINE_AA)

    if plane_angles:
        xy_val = plane_angles.get("xy", 0.0)
        yz_val = plane_angles.get("yz", 0.0)
        zx_val = plane_angles.get("zx", plane_angles.get("xz", 0.0))
        angles_str = f"XY: {xy_val:+.0f}deg   YZ: {yz_val:+.0f}deg   ZX: {zx_val:+.0f}deg"
        angles_color = (0, 230, 230)
    else:
        angles_str = "XY: --deg   YZ: --deg   ZX: --deg"
        angles_color = (130, 130, 130)

    dist_display = f"~{distance_m:.2f} m" if distance_m is not None else "--"
    bottom_left = f"3D ANGLES:  {angles_str}"
    bottom_right = f"DISTANCE: {dist_display}"

    cv2.putText(frame, bottom_left, (20, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.50, angles_color, 1, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(frame, bottom_right, (max(20, w - rw - 20), h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 200), 1, cv2.LINE_AA)



if __name__ == "__main__":
    main()
