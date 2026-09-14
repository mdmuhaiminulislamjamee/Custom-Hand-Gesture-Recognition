#!/usr/bin/env python3
"""
Standalone Real-Time Gesture Recognition with ONNX
Runs directly in an OpenCV window on your PC without starting the web server.

Usage:
  python scripts/run_live_camera.py                 # Uses default PC webcam
  python scripts/run_live_camera.py --source ncm    # Connects to USB-NCM Board Camera (192.168.50.2:5000)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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
    parser = argparse.ArgumentParser(description="Live ONNX Gesture Recognition")
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

    status = engine.status()
    if not status.get("ready"):
        print(f"Error: InferenceEngine is not ready: {status}")
        return

    print("Models loaded successfully!")
    print("Press 'q' or 'ESC' in the camera window to exit.\n")

    if args.source == "ncm":
        print("Connecting to USB-NCM camera at 192.168.50.2:5000...")
        ncm_config = NcmCameraConfig(host_ip="192.168.50.1", device_ip="192.168.50.2", tcp_port=5000)
        client = NcmCameraClient(ncm_config)
        client.start()
        
        window_title = "ONNX Gesture Recognition - USB-NCM Camera (Press 'q' to exit)"
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
                        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                            break
                    continue

                last_frame_id = frame_id
                result = engine.process_frame(jpeg_bytes, session)
                
                # Decode for display
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    _draw_overlay(frame, result)
                    cv2.imshow(window_title, frame)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
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

        window_title = "ONNX Gesture Recognition - PC Webcam (Press 'q' to exit)"
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

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            engine.close()


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
    (5, 9), (9, 13), (13, 17),             # Knuckle bridge
]


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


def _draw_overlay(frame: np.ndarray, result: dict) -> None:
    """Draws targeting guide frame, hand bounding box, skeleton, and HUD banner."""
    h, w = frame.shape[:2]
    status = result.get("status", "")
    pred = result.get("runtime_prediction", "no_gesture")
    action = result.get("runtime_action", "Wait / No Action")
    reason = result.get("action_reason", "")
    conf = float(result.get("confidence", 0.0))
    mass = float(result.get("known_gesture_mass", 0.0))
    fps = float(result.get("actual_fps", 0.0))
    landmarks = result.get("landmarks")

    is_recognized = (pred != "no_gesture" and conf >= 0.8)

    # 1. Draw Centered Interaction Guide Frame
    gw = int(w * 0.52)
    gh = int(h * 0.62)
    gx1 = (w - gw) // 2
    gy1 = max(95, (h - gh) // 2)
    gx2 = gx1 + gw
    gy2 = gy1 + gh

    if landmarks and len(landmarks) == 21:
        # Faint guide frame when hand is active
        guide_color = (60, 60, 60)
        _draw_corner_brackets(frame, gx1, gy1, gx2, gy2, guide_color, thickness=1, corner_len=18)
    else:
        # Prompting guide frame when waiting for hand
        guide_color = (180, 180, 180)
        _draw_corner_brackets(frame, gx1, gy1, gx2, gy2, guide_color, thickness=2, corner_len=28)
        prompt_text = "Place Hand Inside Frame"
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

    # 2. Draw Hand Skeleton & Bounding Box if Hand is Detected
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
        tag_label = f"{pred.upper()} ({conf*100:.0f}%)" if is_recognized else ("DETECTING..." if pred != "no_gesture" else "HAND")
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
    stats_text = f"Mass: {mass:.2f} | FPS: {fps:.1f}"
    cv2.putText(frame, stats_text, (w - 250, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)



if __name__ == "__main__":
    main()

