#!/usr/bin/env python3
"""
Dedicated collector for cupped/claw hand hard negatives.
Captures clean samples of the cupped hand posture and logs them directly
into feedback/misclassification_log.csv as 'no_gesture'.

Usage:
  python scripts/collect_cupped_hand_negatives.py --source ncm --count 40
  python scripts/collect_cupped_hand_negatives.py --source webcam --count 40
"""
from __future__ import annotations

import argparse
import base64
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
from backend.storage import FeedbackRecord, FeedbackStore
from scripts.run_live_camera import LatestWebcamFrame, _draw_overlay


def main():
    parser = argparse.ArgumentParser(description="Collect cupped hand hard negative samples.")
    parser.add_argument("--source", choices=["ncm", "webcam"], default="ncm",
                        help="Camera source: 'ncm' (USB-NCM board) or 'webcam' (PC webcam)")
    parser.add_argument("--count", type=int, default=40,
                        help="Number of valid cupped hand samples to record (default: 40)")
    parser.add_argument("--webcam-id", type=int, default=0,
                        help="PC webcam index (default: 0)")
    parser.add_argument("--countdown", type=int, default=3,
                        help="Countdown seconds before recording starts (default: 3)")
    args = parser.parse_args()

    config = load_runtime_config()
    store = FeedbackStore(config)
    engine = InferenceEngine(config)
    session = RuntimeSession(config)

    target_count = max(10, args.count)
    countdown_duration = max(1, args.countdown)
    window_title = f"Cupped Hand Collector - {args.source.upper()} (Hold hand in view)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 960, 720)

    print(f"\n========================================================")
    print(f"        CUPPED HAND HARD-NEGATIVE DATA COLLECTOR        ")
    print(f"========================================================")
    print(f"Source: {args.source}")
    print(f"Target sample count: {target_count}")
    print(f"Countdown before recording: {countdown_duration} seconds")
    print(f"Goal: Train the model that this posture is 'NO_GESTURE'.")
    print(f"Instructions: When recording starts, slightly tilt and rotate")
    print(f"              your cupped hand in front of the camera.")
    print(f"Press 'q' or ESC in the preview window to cancel.")
    print(f"========================================================\n")

    if args.source == "ncm":
        ncm_cfg = NcmCameraConfig(host_ip="192.168.50.1", device_ip="192.168.50.2", tcp_port=5000)
        client = NcmCameraClient(ncm_cfg)
        client.start()
        source = None
    else:
        client = None
        cap = cv2.VideoCapture(args.webcam_id)
        if not cap.isOpened():
            print(f"Error: Could not open PC webcam at index {args.webcam_id}.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        source = LatestWebcamFrame(cap)
        source.start()

    start_time = time.monotonic()
    last_saved_time = 0.0
    saved_count = 0
    last_frame_id = -1

    try:
        while True:
            # Fetch frame
            if client is not None:
                frame_id, jpeg_bytes = client.wait_for_frame(last_frame_id, timeout_seconds=0.05)
                if jpeg_bytes is None or frame_id == last_frame_id:
                    st = client.status()
                    if not st.get("connected"):
                        display = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(display, f"Connecting to NCM ({st.get('state', 'connecting')})...",
                                    (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        cv2.imshow(window_title, display)
                        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                            break
                    continue
                last_frame_id = frame_id
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                frame_id, frame, _ = source.wait_for_frame(last_frame_id, timeout_seconds=0.05)
                if frame is None or frame_id == last_frame_id:
                    if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                        break
                    continue
                last_frame_id = frame_id

            if frame is None:
                continue

            now = time.monotonic()
            elapsed = now - start_time
            clean_frame = frame.copy()

            # Run inference
            result = engine.process_frame(frame, session, now=now)
            _draw_overlay(frame, result)

            h, w = frame.shape[:2]

            # Phase 1: Countdown
            if elapsed < countdown_duration:
                remaining_cd = int(np.ceil(countdown_duration - elapsed))
                overlay = frame.copy()
                cv2.rectangle(overlay, (20, 20), (w - 20, 120), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.rectangle(frame, (20, 20), (w - 20, 120), (0, 255, 255), 2)
                cv2.putText(frame, f"GET READY: Position Cupped Hand in view", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Recording starts in: {remaining_cd}...", (40, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

            # Phase 2: Active recording
            else:
                landmarks = result.get("landmarks")
                has_hand = bool(landmarks and len(landmarks) == 21)

                # Record rate-limited to 10 FPS
                if has_hand and (now - last_saved_time >= 0.10) and saved_count < target_count:
                    success, buffer = cv2.imencode(".jpg", clean_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if success:
                        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")
                        raw_pred = result.get("raw_prediction") or result.get("runtime_prediction") or "fist"
                        record = FeedbackRecord(
                            actual_label="no_gesture",
                            predicted_label=raw_pred,
                            runtime_action=result.get("runtime_action", "Wait / No Action"),
                            confidence=float(result.get("confidence", 0.0)),
                            model_name="ONNX",
                            probabilities=result.get("probabilities", {}),
                            feature_vector=result.get("feature_vector"),
                            landmarks=landmarks,
                            note="cupped_hand_negative",
                            snapshot_data_url=data_url,
                        )
                        store.save(record)
                        saved_count += 1
                        last_saved_time = now
                        pct = int(saved_count / target_count * 100)
                        print(f"Captured: {saved_count:2d}/{target_count} ({pct:3d}%) | Model said: {raw_pred} ({record.confidence:.2f})")

                # Draw collection HUD
                overlay = frame.copy()
                cv2.rectangle(overlay, (20, 20), (w - 20, 130), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                if saved_count >= target_count:
                    cv2.rectangle(frame, (20, 20), (w - 20, 130), (0, 255, 0), 2)
                    cv2.putText(frame, "COLLECTION COMPLETE!", (40, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Saved {saved_count} cupped hand samples to feedback log", (40, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow(window_title, frame)
                    cv2.waitKey(2000)
                    break
                else:
                    cv2.rectangle(frame, (20, 20), (w - 20, 130), (0, 200, 255), 2)
                    cv2.putText(frame, f"RECORDING CUPPED HAND: {saved_count} / {target_count}", (40, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    if has_hand:
                        status_str = "Hand detected - Slowly tilt / turn hand slightly"
                        color_str = (0, 255, 120)
                    else:
                        status_str = "No hand detected - hold hand inside the box"
                        color_str = (0, 100, 255)
                    cv2.putText(frame, status_str, (40, 92),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_str, 1, cv2.LINE_AA)

                    # Progress bar
                    bar_w = w - 80
                    filled_w = int(bar_w * (saved_count / target_count))
                    cv2.rectangle(frame, (40, 105), (40 + bar_w, 118), (50, 50, 50), -1)
                    cv2.rectangle(frame, (40, 105), (40 + filled_w, 118), (0, 255, 120), -1)

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("\n[CANCELLED] Data collection aborted by user.")
                break

    finally:
        if client is not None:
            client.stop()
        if source is not None:
            source.stop()
        cv2.destroyAllWindows()
        engine.close()

    if saved_count >= target_count:
        print(f"\n========================================================")
        print(f" SUCCESS! Recorded {saved_count} cupped-hand negative samples.")
        print(f" Saved to: feedback/misclassification_log.csv")
        print(f"========================================================")
        print(f"\nNow retrain your ONNX model with one command:")
        print(f"  python scripts/retrain_with_feedback.py\n")


if __name__ == "__main__":
    main()
