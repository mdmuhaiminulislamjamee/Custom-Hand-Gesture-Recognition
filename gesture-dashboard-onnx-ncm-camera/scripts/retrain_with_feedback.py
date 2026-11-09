#!/usr/bin/env python3
"""
One-command model retraining workflow with reviewed feedback.
Trains the 13-class MLP on existing dataset + hard negative feedback
(including newly collected cupped hand samples), audits directional feedback,
installs the new model into models/gesture_mlp_production.onnx, and rebuilds the manifest.

Usage:
  python scripts/retrain_with_feedback.py
  python scripts/retrain_with_feedback.py --epochs 100
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(description: str, cmd: list[str]) -> None:
    print(f"\n>>> [STEP] {description}...")
    print(f"    Command: {' '.join(cmd)}")
    start = time.perf_counter()
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    process = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    elapsed = time.perf_counter() - start
    if process.returncode != 0:
        print(f"\n[FAILED] Step failed with return code {process.returncode}: {description}")
        sys.exit(process.returncode)
    print(f">>> [DONE] {description} ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Retrain gesture recognition ONNX model with feedback.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs (default: 100)")
    args = parser.parse_args()

    python_bin = sys.executable

    print("\n" + "=" * 65)
    print("      GESTURE RECOGNITION RETRAINING PIPELINE      ")
    print("=" * 65)
    print(f"Python: {python_bin}")
    print(f"Epochs: {args.epochs}")
    print("Feedback log: feedback/misclassification_log.csv")
    print("Target model: models/gesture_mlp_production.onnx")
    print("=" * 65)

    # 1. Train candidate MLP
    run_step(
        "Training candidate MLP with feedback",
        [python_bin, "scripts/train_stop_inverted.py", "--epochs", str(args.epochs)],
    )

    # 2. Audit directional feedback against candidate
    candidate_model = PROJECT_ROOT / "artifacts" / "stop_inverted" / "candidate.onnx"
    candidate_feedback = PROJECT_ROOT / "artifacts" / "stop_inverted" / "candidate_feedback.json"
    run_step(
        "Auditing candidate directional feedback & stress transformations",
        [
            python_bin,
            "scripts/audit_directional_feedback.py",
            "--model", str(candidate_model),
            "--output", str(candidate_feedback),
            "--stress",
            "--images",
        ],
    )

    # 3. Install qualified candidate into production
    run_step(
        "Installing qualified ONNX model into production",
        [python_bin, "scripts/train_stop_inverted.py", "--install"],
    )

    # 4. Rebuild artifact manifest
    run_step(
        "Rebuilding artifact manifest (SHA-256 integrity)",
        [python_bin, "scripts/build_artifact_manifest.py"],
    )

    # 5. Run regression tests
    run_step(
        "Running qualification test suite",
        [
            python_bin,
            "-m", "pytest",
            "backend/tests/test_onnx_runtime.py",
            "backend/tests/test_dorsal_and_fist_stability.py",
            "backend/tests/test_false_hand_rejection.py",
        ],
    )

    print("\n" + "=" * 65)
    print(" SUCCESS! RETRAINING & DEPLOYMENT COMPLETE ")
    print("=" * 65)
    print("The updated ONNX model is now active in models/gesture_mlp_production.onnx.")
    print("Cupped/claw hand samples are now learned as 'no_gesture' (REJECT).")
    print("\nTo test live recognition:")
    print("  python scripts/run_live_camera.py --source ncm\n")


if __name__ == "__main__":
    main()
