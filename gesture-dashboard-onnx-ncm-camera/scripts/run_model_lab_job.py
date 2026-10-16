"""Isolated worker used by dashboard model jobs; never promotes a model."""
from __future__ import annotations

import argparse
from pathlib import Path

from backend.data_collection import atomic_json
from research.lab_training import audit_dataset, evaluate, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("audit", "train", "evaluate"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.kind == "audit":
        _, result = audit_dataset(args.dataset)
        atomic_json(args.output / "result.json", result)
        print(f"Audited {result['ingestion']['raw_images']} reviewed images; {result['ingestion']['usable_feature_rows']} usable hand-feature rows.", flush=True)
    elif args.kind == "train":
        if not 1 <= args.epochs <= 500:
            raise ValueError("epochs must be between 1 and 500")
        train(args.dataset, args.output, project, args.epochs)
    else:
        if args.candidate is None:
            raise ValueError("Select a completed training candidate first")
        evaluate(args.candidate, args.output, project)


if __name__ == "__main__":
    main()
