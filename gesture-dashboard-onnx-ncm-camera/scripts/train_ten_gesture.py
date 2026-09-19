"""Build the balanced ten-command candidate without touching production files."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.v18_20 import prepare_data, train_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1820)
    arguments = parser.parse_args()
    root = Path.cwd()
    _, _, audit = prepare_data(
        root, per_class=arguments.per_class, seed=arguments.seed
    )
    print(
        f"Prepared {arguments.per_class} rows for each of 11 internal classes; "
        f"split subject overlap: {audit['subject_overlap']}",
        flush=True,
    )
    train_candidate(root, epochs=arguments.epochs, seed=arguments.seed)


if __name__ == "__main__":
    main()
