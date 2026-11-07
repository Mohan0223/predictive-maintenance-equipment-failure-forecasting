"""Print a small provenance and schema summary for the staged AI4I CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from predictive_maintenance.data.dataset import load_ai4i_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CANDIDATES = (
    PROJECT_ROOT / "data" / "ai4i2020.csv",
    PROJECT_ROOT
    / "ai4i+2020+predictive+maintenance+dataset"
    / "ai4i2020.csv",
)
TARGET_COLUMN = "Machine failure"


def find_dataset() -> Path:
    """Return the first available copy of the AI4I CSV."""
    for candidate in DATASET_CANDIDATES:
        if candidate.is_file():
            return candidate

    expected_paths = "\n".join(f"- {path}" for path in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Could not find ai4i2020.csv. Place it at one of these paths:\n"
        f"{expected_paths}"
    )


def main() -> int:
    """Load the CSV and display its dimensions, target counts, and sample."""
    try:
        dataset_path = find_dataset()
        data = load_ai4i_dataset(dataset_path)
    except (FileNotFoundError, OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"Dataset inspection failed: {exc}", file=sys.stderr)
        return 1

    if data.empty:
        print(f"Dataset is empty: {dataset_path}", file=sys.stderr)
        return 1

    if TARGET_COLUMN not in data.columns:
        print(
            f"Expected target column {TARGET_COLUMN!r} was not found in {dataset_path}.",
            file=sys.stderr,
        )
        return 1

    print(f"Dataset: {dataset_path.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print("Column names:")
    for column_name in data.columns:
        print(f"  - {column_name}")

    print(f"\n{TARGET_COLUMN} counts:")
    target_counts = data[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    for label, count in target_counts.items():
        print(f"  {label}: {count:,}")

    print("\nFirst five records:")
    print(data.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
