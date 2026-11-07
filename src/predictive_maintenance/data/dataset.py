"""Loading and schema validation for the staged AI4I source dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from predictive_maintenance.data.preprocessing import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
REQUIRED_DATASET_COLUMNS = {
    "UDI",
    *CATEGORICAL_FEATURES,
    *BASE_NUMERIC_FEATURES,
    TARGET_COLUMN,
}


def load_ai4i_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Read the source CSV and validate columns used by training and EDA."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run `python src/download_data.py`."
        )

    try:
        data = pd.read_csv(dataset_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read dataset {dataset_path}: {exc}") from exc

    if data.empty:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    missing_columns = sorted(REQUIRED_DATASET_COLUMNS.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return data
