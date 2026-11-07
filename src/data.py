"""Convenience data API for notebooks and command-line use."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from predictive_maintenance.data.dataset import (
    DEFAULT_DATASET_PATH,
    REQUIRED_DATASET_COLUMNS,
    load_ai4i_dataset,
)
from predictive_maintenance.data.preprocessing import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    prepare_features,
)


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the AI4I CSV and validate the columns required by this project."""
    return load_ai4i_dataset(path)


__all__ = [
    "BASE_NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "DEFAULT_DATASET_PATH",
    "MODEL_FEATURES",
    "REQUIRED_DATASET_COLUMNS",
    "TARGET_COLUMN",
    "build_preprocessor",
    "load_dataset",
    "prepare_features",
]
