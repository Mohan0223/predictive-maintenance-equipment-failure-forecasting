"""Reusable feature engineering and preprocessing for the AI4I dataset."""

from __future__ import annotations

from math import tau

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "Machine failure"
CATEGORICAL_FEATURES = ("Type",)
BASE_NUMERIC_FEATURES = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
ENGINEERED_NUMERIC_FEATURES = (
    "Temperature difference [K]",
    "Mechanical power [W]",
)
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Select permitted predictors and add deterministic derived features.

    Identifier columns, the target, and failure-mode outcome columns are
    intentionally excluded to avoid target leakage.
    """
    required = set(CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Required feature columns are missing: {missing}")

    features = data.loc[:, CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES].copy()
    for column in BASE_NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    features["Temperature difference [K]"] = (
        features["Process temperature [K]"] - features["Air temperature [K]"]
    )
    angular_speed_rad_per_second = features["Rotational speed [rpm]"] * (tau / 60.0)
    features["Mechanical power [W]"] = (
        features["Torque [Nm]"] * angular_speed_rad_per_second
    )
    return features.loc[:, MODEL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Build an imputation/encoding pipeline to fit on training data only."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )
