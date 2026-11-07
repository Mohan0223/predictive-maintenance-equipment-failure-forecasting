"""Reusable prediction functions for single records and batch CSVs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from predictive_maintenance.data.preprocessing import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    prepare_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
REQUIRED_INPUT_COLUMNS = CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES
OUTCOME_COLUMNS = {"Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"}
PREDICTION_COLUMNS = (
    "predicted_failure",
    "failure_probability",
    "risk_level",
    "recommended_action",
    "prediction_generated_at_utc",
)


def load_model_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load and validate the selected model bundle from disk."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run scripts/package_model.py first."
        )

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise ValueError(f"Invalid model bundle format: {path}")
    if not hasattr(bundle["model"], "predict_proba"):
        raise ValueError("The saved model does not support probability predictions.")
    return bundle


def risk_level(probability: float, metadata: dict[str, Any]) -> str:
    """Map model probability to provisional Low, Medium, or High risk."""
    thresholds = metadata.get("risk_thresholds", {})
    low_below = float(thresholds.get("low_below", 0.25))
    high_from = float(thresholds.get("high_from", 0.50))
    if not 0.0 <= low_below < high_from <= 1.0:
        raise ValueError("Risk thresholds must satisfy 0 <= low < high <= 1.")

    if probability < low_below:
        return "Low"
    if probability < high_from:
        return "Medium"
    return "High"


def recommended_action(risk: str) -> str:
    """Return a cautious maintenance recommendation for a risk band."""
    recommendations = {
        "Low": "Continue routine monitoring and follow the planned maintenance schedule.",
        "Medium": (
            "Review recent sensor trends and schedule an inspection at the next "
            "planned maintenance window."
        ),
        "High": (
            "Arrange a prompt inspection and verify the sensor readings with an "
            "operator before taking equipment offline."
        ),
    }
    try:
        return recommendations[risk]
    except KeyError as exc:
        raise ValueError(f"Unknown risk level: {risk}") from exc


def _validate_readings(readings: pd.DataFrame) -> pd.DataFrame:
    """Validate raw inference fields and reject target-label columns."""
    if not isinstance(readings, pd.DataFrame):
        raise TypeError("Readings must be provided as a pandas DataFrame.")
    if readings.empty:
        raise ValueError("At least one sensor record is required.")

    leaked_columns = sorted(OUTCOME_COLUMNS.intersection(readings.columns))
    if leaked_columns:
        raise ValueError(
            "Remove outcome/target columns before inference: "
            f"{leaked_columns}"
        )

    missing_columns = sorted(set(REQUIRED_INPUT_COLUMNS).difference(readings.columns))
    if missing_columns:
        raise ValueError(f"Required input columns are missing: {missing_columns}")

    validated = readings.copy()
    for column in BASE_NUMERIC_FEATURES:
        original = validated[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            raise ValueError(
                f"Column {column!r} contains {int(invalid.sum())} non-numeric values."
            )
        validated[column] = converted
    return validated


def predict_dataframe(
    readings: pd.DataFrame,
    bundle: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return predictions, probabilities, risk levels, and actions per row."""
    validated = _validate_readings(readings)
    active_bundle = bundle if bundle is not None else load_model_bundle()
    model = active_bundle["model"]
    metadata = active_bundle["metadata"]
    threshold = float(metadata.get("decision_threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The model decision threshold must be between 0 and 1.")

    features = prepare_features(validated)
    probabilities = model.predict_proba(features)[:, 1]
    predicted_labels = (probabilities >= threshold).astype("int8")
    generated_at = datetime.now(timezone.utc).isoformat()

    result = validated.copy()
    result["predicted_failure"] = predicted_labels
    result["failure_probability"] = probabilities
    result["risk_level"] = [risk_level(float(value), metadata) for value in probabilities]
    result["recommended_action"] = [
        recommended_action(level) for level in result["risk_level"]
    ]
    result["prediction_generated_at_utc"] = generated_at
    return result


def predict_one(
    reading: dict[str, Any],
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict failure risk for one sensor record."""
    result = predict_dataframe(pd.DataFrame([reading]), bundle=bundle)
    return result.iloc[0].to_dict()
