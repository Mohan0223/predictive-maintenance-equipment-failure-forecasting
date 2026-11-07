"""Add clearly labeled synthetic dashboard examples for local screenshots."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from predictive_maintenance.data.dataset import load_ai4i_dataset
from predictive_maintenance.data.preprocessing import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from predictive_maintenance.database.repository import MaintenanceRepository
from predictive_maintenance.inference.predictor import (
    load_model_bundle,
    predict_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
RISK_BANDS = ("Low", "Medium", "High")
SAMPLES_PER_BAND = 3


def main() -> int:
    """Seed three demonstration assets without creating confirmed outcomes."""
    if not RAW_DATA_PATH.is_file() or not TEST_PREDICTIONS_PATH.is_file():
        print(
            "Run scripts/prepare_data.py and scripts/train_models.py first.",
            file=sys.stderr,
        )
        return 1

    try:
        bundle = load_model_bundle()
        repository = MaintenanceRepository()
        repository.initialize()
        existing_codes = {
            item["equipment_code"] for item in repository.list_equipment()
        }
        demo_codes = {f"DEMO-{risk.upper()}-01" for risk in RISK_BANDS}
        if demo_codes.intersection(existing_codes):
            print("Demo records already exist; no additional rows were added.")
            return 0

        test_udis = set(pd.read_csv(TEST_PREDICTIONS_PATH)["UDI"].tolist())
        raw_data = load_ai4i_dataset(RAW_DATA_PATH)
        demo_inputs = raw_data.loc[
            raw_data["UDI"].isin(test_udis),
            list(CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES),
        ].copy()
        predictions = predict_dataframe(demo_inputs, bundle=bundle)
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        print(f"Could not prepare synthetic demo records: {exc}", file=sys.stderr)
        return 1

    timestamp_now = datetime.now(timezone.utc)
    created_count = 0
    for risk in RISK_BANDS:
        bucket = predictions[predictions["risk_level"] == risk].sort_values(
            "failure_probability"
        )
        if len(bucket) < SAMPLES_PER_BAND:
            print(
                f"Need at least {SAMPLES_PER_BAND} predictions in the {risk} band; "
                f"found {len(bucket)}.",
                file=sys.stderr,
            )
            return 1

        positions = [
            round(index * (len(bucket) - 1) / (SAMPLES_PER_BAND - 1))
            for index in range(SAMPLES_PER_BAND)
        ]
        selected_rows = bucket.iloc[positions]
        equipment_code = f"DEMO-{risk.upper()}-01"

        for index, (_, row) in enumerate(selected_rows.iterrows()):
            sensor_reading = {
                column: row[column]
                for column in CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES
            }
            prediction = row.to_dict()
            prediction["decision_threshold"] = bundle["metadata"][
                "decision_threshold"
            ]
            recorded_at = (
                timestamp_now - timedelta(hours=36 - 12 * index)
            ).isoformat(timespec="seconds")
            repository.record_prediction(
                equipment_code,
                sensor_reading,
                prediction,
                model_name=bundle["metadata"]["model_name"],
                model_version=str(bundle["metadata"]["artifact_version"]),
                data_source="synthetic_demo",
                equipment_name=f"Synthetic {risk.lower()}-risk example",
                equipment_type="Synthetic dashboard demo",
                location="Local demo environment",
                recorded_at=recorded_at,
            )
            created_count += 1

    print(f"Added {created_count} synthetic demonstration readings.")
    print("No inspection-confirmed failure outcomes were created.")
    print("Remove the local SQLite file to clear these disposable demo records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
