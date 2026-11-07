"""Package the CV-selected tuned pipeline for reusable inference."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from predictive_maintenance.data.preprocessing import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
MODEL_DIRECTORY = PROJECT_ROOT / "models"
CANDIDATE_PATHS = {
    "Random Forest": MODEL_DIRECTORY / "random_forest_tuned.joblib",
    "XGBoost": MODEL_DIRECTORY / "xgboost_tuned.joblib",
}
SELECTED_MODEL_PATH = MODEL_DIRECTORY / "model.joblib"
METADATA_PATH = MODEL_DIRECTORY / "model_metadata.json"


def main() -> int:
    """Load the selected candidate and save model plus inference metadata."""
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        selected_name = metrics["selected_model"]
        candidate_path = CANDIDATE_PATHS[selected_name]
        if not candidate_path.is_file():
            raise FileNotFoundError(f"Selected candidate not found: {candidate_path}")

        selected_metrics: dict[str, Any] = metrics["results"][selected_name]
        model = joblib.load(candidate_path)
        metadata = {
            "model_name": selected_name,
            "artifact_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": "AI4I 2020 Predictive Maintenance Dataset (synthetic)",
            "selection_rule": (
                "Highest mean 4-fold TimeSeriesSplit F1; recall and precision "
                "used as tie-breakers."
            ),
            "decision_threshold": float(selected_metrics["decision_threshold"]),
            "risk_thresholds": {"low_below": 0.25, "high_from": 0.50},
            "input_columns": list(CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES),
            "cross_validation_metrics": selected_metrics["cross_validation"],
            "held_out_test_metrics": selected_metrics["test"],
            "candidate_artifact": candidate_path.name,
        }
        bundle = {"model": model, "metadata": metadata}

        MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, SELECTED_MODEL_PATH)
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
        print(f"Could not package selected model: {exc}", file=sys.stderr)
        return 1

    print(f"Selected model: {selected_name}")
    print(f"Saved: {SELECTED_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
