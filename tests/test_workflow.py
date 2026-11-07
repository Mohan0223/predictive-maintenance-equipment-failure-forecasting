"""Focused unit and integration tests for the main project workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from predictive_maintenance.data.preprocessing import prepare_features
from predictive_maintenance.database.repository import MaintenanceRepository
from predictive_maintenance.inference.predictor import (
    load_model_bundle,
    predict_dataframe,
    predict_one,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_READING = {
    "Type": "L",
    "Air temperature [K]": 301.2,
    "Process temperature [K]": 311.5,
    "Rotational speed [rpm]": 1400,
    "Torque [Nm]": 50.0,
    "Tool wear [min]": 150,
}


@pytest.fixture(scope="session")
def model_bundle() -> dict:
    return load_model_bundle()


def test_feature_engineering_uses_only_permitted_inputs() -> None:
    source = pd.DataFrame(
        [
            {
                **SAMPLE_READING,
                "UDI": 123,
                "Product ID": "L12345",
                "Machine failure": 1,
                "TWF": 1,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            }
        ]
    )

    features = prepare_features(source)

    assert "Machine failure" not in features.columns
    assert "TWF" not in features.columns
    assert "UDI" not in features.columns
    assert features.loc[0, "Temperature difference [K]"] == pytest.approx(10.3)
    expected_power = 50.0 * 1400 * (2 * 3.141592653589793 / 60)
    assert features.loc[0, "Mechanical power [W]"] == pytest.approx(expected_power)


def test_single_and_batch_prediction_return_probabilities(model_bundle: dict) -> None:
    single_result = predict_one(SAMPLE_READING, bundle=model_bundle)
    batch_result = predict_dataframe(
        pd.DataFrame([SAMPLE_READING, SAMPLE_READING]),
        bundle=model_bundle,
    )
    optional_sensor_result = predict_one(
        {**SAMPLE_READING, "vibration": 0.7, "pressure": 101.2},
        bundle=model_bundle,
    )

    assert single_result["predicted_failure"] in (0, 1)
    assert 0.0 <= single_result["failure_probability"] <= 1.0
    assert single_result["risk_level"] in {"Low", "Medium", "High"}
    assert single_result["recommended_action"]
    assert len(batch_result) == 2
    assert batch_result["failure_probability"].between(0, 1).all()
    assert optional_sensor_result["failure_probability"] == pytest.approx(
        single_result["failure_probability"]
    )


def test_inference_rejects_failure_labels(model_bundle: dict) -> None:
    labelled_input = pd.DataFrame(
        [{**SAMPLE_READING, "Machine failure": 1}]
    )

    with pytest.raises(ValueError, match="outcome/target columns"):
        predict_dataframe(labelled_input, bundle=model_bundle)


def test_sqlite_repository_keeps_confirmations_separate(tmp_path: Path) -> None:
    repository = MaintenanceRepository(tmp_path / "workflow.db")
    repository.initialize()
    equipment_id = repository.add_equipment(
        "PUMP-TEST-01",
        "Test pump",
        equipment_type="Pump",
        location="Test bay",
    )
    assert equipment_id > 0
    with pytest.raises(ValueError, match="already exists"):
        repository.add_equipment("PUMP-TEST-01", "Duplicate pump")

    prediction = {
        "predicted_failure": 1,
        "failure_probability": 0.82,
        "decision_threshold": 0.50,
        "risk_level": "High",
        "recommended_action": "Arrange a prompt inspection.",
    }
    saved = repository.record_prediction(
        "PUMP-TEST-01",
        {**SAMPLE_READING, "vibration": 0.7, "pressure": 101.2},
        prediction,
        model_name="Random Forest",
        model_version="test-version",
    )

    assert saved["reading_id"]
    assert saved["prediction_id"]
    assert saved["alert_id"]
    assert repository.confirmation_counts() == {
        "total": 0,
        "failures": 0,
        "non_failures": 0,
    }
    status = repository.latest_equipment_status()[0]
    assert status["risk_level"] == "High"
    assert status["product_type"] == "L"
    assert repository.list_alerts("Open")[0]["equipment_code"] == "PUMP-TEST-01"

    confirmation_id = repository.confirm_failure(
        "PUMP-TEST-01",
        True,
        confirmed_by="Test operator",
        reading_id=saved["reading_id"],
        notes="Inspection confirmed a failure.",
    )
    assert confirmation_id > 0
    assert repository.confirmation_counts()["failures"] == 1
    history = repository.equipment_history("PUMP-TEST-01")
    assert int(history.loc[0, "confirmed_failure"]) == 1
    assert int(history.loc[0, "predicted_failure"]) == 1
    assert history.loc[0, "vibration_value"] == pytest.approx(0.7)
    assert history.loc[0, "pressure_value"] == pytest.approx(101.2)

    alert_id = int(saved["alert_id"])
    repository.update_alert_status(alert_id, "Acknowledged")
    assert repository.list_alerts("Acknowledged")[0]["alert_id"] == alert_id
    repository.update_alert_status(alert_id, "Resolved")
    assert repository.list_alerts("Resolved")[0]["alert_id"] == alert_id


def test_streamlit_pages_render_without_script_errors() -> None:
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"),
        default_timeout=30,
    ).run()
    assert not app.exception

    pages = [
        "Home Dashboard",
        "Equipment Monitoring",
        "Failure Prediction",
        "Model Performance",
        "Maintenance Alerts",
        "Historical Data and Analytics",
    ]
    for page in pages:
        # Keep the deep-link parameter and the navigation widget in sync.
        app.query_params["page"] = page
        navigation = next(widget for widget in app.radio if widget.label == "Navigation")
        navigation.set_value(page)
        app.run()
        assert not app.exception, f"Streamlit page failed: {page}"
        if page == "Failure Prediction":
            input_labels = {widget.label for widget in app.number_input}
            assert "Vibration (optional; logged only)" in input_labels
            assert "Pressure (optional; logged only)" in input_labels
