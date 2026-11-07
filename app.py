"""Streamlit operations dashboard for the predictive-maintenance project."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
DASHBOARD_PAGES = [
    "Home Dashboard",
    "Equipment Monitoring",
    "Failure Prediction",
    "Model Performance",
    "Maintenance Alerts",
    "Historical Data and Analytics",
]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from predictive_maintenance.database.repository import MaintenanceRepository
from predictive_maintenance.inference.predictor import (
    load_model_bundle,
    predict_dataframe,
    predict_one,
)


st.set_page_config(
    page_title="Predictive Maintenance Operations",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_repository() -> MaintenanceRepository:
    repository = MaintenanceRepository()
    repository.initialize()
    return repository


@st.cache_resource
def get_model_bundle() -> dict[str, Any]:
    return load_model_bundle()


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0b1220; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111b2d 0%, #0b1220 100%);
            border-right: 1px solid #22314a;
        }
        [data-testid="stMetric"] {
            background: #111b2d;
            border: 1px solid #26364f;
            border-radius: 14px;
            padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] { color: #9eacc2; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #26364f;
            border-radius: 14px;
        }
        .status-note {
            border-left: 4px solid #20c997;
            background: #111b2d;
            padding: 0.8rem 1rem;
            border-radius: 8px;
            color: #d5deeb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(figure: go.Figure, title: str | None = None) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=55, b=15),
        title=title or figure.layout.title.text,
        legend_title_text="",
    )
    return figure


def render_home(repository: MaintenanceRepository, bundle: dict[str, Any]) -> None:
    st.title("Operations overview")
    st.caption("Equipment status from the latest saved model predictions.")
    st.markdown(
        '<div class="status-note">Predictions are model estimates, not confirmed '
        "equipment failures. The current AI4I model uses temperature, rotational "
        "speed, torque, tool wear, and product type; vibration and pressure are "
        "not part of this dataset or model.</div>",
        unsafe_allow_html=True,
    )

    equipment = repository.latest_equipment_status()
    alerts = repository.list_alerts(status="Open")
    confirmations = repository.confirmation_counts()
    predicted = [row for row in equipment if row.get("risk_level")]
    low_risk = sum(row.get("risk_level") == "Low" for row in predicted)
    at_risk = sum(row.get("risk_level") in {"Medium", "High"} for row in predicted)

    metric_columns = st.columns(5)
    metric_columns[0].metric("Monitored equipment", len(equipment))
    metric_columns[1].metric("Latest prediction: low risk", low_risk)
    metric_columns[2].metric("Latest prediction: at risk", at_risk)
    metric_columns[3].metric("Open maintenance alerts", len(alerts))
    metric_columns[4].metric("Confirmed failures logged", confirmations["failures"])

    demo_rows = [
        row for row in equipment if row.get("data_source") == "synthetic_demo"
    ]
    if demo_rows:
        st.info(
            "This database contains synthetic demonstration readings. "
            "They are not live or confirmed machine data."
        )

    if not equipment:
        st.info(
            "No equipment has been recorded yet. Add an asset in Equipment "
            "Monitoring, then enter a reading in Failure Prediction."
        )
        return

    status_rows = [row for row in equipment if row.get("risk_level")]
    if not status_rows:
        st.info("Equipment is registered. No predictions have been saved yet.")
        return

    left, right = st.columns((1, 1.4))
    with left:
        risk_counts = (
            pd.DataFrame(status_rows)["risk_level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .rename_axis("Risk level")
            .reset_index(name="Equipment")
        )
        figure = px.pie(
            risk_counts,
            names="Risk level",
            values="Equipment",
            hole=0.62,
            color="Risk level",
            color_discrete_map={"Low": "#20c997", "Medium": "#f0ad4e", "High": "#ef5350"},
            title="Latest predicted risk",
        )
        st.plotly_chart(plotly_layout(figure), width="stretch")

    with right:
        chart_data = pd.DataFrame(status_rows).sort_values(
            "failure_probability", ascending=True
        )
        figure = px.bar(
            chart_data,
            x="failure_probability",
            y="equipment_code",
            color="risk_level",
            orientation="h",
            color_discrete_map={"Low": "#20c997", "Medium": "#f0ad4e", "High": "#ef5350"},
            labels={
                "failure_probability": "Predicted failure probability",
                "equipment_code": "Equipment",
                "risk_level": "Predicted risk",
            },
            title="Latest model estimates by equipment",
        )
        figure.update_xaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(plotly_layout(figure), width="stretch")

    table_columns = [
        "equipment_code",
        "equipment_name",
        "recorded_at",
        "risk_level",
        "predicted_failure",
        "failure_probability",
        "data_source",
    ]
    status_table = pd.DataFrame(status_rows)
    st.subheader("Latest prediction by equipment")
    st.dataframe(
        status_table.loc[:, [column for column in table_columns if column in status_table]],
        width="stretch",
        hide_index=True,
    )
    st.caption(f"Selected model: {bundle['metadata']['model_name']}")


def render_equipment_monitoring(repository: MaintenanceRepository) -> None:
    st.title("Equipment monitoring")
    st.caption("Register assets, inspect their latest readings, and log confirmed outcomes.")

    with st.expander("Register equipment", expanded=False):
        with st.form("register_equipment_form", clear_on_submit=True):
            col_a, col_b = st.columns(2)
            code = col_a.text_input("Equipment code *", placeholder="PUMP-001")
            name = col_b.text_input("Equipment name *", placeholder="Cooling pump 1")
            col_c, col_d = st.columns(2)
            equipment_type = col_c.text_input("Equipment type", placeholder="Pump")
            location = col_d.text_input("Location", placeholder="Line 2")
            submitted = st.form_submit_button("Add equipment", type="primary")
        if submitted:
            if not code.strip() or not name.strip():
                st.error("Equipment code and name are required.")
            else:
                try:
                    repository.add_equipment(code, name, equipment_type, location)
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
                else:
                    st.success(f"Added equipment {code.strip()}.")
                    st.rerun()

    equipment = repository.list_equipment()
    if not equipment:
        st.info("Register an equipment item to begin monitoring.")
        return

    code_to_name = {
        item["equipment_code"]: item["equipment_name"] for item in equipment
    }
    selected_code = st.selectbox(
        "Select equipment",
        list(code_to_name),
        format_func=lambda code: f"{code} — {code_to_name[code]}",
    )
    statuses = repository.latest_equipment_status()
    selected_status = next(
        (row for row in statuses if row["equipment_code"] == selected_code),
        None,
    )

    if selected_status and selected_status.get("prediction_id"):
        columns = st.columns(4)
        columns[0].metric("Latest predicted risk", selected_status["risk_level"])
        columns[1].metric(
            "Predicted failure probability",
            f"{selected_status['failure_probability']:.1%}",
        )
        columns[2].metric(
            "Predicted failure",
            "Yes" if selected_status["predicted_failure"] else "No",
        )
        columns[3].metric("Reading time", selected_status.get("recorded_at") or "—")
        st.info(
            "The status above is a model prediction. It does not confirm that "
            "the equipment has failed or is healthy."
        )

    history = repository.equipment_history(selected_code)
    if history.empty:
        st.info("No readings have been saved for this equipment.")
        return

    history = history.sort_values("recorded_at")
    sensor_labels = {
        "air_temperature_k": "Air temperature (K)",
        "process_temperature_k": "Process temperature (K)",
        "rotational_speed_rpm": "Rotational speed (rpm)",
        "torque_nm": "Torque (Nm)",
        "tool_wear_min": "Tool wear (min)",
        "vibration_value": "Vibration (if supplied)",
        "pressure_value": "Pressure (if supplied)",
    }
    available_sensors = [
        column
        for column in sensor_labels
        if column in history.columns and history[column].notna().any()
    ]
    if available_sensors:
        selected_sensor = st.selectbox(
            "Sensor trend",
            available_sensors,
            format_func=lambda column: sensor_labels[column],
            key="equipment_sensor_trend",
        )
        figure = px.line(
            history,
            x="recorded_at",
            y=selected_sensor,
            markers=True,
            title=f"{sensor_labels[selected_sensor]} history",
            labels={selected_sensor: sensor_labels[selected_sensor], "recorded_at": "Recorded at"},
        )
        st.plotly_chart(plotly_layout(figure), width="stretch")

    st.subheader("Historical readings and model estimates")
    st.dataframe(history, width="stretch", hide_index=True)

    with st.expander("Log an inspection-confirmed outcome"):
        reading_ids: list[int | None] = [None] + [
            int(value) for value in history["reading_id"].tolist()
        ]
        with st.form("confirm_outcome_form", clear_on_submit=True):
            linked_reading = st.selectbox(
                "Link to a reading (optional)",
                reading_ids,
                format_func=lambda value: "No linked reading" if value is None else str(value),
            )
            outcome = st.radio(
                "Inspection outcome",
                ["No failure found", "Failure confirmed"],
                horizontal=True,
            )
            confirmed_by = st.text_input("Confirmed by *")
            notes = st.text_area("Inspection notes")
            submitted = st.form_submit_button("Save confirmed outcome")
        if submitted:
            try:
                repository.confirm_failure(
                    selected_code,
                    confirmed_failure=(outcome == "Failure confirmed"),
                    confirmed_by=confirmed_by,
                    reading_id=linked_reading,
                    notes=notes,
                )
            except (ValueError, OSError) as exc:
                st.error(str(exc))
            else:
                st.success("Inspection outcome saved separately from predictions.")
                st.rerun()

    confirmations = repository.list_confirmations(selected_code)
    if confirmations:
        st.subheader("Inspection-confirmed outcomes")
        st.dataframe(pd.DataFrame(confirmations), width="stretch", hide_index=True)


def render_prediction(repository: MaintenanceRepository, bundle: dict[str, Any]) -> None:
    st.title("Failure prediction")
    st.caption("Enter one machine reading or upload a CSV to score several readings at once.")
    st.info(
        "The model scores product type, temperature, speed, torque, and tool wear. "
        "Vibration and pressure can be saved with the reading, but this AI4I model "
        "cannot use them because they were not included in its training data. "
        "Every result is an estimate and needs inspection for confirmation."
    )
    single_tab, batch_tab = st.tabs(["Single reading", "Batch CSV"])

    with single_tab:
        st.subheader("Enter machine details")
        st.caption(
            "Replace the example defaults with current sensor readings. Fields marked * are required; "
            "the reading and prediction are saved to local history."
        )
        with st.form("single_prediction_form"):
            st.markdown("**Equipment**")
            col_a, col_b = st.columns(2)
            equipment_code = col_a.text_input(
                "Equipment code *",
                value="PUMP-001",
                help="A short ID used to find this machine in history.",
            )
            equipment_name = col_b.text_input(
                "Equipment name",
                value="Monitored equipment",
                help="Optional friendly name shown in the dashboard.",
            )

            st.markdown("**Readings used by the model**")
            equipment_type = st.selectbox(
                "Machine product type",
                ["L", "M", "H"],
                index=1,
                format_func=lambda value: {
                    "L": "Light (L)",
                    "M": "Medium (M)",
                    "H": "Heavy (H)",
                }[value],
                help="Select the product type recorded for this machine in the AI4I dataset.",
            )
            col_temp_a, col_temp_b = st.columns(2)
            air_temperature = col_temp_a.number_input(
                "Air temperature (K)", min_value=250.0, max_value=330.0,
                value=301.2, step=0.1,
                help="Ambient air temperature around the machine, in kelvin.",
            )
            process_temperature = col_temp_b.number_input(
                "Process temperature (K)", min_value=250.0, max_value=350.0,
                value=311.5, step=0.1,
                help="Operating process temperature, in kelvin.",
            )
            col_speed, col_torque, col_wear = st.columns(3)
            rotational_speed = col_speed.number_input(
                "Rotational speed (rpm)", min_value=0, max_value=10000,
                value=1400, step=10, help="Shaft rotation speed, in revolutions per minute."
            )
            torque = col_torque.number_input(
                "Torque (Nm)", min_value=0.0, max_value=500.0,
                value=50.0, step=0.5, help="Rotational force, in newton-metres."
            )
            tool_wear = col_wear.number_input(
                "Tool wear (min)", min_value=0, max_value=1000,
                value=150, step=1, help="Accumulated tool operating time, in minutes."
            )

            with st.expander("Add optional sensor readings"):
                st.caption("These values are stored for your records. They do not affect this model's score.")
                col_vibration, col_pressure = st.columns(2)
                vibration = col_vibration.number_input(
                    "Vibration (optional; logged only)",
                    value=None,
                    step=0.1,
                    placeholder="Not measured",
                    help="Enter the value in the unit used by your sensor. It is recorded, not scored.",
                )
                pressure = col_pressure.number_input(
                    "Pressure (optional; logged only)",
                    value=None,
                    step=0.1,
                    placeholder="Not measured",
                    help="Enter the value in the unit used by your sensor. It is recorded, not scored.",
                )
            submitted = st.form_submit_button("Predict failure risk", type="primary", width="stretch")

        if submitted:
            st.session_state.pop("last_single_prediction", None)
            if not equipment_code.strip():
                st.error("Equipment code is required.")
            else:
                reading = {
                    "Type": equipment_type,
                    "Air temperature [K]": air_temperature,
                    "Process temperature [K]": process_temperature,
                    "Rotational speed [rpm]": rotational_speed,
                    "Torque [Nm]": torque,
                    "Tool wear [min]": tool_wear,
                    "vibration": vibration,
                    "pressure": pressure,
                }
                try:
                    result = predict_one(reading, bundle=bundle)
                    result["decision_threshold"] = bundle["metadata"]["decision_threshold"]
                    saved = repository.record_prediction(
                        equipment_code.strip(),
                        reading,
                        result,
                        model_name=bundle["metadata"]["model_name"],
                        model_version=str(bundle["metadata"]["artifact_version"]),
                        data_source="operator_input",
                        equipment_name=equipment_name.strip() or equipment_code.strip(),
                        equipment_type="AI4I benchmark machine",
                    )
                except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
                    st.error(f"Prediction could not be saved: {exc}")
                else:
                    st.session_state["last_single_prediction"] = {
                        "equipment_code": equipment_code.strip(),
                        "reading_id": saved["reading_id"],
                        "result": result,
                    }

        latest = st.session_state.get("last_single_prediction")
        if latest:
            result = latest["result"]
            st.divider()
            st.subheader(f"Latest result: {latest['equipment_code']}")
            st.success(
                f"Reading #{latest['reading_id']} and its prediction are saved in equipment history."
            )
            result_columns = st.columns(3)
            result_columns[0].metric("Failure probability", f"{result['failure_probability']:.1%}")
            result_columns[1].metric("Risk level", result["risk_level"])
            result_columns[2].metric(
                "Model decision",
                "Failure predicted" if result["predicted_failure"] else "No failure predicted",
            )
            st.progress(
                float(result["failure_probability"]),
                text="Estimated probability for this reading",
            )
            if result["risk_level"] == "High":
                st.error(
                    "High model risk. Arrange a prompt inspection and verify the sensor readings."
                )
            elif result["risk_level"] == "Medium":
                st.warning("Medium model risk. Review recent trends and plan an inspection.")
            else:
                st.info("Low model risk. Continue routine monitoring and planned maintenance.")
            st.write("**Suggested next step:** " + result["recommended_action"])
            st.caption(
                "This is a model estimate, not a confirmed failure. Record an inspection outcome "
                "separately on the Equipment Monitoring page."
            )

    with batch_tab:
        st.subheader("Score readings from a CSV")
        st.write(
            "Use the template below or prepare a CSV with one row per reading. "
            "Every row needs product type, both temperatures, rotational speed, torque, and tool wear."
        )
        template = pd.DataFrame(
            [
                {
                    "equipment_code": "PUMP-001",
                    "Type": "M",
                    "Air temperature [K]": 301.2,
                    "Process temperature [K]": 311.5,
                    "Rotational speed [rpm]": 1400,
                    "Torque [Nm]": 50.0,
                    "Tool wear [min]": 150,
                    "vibration": "",
                    "pressure": "",
                }
            ]
        )
        st.download_button(
            "Download CSV template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="predictive_maintenance_input_template.csv",
            mime="text/csv",
            help="Includes one editable example row and all supported input columns.",
        )
        st.caption(
            "Required columns: `Type`, `Air temperature [K]`, `Process temperature [K]`, "
            "`Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`. Optional columns: "
            "`equipment_code`, `vibration`, and `pressure`. Do not include failure labels."
        )
        uploaded = st.file_uploader("Choose a sensor CSV", type=["csv"], key="batch_csv")
        if uploaded is not None:
            try:
                batch_input = pd.read_csv(uploaded)
                batch_result = predict_dataframe(batch_input, bundle=bundle)
            except (ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                st.error(f"Could not process this CSV: {exc}")
            else:
                st.success(f"Generated predictions for {len(batch_result):,} readings.")
                st.dataframe(batch_result, width="stretch", hide_index=True)
                csv_bytes = batch_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions as CSV",
                    data=csv_bytes,
                    file_name="predictive_maintenance_predictions.csv",
                    mime="text/csv",
                )
                has_equipment_codes = (
                    "equipment_code" in batch_result.columns
                    and batch_result["equipment_code"].notna().all()
                    and batch_result["equipment_code"].astype(str).str.strip().ne("").all()
                )
                if "equipment_code" in batch_result.columns and not has_equipment_codes:
                    st.warning(
                        "Fill in an equipment_code for every row before saving these results to history."
                    )
                if has_equipment_codes and st.button(
                    "Save predictions to equipment history",
                    type="secondary",
                ):
                    try:
                        for row in batch_result.to_dict(orient="records"):
                            result = dict(row)
                            result["decision_threshold"] = bundle["metadata"][
                                "decision_threshold"
                            ]
                            repository.record_prediction(
                                str(row["equipment_code"]),
                                row,
                                result,
                                model_name=bundle["metadata"]["model_name"],
                                model_version=str(bundle["metadata"]["artifact_version"]),
                                data_source="batch_upload",
                            )
                    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
                        st.error(f"Could not save the batch to SQLite: {exc}")
                    else:
                        st.success("Batch predictions saved to equipment history.")
                elif "equipment_code" not in batch_result.columns:
                    st.caption(
                        "Add an `equipment_code` column if you also want to save predictions in equipment history."
                    )


def render_model_performance(bundle: dict[str, Any]) -> None:
    st.title("Model performance")
    metrics_path = PROJECT_ROOT / "reports" / "metrics.json"
    if not metrics_path.is_file():
        st.error("Model metrics are missing. Run scripts/train_models.py first.")
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    results = metrics["results"]
    selected = metrics["selected_model"]
    st.info(
        f"Selected model: {selected}. The requested goals of 90% precision, "
        "85% recall, and 87% F1 were not reached on the held-out AI4I data. "
        "The model is trained on a synthetic benchmark, not factory telemetry."
    )

    selected_test = results[selected]["test"]
    metric_columns = st.columns(5)
    metric_columns[0].metric("Test accuracy", f"{selected_test['accuracy']:.1%}")
    metric_columns[1].metric("Test precision", f"{selected_test['precision']:.1%}")
    metric_columns[2].metric("Test recall", f"{selected_test['recall']:.1%}")
    metric_columns[3].metric("Test F1", f"{selected_test['f1']:.1%}")
    metric_columns[4].metric("Test ROC-AUC", f"{selected_test['roc_auc']:.3f}")

    names = list(results)
    comparison_rows = []
    for name in names:
        for split_name in ("cross_validation", "test"):
            row = {"Model": name, "Evaluation": split_name.replace("_", " ").title()}
            row.update(results[name][split_name])
            comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    metrics_to_plot = ["precision", "recall", "f1", "roc_auc"]
    plot_data = comparison.melt(
        id_vars=["Model", "Evaluation"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score",
    )
    left, right = st.columns(2)
    with left:
        figure = px.bar(
            plot_data[plot_data["Evaluation"] == "Test"],
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="Held-out test metrics",
            range_y=[0, 1],
        )
        st.plotly_chart(plotly_layout(figure), width="stretch")
    with right:
        matrix = selected_test["confusion_matrix"]
        figure = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted no failure", "Predicted failure"],
                y=["Actual no failure", "Actual failure"],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        figure.update_yaxes(autorange="reversed")
        st.plotly_chart(
            plotly_layout(figure, title=f"{selected} confusion matrix"),
            width="stretch",
        )

    st.subheader("Cross-validation and held-out results")
    display_columns = [
        column
        for column in (
            "Model", "Evaluation", "accuracy", "precision", "recall", "f1",
            "roc_auc", "average_precision",
        )
        if column in comparison.columns
    ]
    st.dataframe(comparison.loc[:, display_columns], width="stretch", hide_index=True)
    st.caption(
        "The default decision threshold is 0.50. Accuracy is less informative "
        "when failures are rare; the holdout contains only 39 failure records."
    )

    selected_result = results[selected]
    st.subheader("Selected model details")
    st.json(selected_result["best_parameters"])
    st.caption(
        f"Inference artifact: {bundle['metadata']['model_name']} "
        f"(artifact version {bundle['metadata']['artifact_version']})."
    )
    with st.expander("What the metrics mean"):
        st.markdown(
            "- **Precision:** of the records flagged as failures, the share that "
            "were failures in the labelled holdout.\n"
            "- **Recall:** of the labelled failures, the share the model detected.\n"
            "- **F1:** the harmonic mean of precision and recall.\n"
            "- **ROC-AUC:** how well scores rank positive records above negative "
            "records across thresholds.\n"
            "- **Average precision:** a summary of precision across recall levels, "
            "useful when the positive class is uncommon."
        )


def render_alerts(repository: MaintenanceRepository) -> None:
    st.title("Maintenance alerts")
    st.caption("Alerts are created from Medium and High model risk predictions.")
    status_filter = st.selectbox(
        "Filter alerts",
        ["Open", "Acknowledged", "Resolved", "All"],
    )
    alerts = repository.list_alerts(
        status=None if status_filter == "All" else status_filter
    )
    if not alerts:
        st.info("No alerts match this filter.")
        return

    alert_frame = pd.DataFrame(alerts)
    st.dataframe(alert_frame, width="stretch", hide_index=True)
    alert_ids = [int(alert["alert_id"]) for alert in alerts]
    selected_alert = st.selectbox("Select alert ID", alert_ids)
    selected_record = next(
        alert for alert in alerts if int(alert["alert_id"]) == selected_alert
    )
    st.write(
        f"**{selected_record['equipment_code']} — {selected_record['severity']} risk**  "
        f"Predicted failure probability: {selected_record['failure_probability']:.1%}"
    )
    st.write(selected_record["recommendation"])
    st.caption("An alert is a model-generated prompt, not a confirmed equipment failure.")

    col_a, col_b = st.columns(2)
    if col_a.button("Acknowledge alert", disabled=selected_record["status"] != "Open"):
        repository.update_alert_status(selected_alert, "Acknowledged")
        st.success("Alert acknowledged.")
        st.rerun()
    if col_b.button(
        "Resolve alert",
        disabled=selected_record["status"] == "Resolved",
    ):
        repository.update_alert_status(selected_alert, "Resolved")
        st.success("Alert resolved.")
        st.rerun()


def render_history(repository: MaintenanceRepository) -> None:
    st.title("Historical data and analytics")
    equipment = repository.list_equipment()
    if not equipment:
        st.info("No equipment history is available yet.")
        return

    names = {item["equipment_code"]: item["equipment_name"] for item in equipment}
    selected_code = st.selectbox(
        "Equipment history",
        list(names),
        format_func=lambda code: f"{code} — {names[code]}",
    )
    history = repository.equipment_history(selected_code)
    if history.empty:
        st.info("No saved readings exist for this equipment.")
        return

    history["recorded_at"] = pd.to_datetime(history["recorded_at"], errors="coerce", utc=True)
    history = history.dropna(subset=["recorded_at"]).sort_values("recorded_at")
    sensor_labels = {
        "air_temperature_k": "Air temperature (K)",
        "process_temperature_k": "Process temperature (K)",
        "rotational_speed_rpm": "Rotational speed (rpm)",
        "torque_nm": "Torque (Nm)",
        "tool_wear_min": "Tool wear (min)",
        "vibration_value": "Vibration (if supplied)",
        "pressure_value": "Pressure (if supplied)",
    }
    available_sensors = [
        column
        for column in sensor_labels
        if column in history.columns and history[column].notna().any()
    ]
    if available_sensors:
        selected_sensor = st.selectbox(
            "Sensor metric",
            available_sensors,
            format_func=lambda column: sensor_labels[column],
        )
        figure = px.line(
            history,
            x="recorded_at",
            y=selected_sensor,
            markers=True,
            title=f"{sensor_labels[selected_sensor]} over time",
            labels={selected_sensor: sensor_labels[selected_sensor], "recorded_at": "Recorded at"},
        )
        st.plotly_chart(plotly_layout(figure), width="stretch")

    predicted_history = history.dropna(subset=["failure_probability"])
    if not predicted_history.empty:
        figure = px.line(
            predicted_history,
            x="recorded_at",
            y="failure_probability",
            markers=True,
            title="Predicted failure probability over time",
            labels={"failure_probability": "Predicted probability", "recorded_at": "Recorded at"},
        )
        figure.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(plotly_layout(figure), width="stretch")

    actual_confirmations = history.dropna(subset=["confirmed_failure"])
    if not actual_confirmations.empty:
        st.subheader("Inspection-confirmed outcomes")
        st.dataframe(
            actual_confirmations[
                ["reading_id", "recorded_at", "confirmed_failure", "confirmed_at", "confirmed_by"]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No inspection-confirmed outcomes are linked to these readings.")

    st.subheader("Historical records")
    st.dataframe(history, width="stretch", hide_index=True)
    csv_bytes = history.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download equipment history",
        data=csv_bytes,
        file_name=f"{selected_code.lower()}_history.csv",
        mime="text/csv",
    )


def main() -> None:
    apply_styles()
    try:
        repository = get_repository()
        bundle = get_model_bundle()
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        st.error(f"Application setup is incomplete: {exc}")
        st.info(
            "Initialize SQLite with `python scripts/init_database.py` and package "
            "the trained model with `python scripts/package_model.py`."
        )
        st.stop()

    with st.sidebar:
        st.markdown("## ⚙️ MaintAI")
        st.caption("Predictive maintenance operations")
        requested_page = st.query_params.get("page", DASHBOARD_PAGES[0])
        if requested_page not in DASHBOARD_PAGES:
            requested_page = DASHBOARD_PAGES[0]
        page = st.radio(
            "Navigation",
            DASHBOARD_PAGES,
            index=DASHBOARD_PAGES.index(requested_page),
            key="dashboard_navigation",
            label_visibility="collapsed",
        )
        st.query_params["page"] = page
        st.divider()
        st.caption(f"Active model: {bundle['metadata']['model_name']}")
        st.caption("Failure estimates are not inspection confirmations.")

    if page == "Home Dashboard":
        render_home(repository, bundle)
    elif page == "Equipment Monitoring":
        render_equipment_monitoring(repository)
    elif page == "Failure Prediction":
        render_prediction(repository, bundle)
    elif page == "Model Performance":
        render_model_performance(bundle)
    elif page == "Maintenance Alerts":
        render_alerts(repository)
    else:
        render_history(repository)


if __name__ == "__main__":
    main()
