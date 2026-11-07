# Dashboard user guide

## Start locally

From the project root with the virtual environment active:

1. Run `python src/download_data.py` if `data/ai4i2020.csv` is missing.
2. If `models/model.joblib` is missing or you want to retrain, run `python scripts/prepare_data.py` followed by `python src/train.py`.
3. Run `python scripts/init_database.py` to create the local SQLite schema.
4. Run `python scripts/seed_demo_data.py` if you want sample dashboard records.
5. Run `streamlit run app.py`.

The app creates the SQLite database on startup if needed.

Seeding adds nine clearly labeled synthetic demonstration readings and no inspection-confirmed outcomes. These disposable records are tagged `synthetic_demo`; remove the local SQLite file to clear them.

## Pages

### Home Dashboard

Shows registered equipment, the latest predicted risk per asset, open alerts, and the number of inspection-confirmed failures logged. “Low risk” is a prediction label, not proof that a machine is healthy.

### Equipment Monitoring

Register equipment with a unique code. Select an asset to view its latest reading and history. Use **Log an inspection-confirmed outcome** only after an inspection; those records are stored separately from model estimates.

### Failure Prediction

Open **Failure Prediction** from the sidebar. For one reading, enter an equipment code, optionally give the asset a friendly name, then enter product type, air and process temperature in Kelvin, speed in rpm, torque in Nm, and tool wear in minutes. Press **Predict failure risk** to save the reading and show its probability, risk level, decision, and suggested next step. A first reading automatically registers the equipment code in SQLite; later readings with the same code are attached to that equipment's history.

The product type selector shows Light, Medium, and Heavy with the dataset codes in parentheses. Expand **Add optional sensor readings** to save vibration or pressure values for future reference. Those two values do not change the model score. The result is an estimate, not a confirmed failure; record an inspection outcome separately under **Equipment Monitoring**.

For batch scoring, upload a CSV containing these exact columns:

`Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.

Use **Download CSV template** on the page for a ready-to-edit example. Optional `vibration` and `pressure` columns are accepted and stored in history, but are not used to calculate the model prediction.

Add an optional `equipment_code` column to save batch predictions into equipment history. Outcome columns such as `Machine failure` and `TWF` must not be included in inference files.

### Model Performance

Shows the cross-validation and held-out metrics, chosen parameters, and the selected model’s confusion matrix. The positive class is rare, so do not judge the model by accuracy alone.

### Maintenance Alerts

Medium and High risk predictions create alerts. Acknowledge an alert when reviewed and resolve it when follow-up is complete. Alert status does not confirm an equipment failure.

### Historical Data and Analytics

Select equipment to chart recorded sensor values and predicted probability over time. Inspection-confirmed outcomes appear in a separate table. Export the available history to CSV.

## Model input coverage

The current AI4I model uses temperature, rotational speed, torque, tool wear, and product type. It has no vibration or pressure feature because the training dataset has neither field. Optional vibration and pressure readings are labeled as log-only in the form and do not affect model predictions.
