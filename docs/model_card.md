# Model card

## Intended use

Educational demonstration of binary failure classification and a maintenance dashboard. The model estimates whether an AI4I record is labeled as a failure at the recorded operating point. It is not a certified safety system, does not trigger automatic shutdown, and does not forecast failure within a specified future time window.

## Data

- Source: UCI AI4I 2020 Predictive Maintenance Dataset, CC BY 4.0.
- Nature: synthetic data reflecting industrial operating patterns, not actual factory telemetry.
- Size: 10,000 records; 339 positive failure labels.
- Predictors: product type, air temperature, process temperature, rotational speed, torque, and tool wear, plus two deterministic derived features.
- Excluded columns: identifiers and failure outcome/mode labels.
- Missing fields: vibration and pressure are absent and are not imputed or fabricated.

## Training and evaluation

- Split: first 80% of records ordered by UDI for training; final 20% held out.
- Tuning: 12 randomized parameter combinations per model; four expanding `TimeSeriesSplit` folds.
- Primary search metric: F1 at a fixed 0.50 threshold; recall and precision are reported as tie-breakers and context.
- Preprocessing: median numeric imputation, most-frequent categorical imputation, one-hot encoding of `Type`, all fitted within each training fold.
- Selected baseline: Random Forest by mean training-CV F1.

| Model | Test precision | Test recall | Test F1 | Test ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7222 | 0.6667 | 0.6933 | 0.9549 | 0.7419 |
| XGBoost | 0.6667 | 0.7179 | 0.6914 | 0.9550 | 0.7358 |

The holdout contains 39 positive cases. Metrics therefore have notable sampling uncertainty. The requested precision/recall/F1 goals were not met.

## Risks and limitations

- Synthetic-to-real domain shift may be large.
- No vibration or pressure inputs are available.
- No time-to-failure or lead-time target exists.
- Probabilities have not been calibrated. Low/Medium/High bands are provisional UI categories, not validated operational thresholds.
- Class imbalance makes accuracy misleading; use precision, recall, F1, ROC-AUC, average precision, and confusion matrices together.
- Human inspection should determine whether a failure actually occurred.

## Monitoring before real use

Validate on representative machine-specific data, define a forecast horizon, calibrate probabilities, choose thresholds with maintenance stakeholders, monitor data drift and false negatives, and preserve inspection-confirmed labels separately from predictions.
