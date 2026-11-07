# Resume description and interview preparation

## Resume-ready project entry

**Predictive Maintenance & Equipment Failure Forecasting** — Python, Pandas, scikit-learn, XGBoost, SQLite, Streamlit, Plotly

- Built a modular predictive-maintenance application with feature engineering, leakage-safe time-ordered validation, and Random Forest/XGBoost comparison on the synthetic UCI AI4I benchmark.
- Packaged preprocessing with the selected model and built single-record and batch inference with failure probabilities, risk bands, and maintenance recommendations.
- Developed a six-page Streamlit dashboard backed by SQLite, separating model predictions and alerts from inspection-confirmed failure outcomes.
- Reported precision, recall, F1, ROC-AUC, average precision, and confusion matrices; documented that benchmark performance did not meet the initial target metrics.

## Interview questions and answers

### 1. Why did you use both Random Forest and XGBoost?

Both are tree-based classifiers that work well with nonlinear interactions and do not require feature scaling. Random Forest averages many independently sampled trees; XGBoost builds trees sequentially to reduce earlier errors. Comparing them gives a useful baseline-versus-boosting comparison.

### 2. Why is accuracy misleading here?

Only 339 of 10,000 records are labeled as failures. A model that predicts no failures can still achieve high accuracy. Precision, recall, F1, ROC-AUC, average precision, and the confusion matrix describe the minority class more clearly.

### 3. What does recall mean for maintenance?

Recall is the fraction of labeled failures the model detects. Higher recall can reduce missed failures, though it may create more false alarms. The right trade-off depends on inspection and downtime costs.

### 4. How did you prevent preprocessing leakage?

Imputation and one-hot encoding are inside a scikit-learn pipeline. Randomized search clones and fits that pipeline separately on each time-series training fold; the validation fold is only transformed.

### 5. Why use a time-ordered split?

AI4I is presented as multivariate time-series data. Training on earlier UDI rows and holding out later rows better approximates predicting later operating records than a random shuffle. UDI is used only for ordering, never as a predictor.

### 6. Why not scale the inputs?

Random Forest and XGBoost split on feature thresholds and are generally insensitive to differences in numeric units. Scaling would add work without benefiting these model families.

### 7. Why did you keep IQR outliers?

Some unusual sensor values may represent the operating conditions associated with failure. I reported outliers instead of deleting them automatically; tree models can handle different numeric ranges.

### 8. Why did you not use SMOTE before splitting?

Resampling before splitting can leak information and make the holdout unrealistic. The test set retains its natural distribution; class weights are tuned only during model fitting.

### 9. What are the current model’s limitations?

AI4I is synthetic, has no vibration or pressure readings, and has no explicit prediction horizon. Its scores do not establish real factory performance. The requested precision, recall, and F1 goals were not reached.

### 10. How are predictions distinguished from confirmed failures?

Predictions and alerts are stored in their own tables. An operator inspection is stored in a separate `failure_confirmations` table with confirmer and timestamp information.

### 11. What would you improve before real deployment?

Collect representative timestamped machine data, define a lead-time target, validate across machines and operating periods, calibrate probabilities, choose thresholds with maintenance stakeholders, and monitor false negatives and data drift.

### 12. What happens to SQLite data on Streamlit Community Cloud?

Local file storage is not guaranteed to persist there. SQLite is suitable for local work and a portfolio demo; durable deployed records need a persistent managed database.
