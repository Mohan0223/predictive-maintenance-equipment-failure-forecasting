# Model comparison

## Evaluation protocol

- Training rows: 8,000; held-out test rows: 2,000.
- Hyperparameter tuning used 4-fold `TimeSeriesSplit` inside the training partition.
- Each cross-validation fold fit its own imputer and encoder inside the model pipeline.
- Search selection metric: F1 at the default 0.50 threshold. Recall and precision are reported alongside it.
- The test set was evaluated once after tuning; no threshold was selected using test labels.
- Accuracy is included but can look high simply because failures are rare. Review precision, recall, F1, ROC-AUC, average precision, and the confusion matrices together.

## Metrics

| Model | CV precision | CV recall | CV F1 | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC | Test average precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8486 | 0.5821 | 0.6470 | 0.8966 | 0.9885 | 0.7222 | 0.6667 | 0.6933 | 0.9549 | 0.7419 |
| XGBoost | 0.7186 | 0.6143 | 0.6090 | 0.8963 | 0.9875 | 0.6667 | 0.7179 | 0.6914 | 0.9550 | 0.7358 |

Selected using training-only CV F1 (recall and precision as tie-breakers): **Random Forest**.

## Best parameters and confusion matrices

### Random Forest

- Best parameters: `{"classifier__class_weight": "balanced_subsample", "classifier__max_depth": 14, "classifier__max_features": 0.7, "classifier__min_samples_leaf": 4, "classifier__min_samples_split": 10, "classifier__n_estimators": 400}`
- Test confusion matrix (actual rows [0, 1], predicted columns [0, 1]): `[[1951, 10], [13, 26]]`
- Test threshold: `0.50`

### XGBoost

- Best parameters: `{"classifier__colsample_bytree": 1.0, "classifier__learning_rate": 0.1, "classifier__max_depth": 3, "classifier__min_child_weight": 1, "classifier__n_estimators": 400, "classifier__reg_lambda": 1, "classifier__scale_pos_weight": 10, "classifier__subsample": 0.75}`
- Test confusion matrix (actual rows [0, 1], predicted columns [0, 1]): `[[1947, 14], [11, 28]]`
- Test threshold: `0.50`

The dataset is synthetic. These scores describe performance on its held-out records and do not establish performance on factory data.
