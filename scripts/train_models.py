"""Tune, compare, and evaluate Random Forest and XGBoost pipelines."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from predictive_maintenance.data.dataset import load_ai4i_dataset
from predictive_maintenance.data.preprocessing import (
    TARGET_COLUMN,
    build_preprocessor,
    prepare_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
TRAIN_FRACTION = 0.80
N_SPLITS = 4
N_ITERATIONS = 12
RANDOM_STATE = 42
CV = TimeSeriesSplit(n_splits=N_SPLITS)
SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def load_ordered_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load AI4I and recreate the Phase 3 UDI-ordered split."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = load_ai4i_dataset(DATASET_PATH)
    required_columns = {"UDI", TARGET_COLUMN}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    data = data.drop_duplicates().copy()
    data["UDI"] = pd.to_numeric(data["UDI"], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.dropna(subset=["UDI", TARGET_COLUMN]).copy()

    unexpected_targets = sorted(set(data[TARGET_COLUMN].unique()) - {0, 1})
    if unexpected_targets:
        raise ValueError(f"Unexpected target values: {unexpected_targets}")
    if data["UDI"].duplicated().any():
        raise ValueError("UDI must be unique to preserve row order.")

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("int8")
    data = data.sort_values("UDI", kind="mergesort").reset_index(drop=True)
    split_index = int(len(data) * TRAIN_FRACTION)
    if split_index <= 0 or split_index >= len(data):
        raise ValueError("The configured train/test split is empty.")

    train_rows = data.iloc[:split_index].copy()
    test_rows = data.iloc[split_index:].copy()
    X_train = prepare_features(train_rows)
    y_train = train_rows[TARGET_COLUMN].reset_index(drop=True)
    X_test = prepare_features(test_rows)
    y_test = test_rows[TARGET_COLUMN].reset_index(drop=True)
    return X_train, y_train, X_test, y_test, test_rows


def make_pipeline(classifier: Any) -> Pipeline:
    """Build a pipeline that refits preprocessing inside every CV fold."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def make_search(
    pipeline: Pipeline,
    parameter_distributions: dict[str, list[Any]],
) -> RandomizedSearchCV:
    """Create deterministic randomized tuning using time-ordered folds."""
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=N_ITERATIONS,
        scoring=SCORING,
        refit="f1",
        cv=CV,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        error_score="raise",
        return_train_score=False,
    )


def cv_metrics(search: RandomizedSearchCV) -> dict[str, float]:
    """Return cross-validation metrics for the selected hyperparameters."""
    index = search.best_index_
    results = search.cv_results_
    metric_names = ("precision", "recall", "f1", "roc_auc", "average_precision")
    return {
        name: float(results[f"mean_test_{name}"][index])
        for name in metric_names
    }


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Calculate requested metrics at the default 0.5 threshold."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    if y_test.nunique() == 2:
        roc_auc: float | None = float(roc_auc_score(y_test, probabilities))
        average_precision: float | None = float(
            average_precision_score(y_test, probabilities)
        )
    else:
        roc_auc = None
        average_precision = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "confusion_matrix": matrix.tolist(),
    }
    return metrics, predictions, probabilities


def save_confusion_matrix(
    y_true: pd.Series,
    predictions: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Save a labeled held-out confusion matrix image."""
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=["No failure", "Failure"],
        values_format="d",
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"{model_name} — held-out test set")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_report(
    results: dict[str, dict[str, Any]],
    selected_model: str,
    train_size: int,
    test_size: int,
) -> None:
    """Write a human-readable model comparison report."""
    rows = []
    for name, result in results.items():
        cv = result["cross_validation"]
        test = result["test"]
        rows.append(
            f"| {name} | {format_metric(cv['precision'])} | "
            f"{format_metric(cv['recall'])} | {format_metric(cv['f1'])} | "
            f"{format_metric(cv['roc_auc'])} | "
            f"{format_metric(test['accuracy'])} | "
            f"{format_metric(test['precision'])} | "
            f"{format_metric(test['recall'])} | "
            f"{format_metric(test['f1'])} | "
            f"{format_metric(test['roc_auc'])} | "
            f"{format_metric(test['average_precision'])} |"
        )

    detail_sections = []
    for name, result in results.items():
        matrix = result["test"]["confusion_matrix"]
        detail_sections.extend(
            [
                f"### {name}",
                "",
                f"- Best parameters: `{json.dumps(result['best_parameters'], sort_keys=True)}`",
                f"- Test confusion matrix (actual rows [0, 1], predicted columns [0, 1]): "
                f"`{matrix}`",
                f"- Test threshold: `{result['decision_threshold']:.2f}`",
                "",
            ]
        )

    report = "\n".join(
        [
            "# Model comparison",
            "",
            "## Evaluation protocol",
            "",
            f"- Training rows: {train_size:,}; held-out test rows: {test_size:,}.",
            f"- Hyperparameter tuning used {N_SPLITS}-fold `TimeSeriesSplit` "
            "inside the training partition.",
            "- Each cross-validation fold fit its own imputer and encoder inside "
            "the model pipeline.",
            "- Search selection metric: F1 at the default 0.50 threshold. "
            "Recall and precision are reported alongside it.",
            "- The test set was evaluated once after tuning; no threshold was "
            "selected using test labels.",
            "- Accuracy is included but can look high simply because failures "
            "are rare. Review precision, recall, F1, ROC-AUC, average precision, "
            "and the confusion matrices together.",
            "",
            "## Metrics",
            "",
            "| Model | CV precision | CV recall | CV F1 | CV ROC-AUC | "
            "Test accuracy | Test precision | Test recall | Test F1 | "
            "Test ROC-AUC | Test average precision |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Selected using training-only CV F1 (recall and precision as "
            f"tie-breakers): **{selected_model}**.",
            "",
            "## Best parameters and confusion matrices",
            "",
            *detail_sections,
            "The dataset is synthetic. These scores describe performance on its "
            "held-out records and do not establish performance on factory data.",
            "",
        ]
    )
    report_path = PROJECT_ROOT / "reports" / "model_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved: {report_path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    """Tune both classifiers, evaluate them, and save comparison artifacts."""
    try:
        X_train, y_train, X_test, y_test, test_rows = load_ordered_split()

        rf_pipeline = make_pipeline(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
        )
        rf_parameters = {
            "classifier__n_estimators": [150, 250, 400],
            "classifier__max_depth": [None, 8, 14, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", 0.7],
            "classifier__class_weight": [None, "balanced", "balanced_subsample"],
        }
        rf_search = make_search(rf_pipeline, rf_parameters)
        print("Tuning Random Forest...")
        rf_search.fit(X_train, y_train)

        xgb_pipeline = make_pipeline(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            )
        )
        xgb_parameters = {
            "classifier__n_estimators": [150, 250, 400],
            "classifier__max_depth": [2, 3, 4, 5],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.75, 1.0],
            "classifier__colsample_bytree": [0.75, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__scale_pos_weight": [1, 5, 10, 20, 30],
            "classifier__reg_lambda": [1, 5, 10],
        }
        xgb_search = make_search(xgb_pipeline, xgb_parameters)
        print("Tuning XGBoost...")
        xgb_search.fit(X_train, y_train)

        searches = {
            "Random Forest": rf_search,
            "XGBoost": xgb_search,
        }
        results: dict[str, dict[str, Any]] = {}
        predictions_by_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for name, search in searches.items():
            model = search.best_estimator_
            test_metrics, predictions, probabilities = evaluate_model(
                model, X_test, y_test
            )
            results[name] = {
                "cross_validation": cv_metrics(search),
                "test": test_metrics,
                "best_parameters": search.best_params_,
                "decision_threshold": 0.5,
            }
            predictions_by_model[name] = (predictions, probabilities)
            print(
                f"{name}: CV F1={results[name]['cross_validation']['f1']:.4f}; "
                f"test precision={test_metrics['precision']:.4f}, "
                f"recall={test_metrics['recall']:.4f}, "
                f"F1={test_metrics['f1']:.4f}"
            )

        selected_model = max(
            results,
            key=lambda name: (
                results[name]["cross_validation"]["f1"],
                results[name]["cross_validation"]["recall"],
                results[name]["cross_validation"]["precision"],
            ),
        )

        model_dir = PROJECT_ROOT / "models"
        metrics_dir = PROJECT_ROOT / "reports"
        figure_dir = PROJECT_ROOT / "reports" / "figures"
        model_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        model_paths = {
            "Random Forest": model_dir / "random_forest_tuned.joblib",
            "XGBoost": model_dir / "xgboost_tuned.joblib",
        }
        prediction_output = pd.DataFrame(
            {
                "UDI": test_rows["UDI"].to_numpy(),
                "actual_failure": y_test.to_numpy(),
            }
        )

        for name, search in searches.items():
            joblib.dump(search.best_estimator_, model_paths[name])
            predictions, probabilities = predictions_by_model[name]
            short_name = "random_forest" if name == "Random Forest" else "xgboost"
            prediction_output[f"{short_name}_prediction"] = predictions
            prediction_output[f"{short_name}_probability"] = probabilities
            save_confusion_matrix(
                y_test,
                predictions,
                name,
                figure_dir / f"{short_name}_confusion_matrix.png",
            )
            print(f"Saved: {model_paths[name].relative_to(PROJECT_ROOT)}")

        prediction_path = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_output.to_csv(prediction_path, index=False)

        metrics_path = metrics_dir / "metrics.json"
        metrics_payload = {
            "selected_model": selected_model,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "results": results,
        }
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, default=str),
            encoding="utf-8",
        )
        write_report(results, selected_model, len(X_train), len(X_test))

        print(f"Selected by training CV: {selected_model}")
        print(f"Saved: {metrics_path.relative_to(PROJECT_ROOT)}")
        print(f"Saved: {prediction_path.relative_to(PROJECT_ROOT)}")
        return 0
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(f"Model training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
