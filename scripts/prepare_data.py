"""Clean, engineer, split, and encode the AI4I dataset without leakage."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

from predictive_maintenance.data.dataset import load_ai4i_dataset
from predictive_maintenance.data.preprocessing import (
    BASE_NUMERIC_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    prepare_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CANDIDATES = (
    PROJECT_ROOT / "data" / "ai4i2020.csv",
    PROJECT_ROOT
    / "ai4i+2020+predictive+maintenance+dataset"
    / "ai4i2020.csv",
)
TEST_FRACTION = 0.20


def find_dataset() -> Path:
    """Return the first available AI4I CSV path."""
    for candidate in DATASET_CANDIDATES:
        if candidate.is_file():
            return candidate

    expected_paths = "\n".join(f"- {path}" for path in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Could not find ai4i2020.csv. Place it at one of these paths:\n"
        f"{expected_paths}"
    )


def print_class_counts(name: str, labels: pd.Series) -> None:
    """Print class counts and proportions for a dataset split."""
    counts = labels.value_counts().sort_index()
    total = len(labels)
    print(f"{name} class distribution:")
    for label, count in counts.items():
        percentage = 100.0 * count / total if total else 0.0
        print(f"  {label}: {count:,} ({percentage:.2f}%)")


def print_missing_counts(name: str, features: pd.DataFrame) -> None:
    """Print feature missing-value counts before imputation."""
    print(f"{name} feature missing values before imputation:")
    for column, count in features.isna().sum().items():
        print(f"  {column}: {int(count):,}")


def report_iqr_outliers(features: pd.DataFrame) -> None:
    """Report IQR outliers in training data without removing observations."""
    columns = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
    print("Training-set IQR outlier counts (reported, not removed):")
    for column in columns:
        values = pd.to_numeric(features[column], errors="coerce").dropna()
        if values.empty:
            print(f"  {column}: unavailable (no numeric values)")
            continue

        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range
        outlier_count = int(((values < lower_bound) | (values > upper_bound)).sum())
        print(f"  {column}: {outlier_count:,}")


def main() -> int:
    """Prepare chronological train/test files and a reusable preprocessor."""
    try:
        dataset_path = find_dataset()
        data = load_ai4i_dataset(dataset_path)
    except (FileNotFoundError, OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"Could not load the dataset: {exc}", file=sys.stderr)
        return 1

    if data.empty:
        print(f"Dataset is empty: {dataset_path}", file=sys.stderr)
        return 1

    if "UDI" not in data.columns or TARGET_COLUMN not in data.columns:
        print(
            "The dataset must contain both 'UDI' and 'Machine failure' columns.",
            file=sys.stderr,
        )
        return 1

    original_rows = len(data)
    duplicate_rows = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()

    data["UDI"] = pd.to_numeric(data["UDI"], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    missing_target_rows = int(data[TARGET_COLUMN].isna().sum())
    missing_udi_rows = int(data["UDI"].isna().sum())
    data = data.dropna(subset=["UDI", TARGET_COLUMN]).copy()

    unexpected_targets = sorted(set(data[TARGET_COLUMN].unique()) - {0, 1})
    if unexpected_targets:
        print(
            f"Unexpected values in {TARGET_COLUMN}: {unexpected_targets}",
            file=sys.stderr,
        )
        return 1

    if data["UDI"].duplicated().any():
        print("UDI must be unique to preserve the dataset row order.", file=sys.stderr)
        return 1

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("int8")
    data = data.sort_values("UDI", kind="mergesort").reset_index(drop=True)

    if len(data) < 2:
        print("At least two valid records are required for a split.", file=sys.stderr)
        return 1

    split_index = int(len(data) * (1.0 - TEST_FRACTION))
    if split_index <= 0 or split_index >= len(data):
        print("The configured train/test split leaves an empty partition.", file=sys.stderr)
        return 1

    train_rows = data.iloc[:split_index].copy()
    test_rows = data.iloc[split_index:].copy()
    y_train = train_rows[TARGET_COLUMN]
    y_test = test_rows[TARGET_COLUMN]

    try:
        X_train_raw = prepare_features(train_rows)
        X_test_raw = prepare_features(test_rows)
    except ValueError as exc:
        print(f"Could not prepare model features: {exc}", file=sys.stderr)
        return 1

    print_missing_counts("Train", X_train_raw)
    print_missing_counts("Test", X_test_raw)

    try:
        preprocessor = build_preprocessor()
        X_train_array = preprocessor.fit_transform(X_train_raw)
        X_test_array = preprocessor.transform(X_test_raw)
    except (TypeError, ValueError) as exc:
        print(f"Could not fit/transform model features: {exc}", file=sys.stderr)
        return 1
    feature_names = preprocessor.get_feature_names_out()

    X_train = pd.DataFrame(X_train_array, columns=feature_names)
    X_test = pd.DataFrame(X_test_array, columns=feature_names)
    train_output = X_train.copy()
    test_output = X_test.copy()
    train_output[TARGET_COLUMN] = y_train.reset_index(drop=True)
    test_output[TARGET_COLUMN] = y_test.reset_index(drop=True)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    model_dir = PROJECT_ROOT / "models"
    try:
        processed_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        train_output.to_csv(processed_dir / "train.csv", index=False)
        test_output.to_csv(processed_dir / "test.csv", index=False)
        joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
    except OSError as exc:
        print(f"Could not save prepared outputs: {exc}", file=sys.stderr)
        return 1

    print(f"Source file: {dataset_path.relative_to(PROJECT_ROOT)}")
    print(f"Rows read: {original_rows:,}")
    print(f"Exact duplicate rows removed: {duplicate_rows:,}")
    print(f"Rows with missing target removed: {missing_target_rows:,}")
    print(f"Rows with missing UDI removed: {missing_udi_rows:,}")
    print(f"Rows retained: {len(data):,}")
    print(f"Features after one-hot encoding: {len(feature_names)}")
    print(
        "Chronological split by UDI: "
        f"train UDI {train_rows['UDI'].min():.0f} to {train_rows['UDI'].max():.0f}; "
        f"test UDI {test_rows['UDI'].min():.0f} to {test_rows['UDI'].max():.0f}"
    )
    print_class_counts("Train", y_train)
    print_class_counts("Test", y_test)
    report_iqr_outliers(X_train_raw)
    print(f"Saved: {processed_dir / 'train.csv'}")
    print(f"Saved: {processed_dir / 'test.csv'}")
    print(f"Saved: {model_dir / 'preprocessor.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
