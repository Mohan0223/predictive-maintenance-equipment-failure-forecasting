"""Generate exploratory plots and a short report for the AI4I training split."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from predictive_maintenance.data.dataset import load_ai4i_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "eda_summary.md"
TARGET_COLUMN = "Machine failure"
NUMERIC_FEATURES = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
REQUIRED_COLUMNS = ("UDI", "Type", TARGET_COLUMN) + NUMERIC_FEATURES
TRAIN_FRACTION = 0.80


def load_training_data() -> pd.DataFrame:
    """Load and return the first 80% of the cleaned, UDI-ordered dataset."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = load_ai4i_dataset(DATASET_PATH)
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    data = data.drop_duplicates().copy()
    data["UDI"] = pd.to_numeric(data["UDI"], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["UDI", TARGET_COLUMN]).copy()
    unexpected_targets = sorted(set(data[TARGET_COLUMN].unique()) - {0, 1})
    if unexpected_targets:
        raise ValueError(f"Unexpected target values: {unexpected_targets}")

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("int8")
    data = data.sort_values("UDI", kind="mergesort").reset_index(drop=True)
    split_index = int(len(data) * TRAIN_FRACTION)
    if split_index <= 0:
        raise ValueError("Not enough rows to create a training partition.")

    return data.iloc[:split_index].copy()


def save_figure(figure: plt.Figure, filename: str) -> None:
    """Save a plot in the report figure directory and close it."""
    output_path = FIGURE_DIR / filename
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")


def plot_failure_distribution(data: pd.DataFrame) -> None:
    counts = data[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    labels = ["No failure (0)", "Failure (1)"]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(labels, counts.values, color=["#2f855a", "#c53030"])
    axis.set_title("Recorded equipment failure distribution")
    axis.set_ylabel("Number of records")
    axis.set_xlabel("Machine failure label")
    axis.bar_label(bars, labels=[f"{value:,}" for value in counts.values], padding=3)
    save_figure(figure, "failure_distribution.png")


def plot_temperature_vs_failure(data: pd.DataFrame) -> None:
    temperature_columns = (
        "Air temperature [K]",
        "Process temperature [K]",
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for axis, column in zip(axes, temperature_columns):
        sns.boxplot(
            data=data,
            x=TARGET_COLUMN,
            y=column,
            order=[0, 1],
            showfliers=False,
            ax=axis,
        )
        axis.set_title(f"{column} by recorded failure")
        axis.set_xlabel("Machine failure (0 = no, 1 = yes)")
        axis.set_ylabel(column)

    save_figure(figure, "temperature_vs_failure.png")


def plot_unavailable_sensors() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    unavailable = ("Vibration", "Pressure")

    for axis, sensor in zip(axes, unavailable):
        axis.axis("off")
        axis.set_title(f"{sensor} vs failure: unavailable")
        axis.text(
            0.5,
            0.55,
            f"The AI4I CSV contains no {sensor.lower()} measurements.\n"
            "No values are plotted.",
            ha="center",
            va="center",
            fontsize=12,
            wrap=True,
        )

    figure.suptitle("Requested sensors absent from this dataset", fontsize=14)
    save_figure(figure, "vibration_pressure_unavailable.png")


def plot_correlation_heatmap(data: pd.DataFrame) -> None:
    columns = list(NUMERIC_FEATURES) + [TARGET_COLUMN]
    correlation = data.loc[:, columns].corr(method="spearman")

    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.5,
        ax=axis,
    )
    axis.set_title("Spearman correlations: measured inputs and failure label")
    save_figure(figure, "correlation_heatmap.png")


def plot_feature_distributions(data: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))

    for axis, column in zip(axes.flat, NUMERIC_FEATURES):
        sns.histplot(data=data, x=column, bins=30, kde=True, ax=axis)
        axis.set_title(column)

    axes.flat[-1].axis("off")
    figure.suptitle("Training-set sensor and operating-feature distributions")
    save_figure(figure, "feature_distributions.png")


def plot_operating_patterns(data: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    operating_features = (
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    )

    for axis, column in zip(axes.flat[:3], operating_features):
        sns.boxplot(
            data=data,
            x=TARGET_COLUMN,
            y=column,
            order=[0, 1],
            showfliers=False,
            ax=axis,
        )
        axis.set_title(f"{column} by recorded failure")
        axis.set_xlabel("Machine failure (0 = no, 1 = yes)")

    type_summary = (
        data.groupby("Type", observed=True)[TARGET_COLUMN]
        .agg(failure_rate="mean", records="size")
        .reindex(["L", "M", "H"])
        .dropna()
    )
    type_axis = axes.flat[3]
    bars = type_axis.bar(
        type_summary.index,
        type_summary["failure_rate"] * 100,
        color="#4c78a8",
    )
    type_axis.set_title("Failure rate by product type")
    type_axis.set_xlabel("Product type")
    type_axis.set_ylabel("Recorded failures (%)")
    max_failure_rate = float((type_summary["failure_rate"] * 100).max())
    type_axis.set_ylim(0, max_failure_rate * 1.35)
    type_axis.bar_label(
        bars,
        labels=[
            f"{rate:.1f}%\nn={count:,}"
            for rate, count in zip(
                type_summary["failure_rate"] * 100,
                type_summary["records"],
            )
        ],
        padding=3,
    )

    save_figure(figure, "operating_patterns_vs_failure.png")


def write_summary(data: pd.DataFrame) -> None:
    class_counts = data[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    total = len(data)
    failure_rate = 100.0 * class_counts.loc[1] / total if total else 0.0

    median_rows = []
    for column in NUMERIC_FEATURES:
        normal_median = data.loc[data[TARGET_COLUMN] == 0, column].median()
        failure_median = data.loc[data[TARGET_COLUMN] == 1, column].median()
        median_rows.append(
            f"| {column} | {normal_median:.3f} | {failure_median:.3f} |"
        )

    type_rows = []
    type_summary = (
        data.groupby("Type", observed=True)[TARGET_COLUMN]
        .agg(failure_rate="mean", records="size")
        .reindex(["L", "M", "H"])
        .dropna()
    )
    for product_type, row in type_summary.iterrows():
        type_rows.append(
            f"| {product_type} | {int(row['records']):,} | "
            f"{100.0 * row['failure_rate']:.2f}% |"
        )

    report = "\n".join(
        [
            "# Exploratory data analysis summary",
            "",
            "The charts use only the first 80% of rows ordered by UDI, "
            "matching the training partition prepared in Phase 3.",
            "",
            f"- Training records: {total:,}",
            f"- No-failure records: {int(class_counts.loc[0]):,}",
            f"- Failure records: {int(class_counts.loc[1]):,}",
            f"- Recorded failure rate: {failure_rate:.2f}%",
            "- Dataset provenance: synthetic AI4I benchmark data, not factory telemetry.",
            "- Vibration and pressure are absent, so no sensor-vs-failure values "
            "are available for those fields.",
            "",
            "## Median measured features by recorded outcome",
            "",
            "| Feature | No failure | Failure |",
            "|---|---:|---:|",
            *median_rows,
            "",
            "## Failure rate by product type",
            "",
            "| Type | Records | Failure rate |",
            "|---|---:|---:|",
            *type_rows,
            "",
            "These are descriptive summaries of a synthetic dataset. "
            "Differences and correlations are not causal evidence.",
            "",
        ]
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


def main() -> int:
    """Create the EDA figures and summary report."""
    try:
        data = load_training_data()
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style="whitegrid", context="notebook")
        plot_failure_distribution(data)
        plot_temperature_vs_failure(data)
        plot_unavailable_sensors()
        plot_correlation_heatmap(data)
        plot_feature_distributions(data)
        plot_operating_patterns(data)
        write_summary(data)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(f"EDA generation failed: {exc}", file=sys.stderr)
        return 1

    print(f"EDA complete. Figures are in {FIGURE_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
