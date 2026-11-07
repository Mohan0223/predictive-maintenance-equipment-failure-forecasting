"""Download and validate the public UCI AI4I CSV dataset."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
)
REQUIRED_COLUMNS = {
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
}
MAX_ARCHIVE_SIZE = 20 * 1024 * 1024


def _validate_csv(csv_bytes: bytes) -> None:
    """Reject an unexpected or corrupted archive member before writing it."""
    try:
        data = pd.read_csv(BytesIO(csv_bytes))
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Downloaded CSV is invalid: {exc}") from exc

    missing_columns = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if len(data) < 10_000 or missing_columns:
        raise ValueError(
            "Downloaded file does not match the expected AI4I schema. "
            f"Rows: {len(data):,}; missing columns: {missing_columns}"
        )


def download_dataset(*, force: bool = False) -> Path:
    """Download the UCI zip and atomically save its validated CSV member."""
    if DATASET_PATH.exists() and not force:
        _validate_csv(DATASET_PATH.read_bytes())
        return DATASET_PATH

    request = Request(
        DATASET_URL,
        headers={"User-Agent": "predictive-maintenance-portfolio/1.0"},
    )
    try:
        with urlopen(request, timeout=45) as response:
            archive_bytes = response.read(MAX_ARCHIVE_SIZE + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Could not download the UCI dataset: {exc}") from exc

    if len(archive_bytes) > MAX_ARCHIVE_SIZE:
        raise ValueError("UCI dataset archive exceeded the 20 MB safety limit.")

    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            csv_member = next(
                (
                    name
                    for name in archive.namelist()
                    if PurePosixPath(name).name.lower() == "ai4i2020.csv"
                ),
                None,
            )
            if csv_member is None:
                raise ValueError("The downloaded UCI archive has no ai4i2020.csv.")
            csv_bytes = archive.read(csv_member)
    except BadZipFile as exc:
        raise ValueError("The UCI download was not a valid ZIP archive.") from exc

    _validate_csv(csv_bytes)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = DATASET_PATH.with_suffix(".csv.tmp")
    try:
        temporary_path.write_bytes(csv_bytes)
        temporary_path.replace(DATASET_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)
    return DATASET_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again and replace the existing CSV",
    )
    arguments = parser.parse_args()
    try:
        dataset_path = download_dataset(force=arguments.force)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Dataset download failed: {exc}\n")

    print(f"Dataset ready: {dataset_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
