"""Train both model families and package the CV-selected inference bundle."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.package_model import main as package_model
from scripts.train_models import main as train_models


def main() -> int:
    """Run tuning/evaluation, then package the training-selected model."""
    result = train_models()
    if result != 0:
        return result
    return package_model()


if __name__ == "__main__":
    raise SystemExit(main())
