"""General utilities used across Meridian Retail Group notebooks and scripts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


DATASET_NAMES = (
    "customers", "products", "orders", "order_items", "support_tickets"
)


def get_project_root() -> Path:
    """Return the project root (the parent directory of ``src``)."""
    return Path(__file__).resolve().parents[1]


def ensure_directory(path: str | Path) -> Path:
    """Create a directory, including missing parents, and return its Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_csv(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV file with pandas and raise a clear error if it is absent."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    return pd.read_csv(file_path, **read_csv_kwargs)


def resolve_data_directory(
    project_root: str | Path | None = None,
    stage: str = "processed",
) -> Path:
    """Resolve raw/processed data paths for both official and ZIP folder names.

    Supports the official ``data/raw`` and ``data/processed`` names as well as
    the original ZIP's ``Data/Raw`` and ``Data/Cleaned`` names.
    """
    root = Path(project_root) if project_root is not None else get_project_root()
    options = {
        "raw": [root / "data" / "raw", root / "Data" / "Raw"],
        "processed": [root / "data" / "processed", root / "Data" / "Cleaned"],
        "cleaned": [root / "data" / "processed", root / "Data" / "Cleaned"],
    }
    key = stage.lower()
    if key not in options:
        raise ValueError("stage must be 'raw', 'processed' or 'cleaned'")
    for candidate in options[key]:
        if candidate.is_dir():
            return candidate
    # Return the official path even when it does not exist yet, for clear errors.
    return options[key][0]


def load_meridian_datasets(
    data_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    stage: str = "processed",
) -> dict[str, pd.DataFrame]:
    """Load all five Meridian CSV datasets from the selected data stage."""
    directory = Path(data_dir) if data_dir is not None else resolve_data_directory(
        project_root=project_root, stage=stage
    )
    datasets: dict[str, pd.DataFrame] = {}
    missing = []
    for name in DATASET_NAMES:
        file_path = directory / f"{name}.csv"
        if not file_path.is_file():
            missing.append(str(file_path))
        else:
            datasets[name] = pd.read_csv(file_path)
    if missing:
        raise FileNotFoundError(
            "Could not load all five Meridian datasets. Missing files:\n- "
            + "\n- ".join(missing)
        )
    return datasets


def dataset_summary(df: pd.DataFrame, dataset_name: str = "dataset") -> dict[str, int | str]:
    """Return the row, column, missing-value and duplicate-row counts."""
    return {
        "dataset": dataset_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def dataset_summary_table(
    datasets: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create a compact summary table for a dictionary of DataFrames."""
    rows = [dataset_summary(df, name) for name, df in datasets.items()]
    return pd.DataFrame(rows)


def count_by_category(
    df: pd.DataFrame,
    column_name: str,
    include_missing: bool = False,
) -> pd.Series:
    """Count records by category, optionally retaining missing values."""
    if column_name not in df.columns:
        raise KeyError(f"Column {column_name!r} does not exist in the DataFrame")
    return df[column_name].value_counts(dropna=not include_missing)


def set_random_seed(seed: int = 42) -> None:
    """Set Python and NumPy random seeds for reproducible operations."""
    random.seed(seed)
    np.random.seed(seed)
