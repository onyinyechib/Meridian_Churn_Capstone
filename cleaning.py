"""Reusable data-cleaning functions for the Meridian Retail Group project.

The transformations in this module are based on the documented Phase 6
cleaning workflow. Raw input DataFrames are never modified in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


DATASET_NAMES = (
    "customers",
    "products",
    "orders",
    "order_items",
    "support_tickets",
)

DATE_COLUMNS = {
    "customers": ["date_of_birth", "signup_date"],
    "products": ["launch_date"],
    "orders": ["order_date"],
    "order_items": [],
    "support_tickets": ["ticket_date"],
}

NUMERIC_COLUMNS = {
    "products": ["unit_cost", "unit_price"],
    "orders": ["shipping_cost", "discount_pct", "total_amount"],
    "order_items": ["quantity", "unit_price_at_purchase"],
    "support_tickets": ["resolution_time_hours", "satisfaction_score"],
}

TITLE_CASE_COLUMNS = {
    "customers": [
        "gender", "state", "city", "membership_tier",
        "preferred_channel", "referral_source",
    ],
    "products": ["category", "sub_category", "brand"],
    "orders": ["channel", "order_status", "payment_method"],
    "support_tickets": ["issue_category", "channel"],
}

MISSING_FLAG_COLUMNS = {
    "customers": [
        "email", "phone", "date_of_birth", "city",
        "marketing_opt_in", "referral_source",
    ],
    "products": ["sub_category"],
    "orders": ["order_date"],
    "order_items": ["unit_price_at_purchase"],
    "support_tickets": ["satisfaction_score"],
}

BOOLEAN_VALUE_MAP = {
    "Y": "Yes",
    "YES": "Yes",
    "1": "Yes",
    "TRUE": "Yes",
    "N": "No",
    "NO": "No",
    "0": "No",
    "FALSE": "No",
}


def _require_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"The {dataset_name!r} dataset is missing required columns: {missing}"
        )


def clean_meridian_datasets(
    raw_datasets: Mapping[str, pd.DataFrame],
    snapshot_date: str | pd.Timestamp = "2026-07-01",
) -> dict[str, pd.DataFrame]:
    """Clean the five raw Meridian datasets using the Phase 6 rules.

    Exact duplicate rows are removed only from ``orders`` and ``order_items``.
    Repeated customer email addresses and repeated IDs with conflicting records
    are not automatically deleted. Selected invalid values are flagged, and
    only the corrections explicitly documented in Phase 6 are applied.

    Parameters
    ----------
    raw_datasets:
        Mapping keyed by the five lowercase dataset names.
    snapshot_date:
        Date used to flag implausible customer birth dates.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A new dictionary containing cleaned copies of all five datasets.
    """
    missing_names = [name for name in DATASET_NAMES if name not in raw_datasets]
    if missing_names:
        raise KeyError(f"Missing required datasets: {missing_names}")

    cleaned = {
        name: raw_datasets[name].copy(deep=True)
        for name in DATASET_NAMES
    }
    snapshot = pd.Timestamp(snapshot_date)

    # Phase 6 removed verified exact duplicate transaction rows only.
    for name in ("orders", "order_items"):
        cleaned[name] = cleaned[name].drop_duplicates().copy()

    # Strip leading/trailing whitespace from text columns in all datasets.
    for name, df in cleaned.items():
        text_columns = df.select_dtypes(include=["object", "string"]).columns
        for column in text_columns:
            df[column] = df[column].astype("string").str.strip()

    # Parse known date columns; malformed dates become missing values.
    for name, columns in DATE_COLUMNS.items():
        for column in columns:
            if column in cleaned[name].columns:
                cleaned[name][column] = pd.to_datetime(
                    cleaned[name][column], format="mixed", errors="coerce"
                )

    # Convert selected numeric columns, allowing currency symbols and commas.
    for name, columns in NUMERIC_COLUMNS.items():
        for column in columns:
            if column in cleaned[name].columns:
                text_values = (
                    cleaned[name][column]
                    .astype("string")
                    .str.replace(r"[$,]", "", regex=True)
                    .str.strip()
                )
                cleaned[name][column] = pd.to_numeric(
                    text_values, errors="coerce"
                )

    # Standardise selected category labels to title case.
    for name, columns in TITLE_CASE_COLUMNS.items():
        for column in columns:
            if column in cleaned[name].columns:
                cleaned[name][column] = (
                    cleaned[name][column].astype("string").str.strip().str.title()
                )

    # Match the Phase 6 mappings: unrecognised values become missing.
    if "marketing_opt_in" in cleaned["customers"].columns:
        cleaned["customers"]["marketing_opt_in"] = (
            cleaned["customers"]["marketing_opt_in"]
            .astype("string").str.strip().str.upper()
            .map(BOOLEAN_VALUE_MAP)
        )

    if "is_active" in cleaned["products"].columns:
        cleaned["products"]["is_active"] = (
            cleaned["products"]["is_active"]
            .astype("string").str.strip().str.upper()
            .map(BOOLEAN_VALUE_MAP)
        )

    # Capture missingness before selected invalid values are changed to NaN.
    for name, columns in MISSING_FLAG_COLUMNS.items():
        for column in columns:
            if column in cleaned[name].columns:
                cleaned[name][f"{column}_missing"] = (
                    cleaned[name][column].isna().astype("int8")
                )

    customers = cleaned["customers"]
    products = cleaned["products"]
    orders = cleaned["orders"]
    order_items = cleaned["order_items"]
    tickets = cleaned["support_tickets"]

    # Customer birth dates outside the plausible range are flagged and nulled.
    _require_columns(customers, ["date_of_birth"], "customers")
    customers["dob_before_1900_invalid"] = (
        customers["date_of_birth"] < pd.Timestamp("1900-01-01")
    )
    customers["dob_after_snapshot_invalid"] = (
        customers["date_of_birth"] > snapshot
    )
    invalid_dob = (
        customers["dob_before_1900_invalid"]
        | customers["dob_after_snapshot_invalid"]
    )
    customers.loc[invalid_dob, "date_of_birth"] = pd.NaT

    # Product cost/price inconsistencies are flagged but left unchanged.
    _require_columns(products, ["unit_cost", "unit_price"], "products")
    products["cost_exceeds_price_invalid"] = (
        products["unit_cost"] > products["unit_price"]
    )

    # Invalid negative shipping costs are flagged and set to missing.
    _require_columns(orders, ["shipping_cost"], "orders")
    orders["negative_shipping_cost_invalid"] = orders["shipping_cost"] < 0
    orders.loc[orders["negative_shipping_cost_invalid"], "shipping_cost"] = float("nan")

    # Zero/negative order quantities are flagged and set to missing.
    _require_columns(order_items, ["quantity"], "order_items")
    order_items["nonpositive_quantity_invalid"] = order_items["quantity"] <= 0
    order_items.loc[
        order_items["nonpositive_quantity_invalid"], "quantity"
    ] = pd.NA

    # Negative support-ticket resolution durations are flagged and nulled.
    _require_columns(tickets, ["resolution_time_hours"], "support_tickets")
    tickets["negative_resolution_time_invalid"] = (
        tickets["resolution_time_hours"] < 0
    )
    tickets.loc[
        tickets["negative_resolution_time_invalid"], "resolution_time_hours"
    ] = float("nan")

    # Preserve order records with missing customer references and flag them.
    _require_columns(customers, ["customer_id"], "customers")
    _require_columns(orders, ["customer_id"], "orders")
    known_customer_ids = set(customers["customer_id"].dropna())
    orders["customer_reference_missing"] = (
        ~orders["customer_id"].isin(known_customer_ids)
    ).astype("int8")

    return cleaned


def load_raw_datasets(raw_data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the five raw CSV files from a directory."""
    directory = Path(raw_data_dir)
    datasets: dict[str, pd.DataFrame] = {}
    for name in DATASET_NAMES:
        file_path = directory / f"{name}.csv"
        if not file_path.is_file():
            raise FileNotFoundError(f"Required raw dataset not found: {file_path}")
        datasets[name] = pd.read_csv(file_path)
    return datasets


def save_cleaned_datasets(
    datasets: Mapping[str, pd.DataFrame], output_dir: str | Path
) -> list[Path]:
    """Save cleaned datasets as CSV files and return their paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for name in DATASET_NAMES:
        if name not in datasets:
            raise KeyError(f"Cleaned dataset {name!r} was not supplied")
        destination = directory / f"{name}.csv"
        datasets[name].to_csv(destination, index=False)
        saved_paths.append(destination)
    return saved_paths
