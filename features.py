"""Reusable customer churn and feature-engineering functions.

This module keeps the Phase 7 descriptive churn/RFM analysis separate from
Phase 9's predictive modelling population and time-based target definition.
"""

from __future__ import annotations

import pandas as pd


STATE_ABBREVIATIONS = {
    "Az": "Arizona", "Ca": "California", "Fl": "Florida",
    "Ga": "Georgia", "Il": "Illinois", "In": "Indiana",
    "Ma": "Massachusetts", "Md": "Maryland", "Mi": "Michigan",
    "Mo": "Missouri", "Nc": "North Carolina", "Nj": "New Jersey",
    "Ny": "New York", "Oh": "Ohio", "Pa": "Pennsylvania",
    "Tn": "Tennessee", "Tx": "Texas", "Va": "Virginia",
    "Wa": "Washington", "Wi": "Wisconsin",
}


def _require_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{table_name} is missing required columns: {missing}")


def _completed_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of orders with parsed dates and completed statuses only."""
    _require_columns(orders, ["order_id", "customer_id", "order_date", "order_status"], "orders")
    result = orders.copy()
    result["order_date"] = pd.to_datetime(result["order_date"], errors="coerce")
    status = result["order_status"].astype("string").str.strip().str.lower()
    return result.loc[status.eq("completed") & result["order_date"].notna()].copy()


def build_phase7_customer_features(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    support_tickets: pd.DataFrame,
    snapshot_date: str | pd.Timestamp = "2026-07-01",
    inactivity_days: int = 180,
) -> pd.DataFrame:
    """Build the Phase 7 customer table: 180-day churn label, RFM and support.

    The churn window is ``[snapshot_date - inactivity_days, snapshot_date)``:
    the start date is included and the snapshot date itself is excluded.
    RFM values use all valid completed orders before the snapshot date.
    """
    _require_columns(customers, ["customer_id"], "customers")
    _require_columns(orders, ["total_amount"], "orders")
    _require_columns(support_tickets, [
        "customer_id", "ticket_id", "ticket_date",
        "satisfaction_score", "resolution_time_hours",
    ], "support_tickets")

    snapshot = pd.Timestamp(snapshot_date).normalize()
    window_start = snapshot - pd.Timedelta(days=inactivity_days)
    all_completed = _completed_orders(orders)
    prior_completed = all_completed.loc[all_completed["order_date"] < snapshot].copy()
    in_window = prior_completed.loc[prior_completed["order_date"] >= window_start]

    recent_counts = (
        in_window.groupby("customer_id")["order_id"]
        .nunique()
        .rename("completed_orders_last_180_days")
    )

    result = customers.copy()
    result = result.merge(
        recent_counts, left_on="customer_id", right_index=True, how="left"
    )
    result["completed_orders_last_180_days"] = (
        result["completed_orders_last_180_days"].fillna(0).astype(int)
    )
    result["churn"] = result["completed_orders_last_180_days"].eq(0).astype("int8")

    # Aggregate at order level so joining order items cannot double-count value.
    rfm = (
        prior_completed.groupby("customer_id")
        .agg(
            last_completed_order=("order_date", "max"),
            frequency=("order_id", "nunique"),
            monetary=("total_amount", "sum"),
        )
        .reset_index()
    )
    rfm["recency_days"] = (snapshot - rfm["last_completed_order"]).dt.days
    rfm = rfm[["customer_id", "recency_days", "frequency", "monetary"]]
    result = result.merge(rfm, on="customer_id", how="left")
    result["frequency"] = result["frequency"].fillna(0).astype(int)
    result["monetary"] = result["monetary"].fillna(0)

    tickets = support_tickets.copy()
    tickets["ticket_date"] = pd.to_datetime(tickets["ticket_date"], errors="coerce")
    tickets = tickets.loc[
        tickets["ticket_date"].notna()
        & (tickets["ticket_date"] < snapshot)
        & tickets["customer_id"].notna()
    ].copy()
    support = (
        tickets.groupby("customer_id")
        .agg(
            ticket_count=("ticket_id", "nunique"),
            avg_satisfaction=("satisfaction_score", "mean"),
            avg_resolution_hours=("resolution_time_hours", "mean"),
        )
        .reset_index()
    )
    result = result.merge(support, on="customer_id", how="left")
    result["ticket_count"] = result["ticket_count"].fillna(0).astype(int)
    return result


def churn_summary(
    data: pd.DataFrame,
    segment_col: str,
    target_col: str = "churn",
    customer_id_col: str = "customer_id",
) -> pd.DataFrame:
    """Summarise customer counts and churn percentages for one segment."""
    _require_columns(data, [segment_col, target_col, customer_id_col], "customer data")
    summary = (
        data.groupby(segment_col, dropna=False)
        .agg(
            customers=(customer_id_col, "nunique"),
            churned_customers=(target_col, "sum"),
            churn_rate=(target_col, "mean"),
        )
        .reset_index()
    )
    summary["churn_rate_pct"] = summary["churn_rate"] * 100
    return (
        summary.drop(columns="churn_rate")
        .sort_values("churn_rate_pct", ascending=False)
        .round({"churn_rate_pct": 2})
        .reset_index(drop=True)
    )


def build_phase9_churn_target(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    outcome_start: str | pd.Timestamp = "2026-01-02",
    outcome_end: str | pd.Timestamp = "2026-06-30",
) -> pd.DataFrame:
    """Create Phase 9 target: no completed order in the inclusive outcome period."""
    _require_columns(customers, ["customer_id"], "customers")
    completed = _completed_orders(orders)
    start = pd.Timestamp(outcome_start).normalize()
    end_exclusive = pd.Timestamp(outcome_end).normalize() + pd.Timedelta(days=1)
    period_orders = completed.loc[
        completed["order_date"].ge(start) & completed["order_date"].lt(end_exclusive)
    ]
    counts = period_orders.groupby("customer_id")["order_id"].nunique()
    target = customers[["customer_id"]].copy()
    target["completed_orders_outcome"] = (
        target["customer_id"].map(counts).fillna(0).astype(int)
    )
    target["churned"] = target["completed_orders_outcome"].eq(0).astype("int8")
    return target


def build_historical_purchase_features(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    prediction_cutoff: str | pd.Timestamp = "2026-01-02",
) -> pd.DataFrame:
    """Create Phase 9 historical frequency, value, recency and quantity features."""
    _require_columns(customers, ["customer_id"], "customers")
    _require_columns(orders, ["total_amount"], "orders")
    _require_columns(order_items, ["order_id", "quantity"], "order_items")
    cutoff = pd.Timestamp(prediction_cutoff).normalize()
    history = _completed_orders(orders)
    history = history.loc[history["order_date"] < cutoff].copy()

    order_features = (
        history.groupby("customer_id")
        .agg(
            historical_frequency=("order_id", "nunique"),
            historical_monetary=("total_amount", "sum"),
            last_purchase_date=("order_date", "max"),
            historical_avg_order_value=("total_amount", "mean"),
        )
        .reset_index()
    )
    order_features["historical_recency"] = (
        cutoff - order_features["last_purchase_date"]
    ).dt.days

    historical_items = order_items.loc[
        order_items["order_id"].isin(history["order_id"])
    ].copy()
    order_quantity = (
        historical_items.groupby("order_id")["quantity"]
        .sum()
        .rename("order_quantity")
    )
    order_customer_quantity = history[["order_id", "customer_id"]].merge(
        order_quantity, on="order_id", how="left"
    )
    customer_quantity = (
        order_customer_quantity.groupby("customer_id")["order_quantity"]
        .sum()
        .rename("historical_quantity")
        .reset_index()
    )
    order_features = order_features.merge(customer_quantity, on="customer_id", how="left")

    # Start with all customers, retaining those with no historical completed orders.
    result = customers[["customer_id"]].merge(
        order_features, on="customer_id", how="left"
    )
    zero_fill_columns = [
        "historical_frequency", "historical_monetary", "historical_quantity"
    ]
    result[zero_fill_columns] = result[zero_fill_columns].fillna(0)
    result["historical_frequency"] = result["historical_frequency"].astype(int)
    result["historical_quantity"] = result["historical_quantity"].fillna(0)
    return result


def build_customer_profile_features(
    customers: pd.DataFrame,
    prediction_cutoff: str | pd.Timestamp = "2026-01-02",
) -> pd.DataFrame:
    """Create Phase 9 profile features for customers registered before cutoff."""
    required = [
        "customer_id", "signup_date", "membership_tier", "preferred_channel",
        "state", "marketing_opt_in", "referral_source",
    ]
    _require_columns(customers, required, "customers")
    cutoff = pd.Timestamp(prediction_cutoff).normalize()
    profile = customers.copy()
    profile["signup_date"] = pd.to_datetime(profile["signup_date"], errors="coerce")
    profile = profile.loc[profile["signup_date"] < cutoff].copy()
    profile["customer_tenure_days"] = (cutoff - profile["signup_date"]).dt.days
    selected = profile[[
        "customer_id", "membership_tier", "preferred_channel", "state",
        "marketing_opt_in", "referral_source", "customer_tenure_days",
    ]].copy()
    selected["marketing_opt_in"] = selected["marketing_opt_in"].fillna("Unknown")
    selected["referral_source"] = selected["referral_source"].fillna(
        "Unknown / Not Recorded"
    )
    selected["state"] = selected["state"].replace(STATE_ABBREVIATIONS)
    return selected


def build_historical_support_features(
    support_tickets: pd.DataFrame,
    eligible_customers: pd.DataFrame,
    prediction_cutoff: str | pd.Timestamp = "2026-01-02",
) -> pd.DataFrame:
    """Aggregate historical support activity to one row per eligible customer."""
    required = [
        "customer_id", "ticket_id", "ticket_date", "resolution_time_hours",
        "satisfaction_score", "issue_category",
    ]
    _require_columns(support_tickets, required, "support_tickets")
    _require_columns(eligible_customers, ["customer_id"], "eligible_customers")
    cutoff = pd.Timestamp(prediction_cutoff).normalize()
    tickets = support_tickets.copy()
    tickets["ticket_date"] = pd.to_datetime(tickets["ticket_date"], errors="coerce")
    eligible_ids = set(eligible_customers["customer_id"].dropna())
    tickets = tickets.loc[
        tickets["ticket_date"].lt(cutoff)
        & tickets["customer_id"].isin(eligible_ids)
    ].copy()

    if "satisfaction_score_missing" not in tickets.columns:
        tickets["satisfaction_score_missing"] = (
            tickets["satisfaction_score"].isna().astype("int8")
        )
    else:
        tickets["satisfaction_score_missing"] = (
            pd.to_numeric(tickets["satisfaction_score_missing"], errors="coerce")
            .fillna(0)
        )

    features = (
        tickets.groupby("customer_id")
        .agg(
            historical_ticket_count=("ticket_id", "nunique"),
            historical_avg_resolution_hours=("resolution_time_hours", "mean"),
            historical_avg_satisfaction=("satisfaction_score", "mean"),
            historical_unique_issue_categories=("issue_category", "nunique"),
            historical_missing_satisfaction_count=(
                "satisfaction_score_missing", "sum"
            ),
        )
        .reset_index()
    )
    result = eligible_customers[["customer_id"]].merge(
        features, on="customer_id", how="left"
    )
    count_columns = [
        "historical_ticket_count",
        "historical_unique_issue_categories",
        "historical_missing_satisfaction_count",
    ]
    result[count_columns] = result[count_columns].fillna(0)
    for column in count_columns:
        result[column] = result[column].astype(int)
    return result


def build_phase9_modelling_dataset(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    support_tickets: pd.DataFrame,
    prediction_cutoff: str | pd.Timestamp = "2026-01-02",
    outcome_start: str | pd.Timestamp = "2026-01-02",
    outcome_end: str | pd.Timestamp = "2026-06-30",
) -> pd.DataFrame:
    """Recreate the Phase 9 modelling table using strictly separated time periods.

    Predictors are derived from records before ``prediction_cutoff``. The
    ``churned`` label is derived only from completed-order activity during the
    later outcome period (inclusive start and end dates).
    """
    target = build_phase9_churn_target(
        customers, orders, outcome_start=outcome_start, outcome_end=outcome_end
    )
    purchase = build_historical_purchase_features(
        customers, orders, order_items, prediction_cutoff=prediction_cutoff
    )
    profile = build_customer_profile_features(
        customers, prediction_cutoff=prediction_cutoff
    )
    eligible_customers = profile[["customer_id"]].copy()
    support = build_historical_support_features(
        support_tickets, eligible_customers, prediction_cutoff=prediction_cutoff
    )

    features = purchase.loc[
        purchase["customer_id"].isin(eligible_customers["customer_id"])
    ].merge(profile, on="customer_id", how="inner")
    features = features.merge(support, on="customer_id", how="inner")
    return features.merge(
        target[["customer_id", "churned"]], on="customer_id", how="left", validate="one_to_one"
    )


def split_modelling_features(
    modelling_data: pd.DataFrame,
    target_col: str = "churned",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictor matrix X and target y as in Phase 9."""
    _require_columns(modelling_data, [target_col, "customer_id"], "modelling_data")
    excluded = [column for column in ["customer_id", "last_purchase_date", target_col]
                if column in modelling_data.columns]
    X = modelling_data.drop(columns=excluded).copy()
    y = modelling_data[target_col].copy()
    return X, y
