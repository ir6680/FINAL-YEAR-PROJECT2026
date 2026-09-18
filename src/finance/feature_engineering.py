from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


EXPENSE_COMPONENT_COLUMNS = [
    "rent_or_mortgage",
    "essential_spending",
    "discretionary_spending",
]

DERIVED_FEATURE_COLUMNS = [
    "monthly_expense_total",
    "surplus",
    "expense_ratio",
    "debt_pressure",
]


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required financial columns: {missing}")


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")


def compute_financial_metrics(
    monthly_income: object,
    rent_or_mortgage: object,
    essential_spending: object,
    discretionary_spending: object,
    loan_payment: object = 0.0,
) -> dict[str, Any]:
    """
    Compute the canonical financial metrics used by ML, UI, clustering, and reports.

    expense_ratio_model is the numeric value used by ML/guardrails. For zero
    income it uses 999.0 as a high-risk sentinel, while expense_ratio_display is
    None so UI/PDF can show "N/A (No Income)" instead of a misleading 0%.
    """
    income = _safe_float(monthly_income)
    rent = _safe_float(rent_or_mortgage)
    essential = _safe_float(essential_spending)
    discretionary = _safe_float(discretionary_spending)
    loan = _safe_float(loan_payment)

    _validate_non_negative("monthly_income", income)
    _validate_non_negative("rent_or_mortgage", rent)
    _validate_non_negative("essential_spending", essential)
    _validate_non_negative("discretionary_spending", discretionary)
    _validate_non_negative("loan_payment", loan)

    total_expense = rent + essential + discretionary
    surplus = income - total_expense

    if income > 0:
        expense_ratio_model = (total_expense / income) * 100.0
        expense_ratio_display = expense_ratio_model
        debt_pressure_model = (loan / income) * 100.0
        debt_pressure_display = debt_pressure_model
    else:
        expense_ratio_model = 999.0
        expense_ratio_display = None
        debt_pressure_model = 999.0 if loan > 0 else 0.0
        debt_pressure_display = None

    return {
        "total_expense": total_expense,
        "surplus": surplus,
        "expense_ratio": expense_ratio_model,
        "expense_ratio_model": expense_ratio_model,
        "expense_ratio_display": expense_ratio_display,
        "debt_pressure": debt_pressure_model,
        "debt_pressure_model": debt_pressure_model,
        "debt_pressure_display": debt_pressure_display,
    }


def correct_financial_features(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """
    Recompute expense totals and derived financial features from atomic columns.

    The previous dataset version had an inconsistent monthly_expense_total column.
    This function makes rent, essential spending, and discretionary spending the
    source of truth.
    """
    _require_columns(df, ["monthly_income", *EXPENSE_COMPONENT_COLUMNS])
    corrected = df.copy() if copy else df

    numeric_columns = ["monthly_income", *EXPENSE_COMPONENT_COLUMNS]
    if "loan_payment" in corrected.columns:
        numeric_columns.append("loan_payment")

    for column in numeric_columns:
        corrected[column] = pd.to_numeric(corrected[column], errors="coerce")

    metrics = corrected.apply(
        lambda row: compute_financial_metrics(
            monthly_income=row["monthly_income"],
            rent_or_mortgage=row["rent_or_mortgage"],
            essential_spending=row["essential_spending"],
            discretionary_spending=row["discretionary_spending"],
            loan_payment=row["loan_payment"] if "loan_payment" in corrected.columns else 0.0,
        ),
        axis=1,
        result_type="expand",
    )
    corrected["monthly_expense_total"] = metrics["total_expense"]
    corrected["surplus"] = metrics["surplus"]
    corrected["expense_ratio"] = metrics["expense_ratio"]
    corrected["debt_pressure"] = metrics["debt_pressure"]

    return corrected


def audit_expense_total_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """Return row-level mismatch diagnostics before correction is applied."""
    _require_columns(df, ["monthly_expense_total", *EXPENSE_COMPONENT_COLUMNS])
    audit_df = df.copy()
    component_total = audit_df[EXPENSE_COMPONENT_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    ).sum(axis=1, min_count=len(EXPENSE_COMPONENT_COLUMNS))
    audit_df["component_expense_total"] = component_total
    audit_df["expense_total_delta"] = (
        component_total - pd.to_numeric(audit_df["monthly_expense_total"], errors="coerce")
    )
    return audit_df
