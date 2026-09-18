from __future__ import annotations

from typing import Dict

import pandas as pd


def _max_drawdown(series: pd.Series) -> float:
    running_max = series.cummax()
    drawdown = (series / running_max) - 1.0
    return float(drawdown.min())


def simulate_12m_portfolio(allocation: Dict[str, float]) -> dict:
    """
    Simulate 12-month performance for BTC/ETH/USDC allocation.
    Allocation values are expected in percentages (0-100).
    """
    months = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12, freq="ME")

    # Deterministic monthly returns to keep simulation stable/reproducible.
    returns = pd.DataFrame(
        {
            "BTC": [0.06, -0.04, 0.09, 0.03, -0.02, 0.07, 0.02, -0.05, 0.08, 0.04, -0.01, 0.05],
            "ETH": [0.07, -0.05, 0.1, 0.02, -0.03, 0.08, 0.03, -0.06, 0.09, 0.05, -0.02, 0.06],
            "USDC": [0.002] * 12,
        },
        index=months,
    )

    weights = {
        "BTC": float(allocation.get("BTC", 0.0)) / 100.0,
        "ETH": float(allocation.get("ETH", 0.0)) / 100.0,
        "USDC": float(allocation.get("USDC", 0.0)) / 100.0,
    }

    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = {"BTC": 0.0, "ETH": 0.0, "USDC": 1.0}
    elif abs(total_weight - 1.0) > 1e-9:
        weights = {k: v / total_weight for k, v in weights.items()}

    portfolio_monthly = (
        returns["BTC"] * weights["BTC"]
        + returns["ETH"] * weights["ETH"]
        + returns["USDC"] * weights["USDC"]
    )
    baseline_monthly = returns["USDC"]

    growth = pd.DataFrame(index=months)
    growth["Recommended Portfolio"] = 1000.0 * (1.0 + portfolio_monthly).cumprod()
    growth["USDC Baseline"] = 1000.0 * (1.0 + baseline_monthly).cumprod()

    roi = (growth["Recommended Portfolio"].iloc[-1] / growth["Recommended Portfolio"].iloc[0] - 1.0) * 100
    mdd = _max_drawdown(growth["Recommended Portfolio"]) * 100

    return {
        "timeseries": growth,
        "estimated_annual_roi_percent": round(float(roi), 2),
        "max_drawdown_percent": round(float(mdd), 2),
    }

