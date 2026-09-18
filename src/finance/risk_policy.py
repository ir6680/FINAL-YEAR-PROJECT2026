from __future__ import annotations

from copy import deepcopy
from typing import Optional


RISK_BANDS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
}

RISK_TO_PORTFOLIO = {
    "High Risk": {"BTC": 0.0, "ETH": 0.0, "USDC": 100.0},
    "Medium Risk": {"BTC": 20.0, "ETH": 20.0, "USDC": 60.0},
    "Low Risk": {"BTC": 35.0, "ETH": 35.0, "USDC": 30.0},
}

RISK_ACTIONS = {
    "High Risk": {
        "core_priority": "Reduce Debt & Build Buffer",
        "spending_advice": "Implement strict budgeting. Focus on debt reduction and cut discretionary spending.",
        "saving_strategies": "Build a 1-month emergency fund before investing.",
        "investment_and_crypto": "STRICT AVOIDANCE",
    },
    "Medium Risk": {
        "core_priority": "Automate Savings & Moderate Investing",
        "spending_advice": "Control lifestyle inflation and avoid unnecessary debt.",
        "saving_strategies": "Automate savings to reach a 3-6 month emergency fund.",
        "investment_and_crypto": (
            "Capital Preservation. Allocate 20% BTC, 20% ETH, and 60% USDC while the profile "
            "still needs controlled volatility."
        ),
    },
    "Low Risk": {
        "core_priority": "Maximize Yield & Growth",
        "spending_advice": "Maintain surplus and optimize financial planning.",
        "saving_strategies": "Maintain 6-month emergency fund, invest remaining surplus.",
        "investment_and_crypto": (
            "Aggressive Growth. Allocate 35% BTC, 35% ETH, and 30% USDC while maintaining "
            "a stablecoin reserve."
        ),
    },
}


def risk_class_to_band(risk_class: int) -> str:
    return RISK_BANDS.get(int(risk_class), "Medium Risk")


def portfolio_for_final_risk(risk_band: str) -> dict[str, float]:
    return deepcopy(RISK_TO_PORTFOLIO.get(risk_band, RISK_TO_PORTFOLIO["Medium Risk"]))


def apply_risk_to_recommendation(recommendation_result: dict, risk_class: int) -> dict:
    risk_band = risk_class_to_band(risk_class)
    action = RISK_ACTIONS[risk_band]
    recommendation_result["risk_assessment"] = risk_band
    recommendation_result["core_priority"] = action["core_priority"]
    recommendation_result["portfolio_allocation"] = portfolio_for_final_risk(risk_band)
    recommendation_result.setdefault("action_plan", {})
    recommendation_result["action_plan"]["spending_advice"] = action["spending_advice"]
    recommendation_result["action_plan"]["saving_strategies"] = action["saving_strategies"]
    recommendation_result["action_plan"]["investment_and_crypto"] = action["investment_and_crypto"]
    return recommendation_result


def resolve_final_risk_class(
    ml_risk_class: int,
    guardrail_risk_class: int,
    cluster_risk_class: Optional[int],
) -> int:
    candidates = [int(ml_risk_class), int(guardrail_risk_class)]
    if cluster_risk_class is not None:
        candidates.append(int(cluster_risk_class))
    return max(candidates)
