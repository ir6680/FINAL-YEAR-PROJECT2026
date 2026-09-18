from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.finance.feature_engineering import compute_financial_metrics
from src.finance.risk_policy import RISK_ACTIONS, RISK_BANDS, portfolio_for_final_risk, risk_class_to_band


GLOBAL_RISK_MAP = RISK_BANDS


class FinancialRecommendationEngine:
    """Creates actionable financial and crypto guidance per risk level."""

    def __init__(self) -> None:
        self.risk_mapping = GLOBAL_RISK_MAP
        self.advice_matrix = RISK_ACTIONS

    def generate_recommendations(
        self,
        predicted_class_label: int,
        user_features: Optional[Dict[str, Any] | list[Dict[str, Any]]] = None,
        cluster_context: Optional[Dict[str, Any]] = None,
        model_confidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        risk_level = self.risk_mapping.get(predicted_class_label, "Medium Risk")
        base_advice = self.advice_matrix[risk_level]

        recommendation_output: Dict[str, Any] = {}

        if user_features:
            if isinstance(user_features, list):
                user_features = user_features[0] if user_features else {}

            def safe_float(value: Any, default: float = 0.0) -> float:
                if value is None or pd.isna(value):
                    return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            income = safe_float(user_features.get("monthly_income", 0))
            credit_score = safe_float(user_features.get("credit_score", 0))
            metrics = compute_financial_metrics(
                monthly_income=income,
                rent_or_mortgage=user_features.get("rent_or_mortgage", 0),
                essential_spending=user_features.get("essential_spending", 0),
                discretionary_spending=user_features.get("discretionary_spending", 0),
                loan_payment=user_features.get("loan_payment", 0),
            )
            surplus = metrics["surplus"]
            expense_ratio_model = metrics["expense_ratio_model"]
            expense_ratio_display = metrics["expense_ratio_display"]
            expense_ratio_text = (
                "N/A (No Income)"
                if expense_ratio_display is None
                else f"{expense_ratio_display:.2f}%"
            )

            recommendation_output["user_metrics"] = {
                "calculated_surplus": round(surplus, 2),
                "expense_ratio_percent": round(expense_ratio_model, 2),
                "expense_ratio_display": (
                    None if expense_ratio_display is None else round(expense_ratio_display, 2)
                ),
                "expense_ratio_display_text": expense_ratio_text,
                "debt_pressure_percent": round(metrics["debt_pressure_model"], 2),
                "debt_pressure_display": (
                    None
                    if metrics["debt_pressure_display"] is None
                    else round(metrics["debt_pressure_display"], 2)
                ),
                "credit_score": int(credit_score),
            }

            if income == 0:
                profile = "No Income / Invalid Data"
            elif surplus < 0:
                profile = "Financial Stress / Deficit"
            elif expense_ratio_model >= 75:
                profile = "High Expense Pressure"
            elif expense_ratio_model >= 60:
                profile = "Moderate Expense Pressure"
            elif surplus < (0.2 * income):
                profile = "Low Surplus"
            else:
                profile = "Healthy Balance"
            recommendation_output["financial_profile"] = profile
            recommendation_output["contextual_actions"] = self._contextual_actions(
                income=income,
                surplus=surplus,
                expense_ratio=expense_ratio_model,
                debt_pressure=metrics["debt_pressure_model"],
                credit_score=credit_score,
            )
        if cluster_context:
            recommendation_output["cluster_context"] = cluster_context
            recommendation_output["cluster_actions"] = self._cluster_actions(cluster_context)
        if model_confidence:
            recommendation_output["model_confidence"] = model_confidence

        recommendation_output.update(
            {
                "risk_assessment": risk_level,
                "core_priority": base_advice["core_priority"],
                "action_plan": {
                    "spending_advice": base_advice["spending_advice"],
                    "saving_strategies": base_advice["saving_strategies"],
                    "investment_and_crypto": base_advice["investment_and_crypto"],
                },
            }
        )

        return recommendation_output

    def apply_decision_context(
        self,
        recommendation_output: Dict[str, Any],
        final_risk_class: int,
        cluster_context: Optional[Dict[str, Any]] = None,
        model_confidence: Optional[Dict[str, Any]] = None,
        drift_detection: Optional[Dict[str, Any]] = None,
        scenario_simulation: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        final_risk_band = risk_class_to_band(final_risk_class)
        recommendation_output["risk_assessment"] = final_risk_band
        recommendation_output["portfolio_allocation"] = portfolio_for_final_risk(final_risk_band)
        recommendation_output["decision_evidence"] = {
            "final_risk_class": int(final_risk_class),
            "final_risk_band": final_risk_band,
            "cluster_context_used": cluster_context is not None,
            "confidence_used": model_confidence is not None,
            "drift_checked": drift_detection is not None,
            "scenario_simulated": scenario_simulation is not None,
        }
        if cluster_context:
            recommendation_output["cluster_context"] = cluster_context
            recommendation_output["cluster_actions"] = self._cluster_actions(cluster_context)
        if model_confidence:
            recommendation_output["model_confidence"] = model_confidence
        if drift_detection:
            recommendation_output["drift_detection"] = drift_detection
            if drift_detection.get("drift_detected"):
                recommendation_output.setdefault("contextual_actions", []).append(
                    "Review unusual input values before trusting the automated recommendation."
                )
        if scenario_simulation:
            recommendation_output["scenario_simulation"] = scenario_simulation
            scenario_action = self._scenario_action(scenario_simulation)
            if scenario_action:
                recommendation_output.setdefault("contextual_actions", []).append(scenario_action)
        return recommendation_output

    def _contextual_actions(
        self,
        income: float,
        surplus: float,
        expense_ratio: float,
        debt_pressure: float,
        credit_score: float,
    ) -> list[str]:
        actions: list[str] = []
        if income <= 0:
            actions.append("Verify income data and keep recommendations fully defensive until income is positive.")
        if surplus < 0:
            actions.append("Close the monthly deficit before allocating funds to volatile DeFi assets.")
        elif surplus < 0.2 * income:
            actions.append("Protect at least 20% monthly surplus before increasing crypto exposure.")

        if expense_ratio >= 90:
            actions.append("Reduce housing, essential, or discretionary costs because expenses consume 90%+ of income.")
        elif expense_ratio >= 60:
            actions.append("Track recurring expenses weekly because expenses consume 60%+ of income.")

        if debt_pressure >= 30:
            actions.append("Prioritize debt payments because loan pressure is above 30% of income.")
        elif debt_pressure >= 15:
            actions.append("Avoid new leverage until loan pressure falls below 15% of income.")

        if credit_score and credit_score < 620:
            actions.append("Improve credit health before using DeFi borrowing or leverage.")

        if not actions:
            actions.append("Maintain the current surplus discipline and rebalance only with excess cash flow.")
        return actions

    def _cluster_actions(self, cluster_context: Dict[str, Any]) -> list[str]:
        risk_band = str(cluster_context.get("cluster_risk_band", "Unavailable"))
        if risk_band == "High Risk":
            return [
                "This profile belongs to the highest-pressure spending cluster; prioritize cash-flow repair.",
                "Do not increase crypto exposure until the user leaves the high-risk cluster profile.",
            ]
        if risk_band == "Medium Risk":
            return [
                "This profile is in the transition cluster; focus on lowering recurring expenses.",
                "Use stablecoin-heavy allocations until surplus and expense ratio improve.",
            ]
        if risk_band == "Low Risk":
            return [
                "This profile resembles the healthier surplus cluster; growth allocation can be considered.",
                "Keep emergency reserves intact before moving extra surplus into volatile assets.",
            ]
        return ["Cluster context unavailable; rely on financial ratios and guardrails."]

    def _scenario_action(self, scenario_simulation: list[Dict[str, Any]]) -> str | None:
        if len(scenario_simulation) < 2:
            return None
        baseline = scenario_simulation[0]
        scenario = scenario_simulation[-1]
        if baseline.get("final_risk_band") != scenario.get("final_risk_band"):
            return (
                f"A 10% spending reduction changes final risk from {baseline.get('final_risk_band')} "
                f"to {scenario.get('final_risk_band')}; make this the first action target."
            )
        if float(scenario.get("surplus", 0.0)) > float(baseline.get("surplus", 0.0)):
            return "A 10% spending reduction improves surplus even if the final risk band is unchanged."
        return None

