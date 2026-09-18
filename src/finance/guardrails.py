from __future__ import annotations

from src.finance.risk_policy import apply_risk_to_recommendation, risk_class_to_band


def apply_financial_guardrails(explanation_result: dict, recommendation_result: dict) -> tuple[dict, dict]:
    """Enforce conservative policy without ever reducing the model risk class."""
    metrics = recommendation_result.get("user_metrics", {})
    surplus = float(metrics.get("calculated_surplus", 0.0))
    expense_ratio = float(metrics.get("expense_ratio_percent", 0.0))
    original_class = int(explanation_result.get("predicted_class", 1))

    target_class = original_class

    if surplus < 0 or expense_ratio >= 90:
        target_class = max(target_class, 2)
    elif expense_ratio >= 60:
        target_class = max(target_class, 1)

    final_risk_band = risk_class_to_band(target_class)
    original_risk_band = risk_class_to_band(original_class)
    override_applied = target_class != original_class

    explanation_result["original_predicted_class"] = original_class
    explanation_result["original_risk_band"] = original_risk_band
    explanation_result["ml_predicted_class"] = original_class
    explanation_result["ml_risk_band"] = original_risk_band
    explanation_result["guardrail_predicted_class"] = target_class
    explanation_result["guardrail_risk_band"] = final_risk_band
    explanation_result["final_predicted_class"] = target_class
    explanation_result["final_risk_band"] = final_risk_band
    explanation_result["predicted_class"] = target_class
    explanation_result["predicted_risk_band"] = final_risk_band
    explanation_result["override_applied"] = override_applied

    apply_risk_to_recommendation(recommendation_result, target_class)

    return explanation_result, recommendation_result
