from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class RiskExplanationEngine:
    """Generates feature-impact explanations for classifier predictions."""

    def __init__(
        self,
        trained_pipeline: Pipeline,
        feature_names: List[str],
        risk_map: Dict[int, str] | None = None,
    ) -> None:
        self.pipeline = trained_pipeline
        self.feature_names = feature_names
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["model"]
        self.risk_map = risk_map or {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

    def explain_prediction(
        self,
        user_df: pd.DataFrame,
        predicted_class_label: int | None = None,
        top_n: int = 3,
    ) -> dict:
        if len(user_df) != 1:
            raise ValueError("user_df must contain exactly one row.")

        transformed = self.preprocessor.transform(user_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed).reshape(1, -1)

        if predicted_class_label is None:
            predicted_class_label = int(self.pipeline.predict(user_df)[0])

        feature_names = self._feature_names(transformed.shape[1])
        contributions = self._feature_contributions(transformed)

        feature_importance: List[Tuple[str, float]] = []
        for idx, feature in enumerate(feature_names):
            feature_importance.append((feature, float(contributions[idx])))

        selected_features = self._select_top_features(feature_importance, top_n=top_n)
        explanation_text = self._generate_text(selected_features, predicted_class_label)

        return {
            "predicted_class": predicted_class_label,
            "predicted_risk_band": self.risk_map.get(predicted_class_label, str(predicted_class_label)),
            "top_features_impact": [
                {
                    "feature": feature,
                    "impact_score": round(score, 4),
                    "direction": "increase" if score > 0 else "decrease",
                }
                for feature, score in selected_features
            ],
            "natural_language_explanation": explanation_text,
            "override_applied": False,
        }

    def _select_top_features(self, feature_importance: List[Tuple[str, float]], top_n: int) -> List[Tuple[str, float]]:
        if top_n <= 0:
            return []
        return sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)[:top_n]

    def _generate_text(self, top_features: List[Tuple[str, float]], predicted_class_label: int) -> str:
        if not top_features:
            return "No feature contribution data is available for explanation."

        parts = []
        for feature, score in top_features:
            clean_name = feature.replace("_", " ").title()
            risk_direction = "higher risk" if score > 0 else "lower risk"
            parts.append(f"Current {clean_name} is associated with {risk_direction} (impact {score:.4f})")

        risk_name = self.risk_map.get(predicted_class_label, str(predicted_class_label))
        return f"Original ML prediction was {risk_name}. Top risk drivers: " + "; ".join(parts) + "."

    def _feature_names(self, transformed_width: int) -> List[str]:
        if len(self.feature_names) == transformed_width:
            return self.feature_names
        try:
            transformed_names = list(self.preprocessor.get_feature_names_out())
        except Exception:
            transformed_names = [f"feature_{idx}" for idx in range(transformed_width)]
        return [name.split("__", 1)[-1] for name in transformed_names]

    def _feature_contributions(self, transformed: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "coef_"):
            risk_coefficients = self._risk_direction_coefficients()
            return np.ravel(transformed * risk_coefficients)

        if hasattr(self.model, "predict_proba"):
            return self._risk_probability_contributions(transformed)

        if hasattr(self.model, "feature_importances_"):
            return np.ravel(transformed * self.model.feature_importances_)

        return np.zeros(transformed.shape[1], dtype=float)

    def _risk_direction_coefficients(self) -> np.ndarray:
        model_classes = list(self.model.classes_)
        low_class = min(model_classes)
        high_class = max(model_classes)
        low_position = model_classes.index(low_class)
        high_position = model_classes.index(high_class)
        return self.model.coef_[high_position] - self.model.coef_[low_position]

    def _risk_probability_contributions(self, transformed: np.ndarray) -> np.ndarray:
        baseline_risk_score = self._expected_risk_score(transformed)
        contributions = []
        for idx in range(transformed.shape[1]):
            neutralized = transformed.copy()
            neutralized[0, idx] = 0.0
            contributions.append(baseline_risk_score - self._expected_risk_score(neutralized))
        return np.asarray(contributions, dtype=float)

    def _expected_risk_score(self, transformed: np.ndarray) -> float:
        class_probabilities = np.ravel(self.model.predict_proba(transformed)[0])
        class_values = np.asarray(self.model.classes_, dtype=float)
        return float(np.dot(class_probabilities, class_values))

