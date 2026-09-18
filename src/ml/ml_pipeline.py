from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    make_scorer,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.finance.feature_engineering import compute_financial_metrics, correct_financial_features
from src.finance.risk_policy import RISK_BANDS, resolve_final_risk_class, risk_class_to_band


RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_CLUSTERS = 3
CV_STRATIFY_BINS = 10
LEAKAGE_ALERT_THRESHOLD = 0.90
DRIFT_Z_THRESHOLD = 3.0
SCENARIO_SPENDING_REDUCTION = 0.10
MIN_RESEARCH_RISK_CLASSES = 3
FINAL_MODEL_NAME = "Multilayer Perceptron"

LEAKY_COLUMNS = [
    "user_id",
    "date",
    "cash_flow_status",
    "financial_advice_score",
    "financial_stress_level",
    "actual_savings",
    "savings_goal_met",
]

EXPENSE_COMPONENT_COLUMNS = [
    "rent_or_mortgage",
    "essential_spending",
    "discretionary_spending",
]

CLUSTER_FEATURE_COLUMNS = [
    "surplus",
    "expense_ratio",
    "debt_pressure",
]

CLASSIFIER_FEATURE_COLUMNS = [
    "monthly_income",
    "credit_score",
    "rent_or_mortgage",
    "essential_spending",
    "discretionary_spending",
]

BLOCKED_CLASSIFIER_COLUMNS = [
    *LEAKY_COLUMNS,
    "monthly_expense_total",
    "surplus",
    "expense_ratio",
    "debt_pressure",
    "cluster_id",
    "cluster",
    "risk_level",
    "risk_band",
]


def load_corrected_data(data_path: Path) -> pd.DataFrame:
    return correct_financial_features(pd.read_csv(data_path))


def prepare_cluster_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    corrected_df = correct_financial_features(raw_df)
    return corrected_df[CLUSTER_FEATURE_COLUMNS].copy()


def prepare_classification_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    corrected_df = correct_financial_features(raw_df)
    return corrected_df[CLASSIFIER_FEATURE_COLUMNS].copy()


def validate_feature_contracts(
    raw_df: pd.DataFrame,
    cluster_features: pd.DataFrame,
    classification_features: pd.DataFrame,
) -> dict[str, Any]:
    required_metric_inputs = {"monthly_income", "loan_payment", *EXPENSE_COMPONENT_COLUMNS}
    missing_metric_inputs = sorted(required_metric_inputs - set(raw_df.columns))
    missing_classifier_inputs = sorted(set(CLASSIFIER_FEATURE_COLUMNS) - set(raw_df.columns))
    if missing_metric_inputs:
        raise ValueError(f"Missing metric inputs: {missing_metric_inputs}")
    if missing_classifier_inputs:
        raise ValueError(f"Missing classifier inputs: {missing_classifier_inputs}")

    if list(cluster_features.columns) != CLUSTER_FEATURE_COLUMNS:
        raise ValueError(
            "Cluster feature order must match the production contract. "
            f"Found: {cluster_features.columns.tolist()}"
        )
    if list(classification_features.columns) != CLASSIFIER_FEATURE_COLUMNS:
        raise ValueError(
            "Classifier feature order must match the production contract. "
            f"Found: {classification_features.columns.tolist()}"
        )

    blocked_overlap = sorted(set(classification_features.columns) & set(BLOCKED_CLASSIFIER_COLUMNS))
    if blocked_overlap:
        raise ValueError(
            "Classifier inputs contain leakage or clustering-derived columns: "
            f"{blocked_overlap}"
        )

    return {
        "cluster_features": CLUSTER_FEATURE_COLUMNS,
        "classifier_features": CLASSIFIER_FEATURE_COLUMNS,
        "blocked_classifier_columns": BLOCKED_CLASSIFIER_COLUMNS,
        "leakage_check_passed": True,
    }


def stable_quantile_bins(series: pd.Series, q: int = CV_STRATIFY_BINS) -> pd.Series:
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=q, labels=False, duplicates="drop")


def build_classifier(features: pd.DataFrame | list[str]) -> Pipeline:
    numeric_cols = list(features.columns) if isinstance(features, pd.DataFrame) else list(features)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_final_estimator()),
        ]
    )


def build_final_estimator() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )


def candidate_estimators() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        FINAL_MODEL_NAME: build_final_estimator(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_candidate_classifier(features: pd.DataFrame | list[str], estimator: Any) -> Pipeline:
    numeric_cols = list(features.columns) if isinstance(features, pd.DataFrame) else list(features)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def fit_clusterer(cluster_features: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> dict[str, Any]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = scaler.fit_transform(imputer.fit_transform(cluster_features[CLUSTER_FEATURE_COLUMNS]))

    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=50)
    labels = model.fit_predict(matrix)
    silhouette = float("nan")
    if len(np.unique(labels)) > 1:
        silhouette = float(silhouette_score(matrix, labels))

    return {
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
        "labels": labels,
        "k": n_clusters,
        "silhouette": silhouette,
        "scaled_matrix": matrix,
    }


def predict_clusters(clusterer: dict[str, Any], cluster_features: pd.DataFrame) -> np.ndarray:
    matrix = clusterer["scaler"].transform(
        clusterer["imputer"].transform(cluster_features[CLUSTER_FEATURE_COLUMNS])
    )
    return clusterer["model"].predict(matrix)


def build_risk_map(
    train_cluster_features: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> tuple[dict[int, int], pd.DataFrame]:
    cluster_profile = train_cluster_features[CLUSTER_FEATURE_COLUMNS].copy()
    cluster_profile["cluster_id"] = cluster_labels
    cluster_profile = cluster_profile.groupby("cluster_id", as_index=True).mean()
    cluster_profile["cluster_size"] = pd.Series(cluster_labels).value_counts().sort_index()

    severity_score = (
        cluster_profile["expense_ratio"].rank(ascending=True)
        + cluster_profile["debt_pressure"].rank(ascending=True)
        + cluster_profile["surplus"].rank(ascending=False)
    )
    ordered_clusters = severity_score.sort_values(ascending=True).index.tolist()
    risk_map = {int(cluster_id): int(risk_level) for risk_level, cluster_id in enumerate(ordered_clusters)}

    cluster_profile = cluster_profile.reset_index()
    cluster_profile["risk_level"] = cluster_profile["cluster_id"].map(risk_map)
    cluster_profile["risk_band"] = cluster_profile["risk_level"].map(RISK_BANDS)
    cluster_profile["severity_score"] = cluster_profile["cluster_id"].map(severity_score.to_dict())
    return cluster_profile.sort_values("risk_level").reset_index(drop=True).pipe(lambda df: (risk_map, df))


def build_risk_labels(cluster_labels: np.ndarray, risk_map: dict[int, int], split_name: str) -> pd.Series:
    labels = pd.Series(cluster_labels).map(risk_map)
    if labels.isna().any():
        missing = sorted(pd.Series(cluster_labels)[labels.isna()].unique().tolist())
        raise ValueError(f"Missing risk mapping for {split_name}: {missing}")
    return labels.astype(int)


def build_proxy_labels(cluster_features: pd.DataFrame) -> tuple[pd.Series, dict[str, Any], pd.DataFrame]:
    """Build the project risk proxy labels from full-data clustering artifacts."""
    clusterer = fit_clusterer(cluster_features)
    risk_map, cluster_profile = build_risk_map(cluster_features, clusterer["labels"])
    risk_labels = build_risk_labels(clusterer["labels"], risk_map, "proxy label dataset")
    label_metadata = {
        "label_type": "cluster-derived proxy label",
        "is_ground_truth": False,
        "source_features": CLUSTER_FEATURE_COLUMNS,
        "selected_k": clusterer["k"],
        "cluster_silhouette": clusterer["silhouette"],
        "risk_map": risk_map,
        "model_learning_target": (
            "The classifier learns to approximate a financial-behavior segmentation created from "
            "surplus, expense ratio, and debt pressure. It is not trained on externally observed "
            "default, fraud, insolvency, or investment-loss outcomes."
        ),
    }
    return risk_labels, label_metadata, cluster_profile


def analyze_label_integrity(
    raw_df: pd.DataFrame,
    cluster_features: pd.DataFrame,
    classification_features: pd.DataFrame,
) -> dict[str, Any]:
    """Document semantic leakage risk and formalize the target as proxy labels."""
    risk_labels, label_metadata, _ = build_proxy_labels(cluster_features)
    candidate_frame = correct_financial_features(raw_df).copy()
    candidate_frame["proxy_risk_level"] = risk_labels.values

    numeric_columns = candidate_frame.select_dtypes(include=[np.number]).columns.tolist()
    correlations = (
        candidate_frame[numeric_columns]
        .corrwith(candidate_frame["proxy_risk_level"], method="spearman")
        .drop(labels=["proxy_risk_level"], errors="ignore")
        .abs()
        .sort_values(ascending=False)
    )
    allowed_correlations = correlations.loc[
        [column for column in classification_features.columns if column in correlations.index]
    ]
    blocked_correlations = correlations.loc[
        [column for column in BLOCKED_CLASSIFIER_COLUMNS if column in correlations.index]
    ]

    return {
        **label_metadata,
        "semantic_leakage_risk": (
            "High if clustering-derived variables are used by the classifier; controlled here by "
            "blocking those variables from classifier inputs."
        ),
        "what_model_is_learning": (
            "A supervised surrogate for the unsupervised risk segmentation, using only deployable "
            "raw user-input features."
        ),
        "what_model_is_not_learning": (
            "It is not a causal or externally validated real-world default/investment-risk model."
        ),
        "allowed_feature_max_abs_spearman_to_proxy": float(allowed_correlations.max()),
        "blocked_feature_max_abs_spearman_to_proxy": float(blocked_correlations.max()),
        "top_proxy_correlations": correlations.head(10).round(6).to_dict(),
        "class_distribution": class_distribution_table(risk_labels).to_dict(orient="records"),
    }


def class_distribution_table(labels: pd.Series | np.ndarray) -> pd.DataFrame:
    label_series = pd.Series(labels, name="risk_level").astype(int)
    counts = label_series.value_counts().sort_index()
    table = pd.DataFrame(
        {
            "risk_level": counts.index,
            "risk_band": [RISK_BANDS[int(label)] for label in counts.index],
            "count": counts.values,
            "proportion": counts.values / counts.values.sum(),
        }
    )
    return table.reset_index(drop=True)


def readable_classification_report(y_true: pd.Series, y_pred: np.ndarray) -> str:
    labels = sorted(RISK_BANDS)
    target_names = [RISK_BANDS[label] for label in labels]
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )


def run_single_split(
    cluster_features: pd.DataFrame,
    classification_features: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    classifier: Pipeline | None = None,
) -> dict[str, Any]:
    train_cluster = cluster_features.iloc[train_idx].reset_index(drop=True)
    test_cluster = cluster_features.iloc[test_idx].reset_index(drop=True)
    x_train = classification_features.iloc[train_idx].reset_index(drop=True)
    x_test = classification_features.iloc[test_idx].reset_index(drop=True)

    clusterer = fit_clusterer(train_cluster)
    train_clusters = clusterer["labels"]
    test_clusters = predict_clusters(clusterer, test_cluster)
    risk_map, cluster_profile = build_risk_map(train_cluster, train_clusters)
    y_train = build_risk_labels(train_clusters, risk_map, "training split")
    y_test = build_risk_labels(test_clusters, risk_map, "test split")

    fitted_classifier = classifier or build_classifier(classification_features)
    fitted_classifier.fit(x_train, y_train)
    predictions = fitted_classifier.predict(x_test)

    return {
        "classifier": fitted_classifier,
        "clusterer": clusterer,
        "cluster_profile": cluster_profile,
        "risk_map": risk_map,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "classification_report": readable_classification_report(y_test, predictions),
        "classification_report_dict": classification_report(
            y_test,
            predictions,
            labels=sorted(RISK_BANDS),
            target_names=[RISK_BANDS[label] for label in sorted(RISK_BANDS)],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=sorted(RISK_BANDS)).tolist(),
    }


def evaluate_holdout(cluster_features: pd.DataFrame, classification_features: pd.DataFrame) -> dict[str, Any]:
    stratify_target = stable_quantile_bins(classification_features["monthly_income"])
    train_idx, test_idx = train_test_split(
        np.arange(len(cluster_features)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_target,
    )
    return run_single_split(cluster_features, classification_features, train_idx, test_idx)


def evaluate_outer_cv(cluster_features: pd.DataFrame, classification_features: pd.DataFrame) -> dict[str, Any]:
    pseudo_strata = stable_quantile_bins(classification_features["monthly_income"])
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(classification_features, pseudo_strata), start=1):
        split_metrics = run_single_split(cluster_features, classification_features, train_idx, test_idx)
        rows.append(
            {
                "fold": fold,
                "accuracy": split_metrics["test_accuracy"],
                "macro_f1": split_metrics["test_f1_macro"],
                "silhouette": split_metrics["clusterer"]["silhouette"],
            }
        )

    fold_table = pd.DataFrame(rows)
    return {
        "fold_table": fold_table,
        "cv_accuracy_mean": float(fold_table["accuracy"].mean()),
        "cv_accuracy_std": float(fold_table["accuracy"].std(ddof=0)),
        "cv_accuracy_ci95_low": float(fold_table["accuracy"].mean() - 1.96 * fold_table["accuracy"].std(ddof=1) / np.sqrt(len(fold_table))),
        "cv_accuracy_ci95_high": float(fold_table["accuracy"].mean() + 1.96 * fold_table["accuracy"].std(ddof=1) / np.sqrt(len(fold_table))),
        "cv_f1_macro_mean": float(fold_table["macro_f1"].mean()),
        "cv_f1_macro_std": float(fold_table["macro_f1"].std(ddof=0)),
        "cv_f1_macro_ci95_low": float(fold_table["macro_f1"].mean() - 1.96 * fold_table["macro_f1"].std(ddof=1) / np.sqrt(len(fold_table))),
        "cv_f1_macro_ci95_high": float(fold_table["macro_f1"].mean() + 1.96 * fold_table["macro_f1"].std(ddof=1) / np.sqrt(len(fold_table))),
        "cv_silhouette_mean": float(fold_table["silhouette"].mean()),
    }


def evaluate_model_comparison(
    cluster_features: pd.DataFrame,
    classification_features: pd.DataFrame,
) -> pd.DataFrame:
    comparison_split = evaluate_holdout_split_indices(classification_features)
    train_idx, test_idx = comparison_split["train_idx"], comparison_split["test_idx"]
    base_split = run_single_split(cluster_features, classification_features, train_idx, test_idx)
    rows = []
    for model_name, estimator in candidate_estimators().items():
        candidate = build_candidate_classifier(classification_features, estimator)
        candidate.fit(base_split["x_train"], base_split["y_train"])
        train_predictions = candidate.predict(base_split["x_train"])
        test_predictions = candidate.predict(base_split["x_test"])
        rows.append(
            {
                "model": model_name,
                "train_accuracy": float(accuracy_score(base_split["y_train"], train_predictions)),
                "test_accuracy": float(accuracy_score(base_split["y_test"], test_predictions)),
                "train_macro_f1": float(
                    f1_score(base_split["y_train"], train_predictions, average="macro", zero_division=0)
                ),
                "test_macro_f1": float(
                    f1_score(base_split["y_test"], test_predictions, average="macro", zero_division=0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_macro_f1", "test_accuracy"], ascending=False).reset_index(drop=True)


def classification_metric_tables(holdout_metrics: dict[str, Any]) -> dict[str, pd.DataFrame]:
    report = holdout_metrics["classification_report_dict"]
    per_class_rows = []
    for label in sorted(RISK_BANDS):
        band = RISK_BANDS[label]
        row = report[band]
        per_class_rows.append(
            {
                "risk_level": label,
                "risk_band": band,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1-score"],
                "support": row["support"],
            }
        )

    aggregate_rows = []
    for key in ["macro avg", "weighted avg"]:
        row = report[key]
        aggregate_rows.append(
            {
                "average": key,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1-score"],
                "support": row["support"],
            }
        )

    return {
        "per_class": pd.DataFrame(per_class_rows),
        "aggregate": pd.DataFrame(aggregate_rows),
    }


def model_selection_statistical_summary(model_comparison: pd.DataFrame, cv_metrics: dict[str, Any]) -> dict[str, Any]:
    sorted_models = model_comparison.sort_values(["test_macro_f1", "test_accuracy"], ascending=False).reset_index(drop=True)
    winner = sorted_models.iloc[0]
    runner_up = sorted_models.iloc[1] if len(sorted_models) > 1 else winner
    return {
        "selection_metric": "holdout macro F1, then holdout accuracy",
        "selected_model": str(winner["model"]),
        "runner_up_model": str(runner_up["model"]),
        "macro_f1_margin_vs_runner_up": float(winner["test_macro_f1"] - runner_up["test_macro_f1"]),
        "accuracy_margin_vs_runner_up": float(winner["test_accuracy"] - runner_up["test_accuracy"]),
        "cv_accuracy_ci95": [
            cv_metrics["cv_accuracy_ci95_low"],
            cv_metrics["cv_accuracy_ci95_high"],
        ],
        "cv_macro_f1_ci95": [
            cv_metrics["cv_f1_macro_ci95_low"],
            cv_metrics["cv_f1_macro_ci95_high"],
        ],
        "interpretation": (
            "Model selection is treated as practically justified when the selected model is competitive "
            "on macro F1 and its cross-validation confidence interval indicates stable generalization. "
            "Small margins should be reported transparently rather than overclaimed."
        ),
    }


def evaluate_holdout_split_indices(classification_features: pd.DataFrame) -> dict[str, np.ndarray]:
    stratify_target = stable_quantile_bins(classification_features["monthly_income"])
    train_idx, test_idx = train_test_split(
        np.arange(len(classification_features)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_target,
    )
    return {"train_idx": train_idx, "test_idx": test_idx}


def scaled_cluster_matrix(cluster_features: pd.DataFrame) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = scaler.fit_transform(imputer.fit_transform(cluster_features[CLUSTER_FEATURE_COLUMNS]))
    return matrix, imputer, scaler


def _cluster_quality_row(
    method: str,
    parameter: float | int,
    matrix: np.ndarray,
    labels: np.ndarray,
    inertia: float | None = None,
    bic: float | None = None,
    aic: float | None = None,
    deployable: bool = True,
    notes: str = "",
) -> dict[str, Any]:
    labels = np.asarray(labels)
    non_noise_mask = labels != -1
    non_noise_labels = labels[non_noise_mask]
    n_clusters = len(set(non_noise_labels.tolist()))
    noise_fraction = float((labels == -1).mean()) if len(labels) else 0.0

    silhouette = np.nan
    calinski_harabasz = np.nan
    davies_bouldin = np.nan
    if n_clusters >= 2 and non_noise_mask.sum() > n_clusters:
        eval_matrix = matrix[non_noise_mask]
        silhouette = float(silhouette_score(eval_matrix, non_noise_labels))
        calinski_harabasz = float(calinski_harabasz_score(eval_matrix, non_noise_labels))
        davies_bouldin = float(davies_bouldin_score(eval_matrix, non_noise_labels))

    eligible_for_final = bool(
        deployable
        and n_clusters == MIN_RESEARCH_RISK_CLASSES
        and noise_fraction <= 0.05
        and not np.isnan(silhouette)
    )
    return {
        "method": method,
        "parameter": parameter,
        "n_clusters": n_clusters,
        "noise_fraction": noise_fraction,
        "silhouette": silhouette,
        "calinski_harabasz": calinski_harabasz,
        "davies_bouldin": davies_bouldin,
        "inertia": np.nan if inertia is None else float(inertia),
        "bic": np.nan if bic is None else float(bic),
        "aic": np.nan if aic is None else float(aic),
        "deployable": deployable,
        "eligible_for_final": eligible_for_final,
        "notes": notes,
    }


def evaluate_clustering_options(cluster_features: pd.DataFrame, k_values: range = range(2, 9)) -> pd.DataFrame:
    matrix, _, _ = scaled_cluster_matrix(cluster_features)
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = model.fit_predict(matrix)
        rows.append(
            _cluster_quality_row(
                method="KMeans",
                parameter=k,
                matrix=matrix,
                labels=labels,
                inertia=model.inertia_,
                notes="Deployable via centroid nearest-cluster prediction.",
            )
        )
    return pd.DataFrame(rows)


def evaluate_clustering_candidates(cluster_features: pd.DataFrame) -> pd.DataFrame:
    matrix, _, _ = scaled_cluster_matrix(cluster_features)
    rows = []

    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = model.fit_predict(matrix)
        rows.append(
            _cluster_quality_row(
                "KMeans",
                k,
                matrix,
                labels,
                inertia=model.inertia_,
                notes="Centroid-based, stable, supports out-of-sample prediction.",
            )
        )

    for components in range(2, 7):
        model = GaussianMixture(
            n_components=components,
            covariance_type="full",
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(matrix)
        rows.append(
            _cluster_quality_row(
                "GaussianMixture",
                components,
                matrix,
                labels,
                bic=model.bic(matrix),
                aic=model.aic(matrix),
                notes="Probabilistic soft clustering; supports out-of-sample prediction.",
            )
        )

    for eps in [0.25, 0.35, 0.45, 0.55, 0.75, 1.00]:
        model = DBSCAN(eps=eps, min_samples=15)
        labels = model.fit_predict(matrix)
        rows.append(
            _cluster_quality_row(
                "DBSCAN",
                eps,
                matrix,
                labels,
                deployable=False,
                notes="Density-based; useful diagnostic, but no native predict for new users.",
            )
        )

    comparison = pd.DataFrame(rows)
    return comparison.sort_values(
        ["eligible_for_final", "silhouette", "calinski_harabasz"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def choose_final_clustering_method(clustering_candidates: pd.DataFrame) -> dict[str, Any]:
    eligible = clustering_candidates[clustering_candidates["eligible_for_final"]].copy()
    if eligible.empty:
        raise ValueError("No deployable three-cluster candidate is eligible for final risk labeling.")
    selected = eligible.sort_values(["silhouette", "calinski_harabasz"], ascending=False).iloc[0].to_dict()
    selected["selection_rule"] = (
        "Highest silhouette among deployable candidates with exactly three non-noise clusters. "
        "This preserves the Low/Medium/High risk label contract."
    )
    return selected


def evaluate_cluster_stability(
    cluster_features: pd.DataFrame,
    repeats: int = 20,
    sample_fraction: float = 0.80,
) -> pd.DataFrame:
    full_clusterer = fit_clusterer(cluster_features)
    full_labels = full_clusterer["labels"]
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for repeat in range(1, repeats + 1):
        sample_size = int(len(cluster_features) * sample_fraction)
        sample_idx = np.sort(rng.choice(len(cluster_features), size=sample_size, replace=False))
        sample_clusterer = fit_clusterer(cluster_features.iloc[sample_idx].reset_index(drop=True))
        predicted_full_labels = predict_clusters(sample_clusterer, cluster_features)
        rows.append(
            {
                "repeat": repeat,
                "sample_fraction": sample_fraction,
                "adjusted_rand_index": float(adjusted_rand_score(full_labels, predicted_full_labels)),
            }
        )
    return pd.DataFrame(rows)


def compute_training_curve(
    cluster_features: pd.DataFrame,
    classification_features: pd.DataFrame,
) -> pd.DataFrame:
    clusterer = fit_clusterer(cluster_features)
    risk_map, _ = build_risk_map(cluster_features, clusterer["labels"])
    labels = build_risk_labels(clusterer["labels"], risk_map, "training curve dataset")
    curve_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scorers = {
        "Accuracy": "accuracy",
        "Macro F1": make_scorer(f1_score, average="macro", zero_division=0),
    }

    frames = []
    for metric_name, scorer in scorers.items():
        train_sizes, train_scores, validation_scores = learning_curve(
            estimator=build_classifier(classification_features),
            X=classification_features,
            y=labels,
            cv=curve_cv,
            train_sizes=np.linspace(0.2, 1.0, 5),
            scoring=scorer,
            n_jobs=-1,
        )
        frames.append(
            pd.DataFrame(
                {
                    "metric": metric_name,
                    "train_size": train_sizes,
                    "train_score_mean": train_scores.mean(axis=1),
                    "train_score_std": train_scores.std(axis=1),
                    "validation_score_mean": validation_scores.mean(axis=1),
                    "validation_score_std": validation_scores.std(axis=1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def analyze_learning_behavior(training_curve: pd.DataFrame) -> dict[str, Any]:
    latest = training_curve.sort_values("train_size").groupby("metric").tail(1)
    rows = []
    for _, row in latest.iterrows():
        gap = float(row["train_score_mean"] - row["validation_score_mean"])
        if gap > 0.10:
            diagnosis = "possible overfitting"
        elif row["validation_score_mean"] < 0.70:
            diagnosis = "possible underfitting or weak labels/features"
        else:
            diagnosis = "acceptable generalization gap"
        rows.append(
            {
                "metric": row["metric"],
                "train_score": float(row["train_score_mean"]),
                "validation_score": float(row["validation_score_mean"]),
                "generalization_gap": gap,
                "diagnosis": diagnosis,
            }
        )
    return {
        "latest_points": rows,
        "interpretation": (
            "Learning behavior is judged from the largest training size. A large train-validation "
            "gap indicates overfitting; low train and validation scores indicate underfitting."
        ),
    }


def build_training_feature_stats(classification_features: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column in CLASSIFIER_FEATURE_COLUMNS:
        series = pd.to_numeric(classification_features[column], errors="coerce")
        std = float(series.std(ddof=0))
        stats[column] = {
            "mean": float(series.mean()),
            "std": std if std > 1e-12 else 1.0,
            "min": float(series.min()),
            "max": float(series.max()),
            "p01": float(series.quantile(0.01)),
            "p99": float(series.quantile(0.99)),
        }
    return stats


def detect_input_drift(
    bundle: dict[str, Any],
    classifier_user_df: pd.DataFrame,
    z_threshold: float = DRIFT_Z_THRESHOLD,
) -> dict[str, Any]:
    stats = bundle.get("training_feature_stats", {})
    rows = []
    for column in bundle["classifier_feature_columns"]:
        value = float(classifier_user_df.iloc[0][column])
        column_stats = stats.get(column)
        if not column_stats:
            continue
        z_score = (value - float(column_stats["mean"])) / float(column_stats["std"])
        outside_quantile_range = value < float(column_stats["p01"]) or value > float(column_stats["p99"])
        rows.append(
            {
                "feature": column,
                "value": value,
                "training_mean": float(column_stats["mean"]),
                "z_score": float(z_score),
                "outside_1_99_percentile": bool(outside_quantile_range),
                "flagged": bool(abs(z_score) >= z_threshold or outside_quantile_range),
            }
        )
    flagged = [row for row in rows if row["flagged"]]
    return {
        "drift_detected": bool(flagged),
        "flagged_features": flagged,
        "feature_checks": rows,
        "rule": f"Flag if abs(z_score) >= {z_threshold} or outside training 1st-99th percentile.",
    }


def prediction_confidence(classifier: Pipeline, classifier_user_df: pd.DataFrame) -> dict[str, Any]:
    prediction = int(classifier.predict(classifier_user_df)[0])
    if not hasattr(classifier, "predict_proba"):
        return {
            "predicted_class": prediction,
            "predicted_risk_band": risk_class_to_band(prediction),
            "confidence": None,
            "probabilities": {},
        }
    probabilities = np.ravel(classifier.predict_proba(classifier_user_df)[0])
    classes = [int(label) for label in classifier.classes_]
    probability_map = {
        risk_class_to_band(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }
    return {
        "predicted_class": prediction,
        "predicted_risk_band": risk_class_to_band(prediction),
        "confidence": float(np.max(probabilities)),
        "probabilities": probability_map,
    }


def global_feature_importance(bundle: dict[str, Any]) -> pd.DataFrame:
    model = bundle["classifier"].named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.zeros(len(bundle["classifier_feature_columns"]))
    return pd.DataFrame(
        {
            "feature": bundle["classifier_feature_columns"],
            "importance": values,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)


def infer_cluster_risk_from_user(bundle: dict[str, Any], user_df: pd.DataFrame) -> dict[str, Any]:
    row = user_df.iloc[0]
    metrics = compute_financial_metrics(
        monthly_income=row.get("monthly_income", 0.0),
        rent_or_mortgage=row.get("rent_or_mortgage", 0.0),
        essential_spending=row.get("essential_spending", 0.0),
        discretionary_spending=row.get("discretionary_spending", 0.0),
        loan_payment=row.get("loan_payment", 0.0),
    )
    derived = pd.DataFrame(
        [
            {
                "surplus": metrics["surplus"],
                "expense_ratio": metrics["expense_ratio"],
                "debt_pressure": metrics["debt_pressure"],
            }
        ]
    )
    cluster_x = bundle["cluster_scaler"].transform(
        bundle["cluster_imputer"].transform(derived[bundle["cluster_feature_columns"]])
    )
    cluster_id = int(bundle["cluster_model"].predict(cluster_x)[0])
    cluster_risk_class = int(bundle["risk_map"].get(cluster_id, 1))
    return {
        "cluster_id": cluster_id,
        "cluster_risk_class": cluster_risk_class,
        "cluster_risk_band": risk_class_to_band(cluster_risk_class),
        "metrics": metrics,
    }


def guardrail_risk_from_metrics(metrics: dict[str, Any], ml_risk_class: int) -> int:
    surplus = float(metrics.get("surplus", 0.0))
    expense_ratio = float(metrics.get("expense_ratio", 0.0))
    target_class = int(ml_risk_class)
    if surplus < 0 or expense_ratio >= 90:
        target_class = max(target_class, 2)
    elif expense_ratio >= 60:
        target_class = max(target_class, 1)
    return target_class


def simulate_spending_reduction(
    bundle: dict[str, Any],
    user_df: pd.DataFrame,
    reduction: float = SCENARIO_SPENDING_REDUCTION,
) -> dict[str, Any]:
    if not 0 <= reduction <= 1:
        raise ValueError("reduction must be between 0 and 1.")

    baseline_df = correct_financial_features(user_df.copy())
    scenario_df = baseline_df.copy()
    for column in EXPENSE_COMPONENT_COLUMNS:
        scenario_df[column] = pd.to_numeric(scenario_df[column], errors="coerce") * (1.0 - reduction)
    scenario_df = correct_financial_features(scenario_df, copy=False)

    rows = []
    for name, frame in [("baseline", baseline_df), (f"spending_minus_{int(reduction * 100)}pct", scenario_df)]:
        classifier_user_df = frame[bundle["classifier_feature_columns"]].copy()
        confidence = prediction_confidence(bundle["classifier"], classifier_user_df)
        cluster_context = infer_cluster_risk_from_user(bundle, frame)
        guardrail_risk = guardrail_risk_from_metrics(cluster_context["metrics"], confidence["predicted_class"])
        final_risk = resolve_final_risk_class(
            ml_risk_class=confidence["predicted_class"],
            guardrail_risk_class=guardrail_risk,
            cluster_risk_class=cluster_context["cluster_risk_class"],
        )
        rows.append(
            {
                "scenario": name,
                "monthly_expense_total": float(frame.iloc[0]["monthly_expense_total"]),
                "surplus": float(frame.iloc[0]["surplus"]),
                "expense_ratio": float(frame.iloc[0]["expense_ratio"]),
                "ml_risk_band": confidence["predicted_risk_band"],
                "cluster_risk_band": cluster_context["cluster_risk_band"],
                "guardrail_risk_band": risk_class_to_band(guardrail_risk),
                "final_risk_band": risk_class_to_band(final_risk),
                "confidence": confidence["confidence"],
            }
        )
    scenario_table = pd.DataFrame(rows)
    return {
        "reduction": reduction,
        "scenario_table": scenario_table,
        "interpretation": (
            "Scenario simulation recomputes canonical financial metrics, ML prediction, cluster risk, "
            "guardrail risk, and final risk after the spending change."
        ),
    }


def fit_full_bundle(
    cluster_features: pd.DataFrame,
    classification_features: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clusterer = fit_clusterer(cluster_features)
    full_clusters = clusterer["labels"]
    risk_map, cluster_profile = build_risk_map(cluster_features, full_clusters)
    risk_levels = build_risk_labels(full_clusters, risk_map, "full dataset")

    classifier = build_classifier(classification_features)
    classifier.fit(classification_features, risk_levels)

    bundle = {
        "cluster_feature_columns": CLUSTER_FEATURE_COLUMNS,
        "classifier_feature_columns": CLASSIFIER_FEATURE_COLUMNS,
        "blocked_classifier_columns": BLOCKED_CLASSIFIER_COLUMNS,
        "removed_leaky_columns": LEAKY_COLUMNS,
        "risk_bands": RISK_BANDS,
        "selected_k": clusterer["k"],
        "cluster_silhouette": clusterer["silhouette"],
        "cluster_imputer": clusterer["imputer"],
        "cluster_scaler": clusterer["scaler"],
        "cluster_model": clusterer["model"],
        "risk_map": risk_map,
        "selected_model_name": FINAL_MODEL_NAME,
        "final_classifier": classifier.named_steps["model"].__class__.__name__,
        "training_feature_stats": build_training_feature_stats(classification_features),
        "cluster_profile": cluster_profile.to_dict(orient="records"),
        "classifier": classifier,
    }

    labeled_clustered = cluster_features.copy()
    labeled_clustered["cluster_id"] = full_clusters
    labeled_clustered["risk_level"] = risk_levels.values
    labeled_clustered["risk_band"] = labeled_clustered["risk_level"].map(RISK_BANDS)

    labeled_classification = classification_features.copy()
    labeled_classification["risk_level"] = risk_levels.values
    labeled_classification["risk_band"] = labeled_classification["risk_level"].map(RISK_BANDS)

    return bundle, cluster_profile, labeled_clustered, labeled_classification


def guard_against_unrealistic_performance(holdout_metrics: dict[str, Any], cv_metrics: dict[str, Any]) -> str:
    suspicious_scores = {
        "holdout_accuracy": holdout_metrics["test_accuracy"],
        "holdout_macro_f1": holdout_metrics["test_f1_macro"],
        "cv_accuracy_mean": cv_metrics["cv_accuracy_mean"],
        "cv_f1_macro_mean": cv_metrics["cv_f1_macro_mean"],
    }
    high_score_flag = any(score >= LEAKAGE_ALERT_THRESHOLD for score in suspicious_scores.values())
    holdout_cv_gap = abs(holdout_metrics["test_accuracy"] - cv_metrics["cv_accuracy_mean"])
    if high_score_flag and holdout_cv_gap >= 0.15:
        raise RuntimeError(
            "Suspicious performance pattern detected: near-perfect metrics plus large holdout/CV gap. "
            f"Observed scores: {suspicious_scores}, holdout_cv_gap={holdout_cv_gap:.4f}"
        )
    if high_score_flag:
        return (
            "High score warning: metrics exceed the leakage alert threshold, but holdout/CV stability "
            "does not indicate a split-specific failure. Review feature lineage."
        )
    return "Performance is below the leakage alert threshold and stable across validation folds."


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_outputs(
    root: Path,
    project_root: Path,
    raw_df: pd.DataFrame,
    cluster_features: pd.DataFrame,
    labeled_clustered: pd.DataFrame,
    labeled_classification: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    bundle: dict[str, Any],
    holdout_metrics: dict[str, Any],
    cv_metrics: dict[str, Any],
    model_comparison: pd.DataFrame | None = None,
    clustering_options: pd.DataFrame | None = None,
    clustering_candidates: pd.DataFrame | None = None,
    selected_clustering_method: dict[str, Any] | None = None,
    cluster_stability: pd.DataFrame | None = None,
    training_curve: pd.DataFrame | None = None,
    label_integrity: dict[str, Any] | None = None,
    class_distribution: pd.DataFrame | None = None,
    metric_tables: dict[str, pd.DataFrame] | None = None,
    model_selection_summary: dict[str, Any] | None = None,
    learning_behavior: dict[str, Any] | None = None,
) -> dict[str, Path]:
    artifacts_dir = root / "artifacts"
    metrics_dir = project_root / "artifacts" / "metrics"
    project_models_dir = project_root / "models" / "trained_models"
    project_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    project_models_dir.mkdir(parents=True, exist_ok=True)
    project_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = project_models_dir / "finance_risk_bundle.joblib"
    project_bundle_path = project_models_dir / "finance_risk_bundle.joblib"
    metrics_path = metrics_dir / "finance_risk_metrics.json"
    project_metrics_path = metrics_dir / "finance_risk_metrics.json"
    project_dataset_path = project_data_dir / "personal_finance_tracker_dataset.csv"
    clustered_dataset_path = processed_data_dir / "clustered_dataset_leakage_safe.csv"
    clustered_features_only_path = processed_data_dir / "clustered_dataset_features_only.csv"
    classification_dataset_path = processed_data_dir / "classification_dataset_leakage_safe.csv"
    cluster_profile_path = processed_data_dir / "risk_cluster_profiles.csv"

    joblib.dump(bundle, bundle_path)
    joblib.dump(bundle, project_bundle_path)

    raw_df.to_csv(project_dataset_path, index=False)
    clustered_export = raw_df.drop(columns=LEAKY_COLUMNS, errors="ignore").copy()
    clustered_export[CLUSTER_FEATURE_COLUMNS] = cluster_features[CLUSTER_FEATURE_COLUMNS]
    clustered_export["cluster_id"] = labeled_clustered["cluster_id"].values
    clustered_export["risk_level"] = labeled_clustered["risk_level"].values
    clustered_export["risk_band"] = labeled_clustered["risk_band"].values
    clustered_export.to_csv(clustered_dataset_path, index=False)
    clustered_export.drop(columns=["risk_level", "risk_band"], errors="ignore").to_csv(
        clustered_features_only_path,
        index=False,
    )
    labeled_classification.to_csv(classification_dataset_path, index=False)
    cluster_profile.to_csv(cluster_profile_path, index=False)

    metrics = {
        "dataset": "personal_finance_tracker_dataset.csv",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_pandas_version": pd.__version__,
        "python_numpy_version": np.__version__,
        "python_sklearn_version": sklearn.__version__,
        "removed_leaky_columns": LEAKY_COLUMNS,
        "blocked_classifier_columns": BLOCKED_CLASSIFIER_COLUMNS,
        "cluster_features": CLUSTER_FEATURE_COLUMNS,
        "classification_features": CLASSIFIER_FEATURE_COLUMNS,
        "selected_model_name": bundle["selected_model_name"],
        "final_classifier": bundle["final_classifier"],
        "selected_k": bundle["selected_k"],
        "cluster_silhouette": bundle["cluster_silhouette"],
        "holdout_test_accuracy": holdout_metrics["test_accuracy"],
        "holdout_test_f1_macro": holdout_metrics["test_f1_macro"],
        "holdout_confusion_matrix": holdout_metrics["confusion_matrix"],
        "holdout_classification_report": holdout_metrics["classification_report"],
        "cv_accuracy_mean": cv_metrics["cv_accuracy_mean"],
        "cv_accuracy_std": cv_metrics["cv_accuracy_std"],
        "cv_accuracy_ci95_low": cv_metrics["cv_accuracy_ci95_low"],
        "cv_accuracy_ci95_high": cv_metrics["cv_accuracy_ci95_high"],
        "cv_f1_macro_mean": cv_metrics["cv_f1_macro_mean"],
        "cv_f1_macro_std": cv_metrics["cv_f1_macro_std"],
        "cv_f1_macro_ci95_low": cv_metrics["cv_f1_macro_ci95_low"],
        "cv_f1_macro_ci95_high": cv_metrics["cv_f1_macro_ci95_high"],
        "cv_silhouette_mean": cv_metrics["cv_silhouette_mean"],
        "cv_folds": cv_metrics["fold_table"].round(6).to_dict(orient="records"),
        "leakage_alert_threshold": LEAKAGE_ALERT_THRESHOLD,
        "performance_audit_note": guard_against_unrealistic_performance(holdout_metrics, cv_metrics),
        "saved_bundle": str(bundle_path.relative_to(root)),
        "project_bundle_path": str(project_bundle_path.relative_to(root)),
        "project_metrics_path": str(project_metrics_path.relative_to(root)),
        "project_dataset_path": str(project_dataset_path.relative_to(root)),
        "clustered_dataset_path": clustered_dataset_path.name,
        "clustered_features_only_path": clustered_features_only_path.name,
        "classification_dataset_path": classification_dataset_path.name,
        "cluster_profile_path": cluster_profile_path.name,
    }
    if model_comparison is not None:
        metrics["model_comparison"] = model_comparison.round(6).to_dict(orient="records")
    if clustering_options is not None:
        metrics["clustering_options"] = clustering_options.round(6).to_dict(orient="records")
    if clustering_candidates is not None:
        metrics["clustering_candidates"] = clustering_candidates.round(6).to_dict(orient="records")
    if selected_clustering_method is not None:
        metrics["selected_clustering_method"] = selected_clustering_method
    if cluster_stability is not None:
        metrics["cluster_stability"] = cluster_stability.round(6).to_dict(orient="records")
        metrics["cluster_stability_ari_mean"] = float(cluster_stability["adjusted_rand_index"].mean())
        metrics["cluster_stability_ari_std"] = float(cluster_stability["adjusted_rand_index"].std(ddof=0))
    if training_curve is not None:
        metrics["final_model_training_curve"] = training_curve.round(6).to_dict(orient="records")
    if label_integrity is not None:
        metrics["label_integrity"] = label_integrity
    if class_distribution is not None:
        metrics["class_distribution"] = class_distribution.round(6).to_dict(orient="records")
    if metric_tables is not None:
        metrics["per_class_metrics"] = metric_tables["per_class"].round(6).to_dict(orient="records")
        metrics["macro_vs_weighted_metrics"] = metric_tables["aggregate"].round(6).to_dict(orient="records")
    if model_selection_summary is not None:
        metrics["model_selection_summary"] = model_selection_summary
    if learning_behavior is not None:
        metrics["learning_behavior"] = learning_behavior

    metrics_json = json.dumps(json_safe(metrics), indent=2, default=str, allow_nan=False)
    metrics_path.write_text(metrics_json, encoding="utf-8")
    project_metrics_path.write_text(metrics_json, encoding="utf-8")

    return {
        "bundle_path": bundle_path,
        "project_bundle_path": project_bundle_path,
        "metrics_path": metrics_path,
        "project_metrics_path": project_metrics_path,
        "project_dataset_path": project_dataset_path,
        "clustered_dataset_path": clustered_dataset_path,
        "clustered_features_only_path": clustered_features_only_path,
        "classification_dataset_path": classification_dataset_path,
        "cluster_profile_path": cluster_profile_path,
    }


def run_training_pipeline(data_path: Path, root: Path, project_root: Path) -> dict[str, Any]:
    raw_df = load_corrected_data(data_path)
    cluster_features = prepare_cluster_features(raw_df)
    classification_features = prepare_classification_features(raw_df)
    feature_audit = validate_feature_contracts(raw_df, cluster_features, classification_features)
    label_integrity = analyze_label_integrity(raw_df, cluster_features, classification_features)
    proxy_labels, _, _ = build_proxy_labels(cluster_features)
    class_distribution = class_distribution_table(proxy_labels)

    clustering_options = evaluate_clustering_options(cluster_features)
    clustering_candidates = evaluate_clustering_candidates(cluster_features)
    selected_clustering_method = choose_final_clustering_method(clustering_candidates)
    cluster_stability = evaluate_cluster_stability(cluster_features)
    holdout_metrics = evaluate_holdout(cluster_features, classification_features)
    metric_tables = classification_metric_tables(holdout_metrics)
    cv_metrics = evaluate_outer_cv(cluster_features, classification_features)
    model_comparison = evaluate_model_comparison(cluster_features, classification_features)
    model_selection_summary = model_selection_statistical_summary(model_comparison, cv_metrics)
    training_curve = compute_training_curve(cluster_features, classification_features)
    learning_behavior = analyze_learning_behavior(training_curve)
    bundle, cluster_profile, labeled_clustered, labeled_classification = fit_full_bundle(
        cluster_features,
        classification_features,
    )
    bundle["label_integrity"] = label_integrity
    bundle["selected_clustering_method"] = selected_clustering_method
    output_paths = save_outputs(
        root=root,
        project_root=project_root,
        raw_df=raw_df,
        cluster_features=cluster_features,
        labeled_clustered=labeled_clustered,
        labeled_classification=labeled_classification,
        cluster_profile=cluster_profile,
        bundle=bundle,
        holdout_metrics=holdout_metrics,
        cv_metrics=cv_metrics,
        model_comparison=model_comparison,
        clustering_options=clustering_options,
        clustering_candidates=clustering_candidates,
        selected_clustering_method=selected_clustering_method,
        cluster_stability=cluster_stability,
        training_curve=training_curve,
        label_integrity=label_integrity,
        class_distribution=class_distribution,
        metric_tables=metric_tables,
        model_selection_summary=model_selection_summary,
        learning_behavior=learning_behavior,
    )
    return {
        "raw_df": raw_df,
        "cluster_features": cluster_features,
        "classification_features": classification_features,
        "feature_audit": feature_audit,
        "label_integrity": label_integrity,
        "class_distribution": class_distribution,
        "clustering_options": clustering_options,
        "clustering_candidates": clustering_candidates,
        "selected_clustering_method": selected_clustering_method,
        "cluster_stability": cluster_stability,
        "holdout_metrics": holdout_metrics,
        "metric_tables": metric_tables,
        "cv_metrics": cv_metrics,
        "model_comparison": model_comparison,
        "model_selection_summary": model_selection_summary,
        "training_curve": training_curve,
        "learning_behavior": learning_behavior,
        "bundle": bundle,
        "cluster_profile": cluster_profile,
        "labeled_clustered": labeled_clustered,
        "labeled_classification": labeled_classification,
        "output_paths": output_paths,
    }
