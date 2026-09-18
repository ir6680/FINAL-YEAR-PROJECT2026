"""Generate publication-quality confusion matrices for finalnb.ipynb models.

This script mirrors the holdout evaluation logic in notebooks/finalnb.ipynb:
the same dataset candidates, feature engineering, train/test split,
KMeans-derived risk labels, candidate estimators, label order, and random
state are used. It does not change model selection or saved metrics.
"""

from __future__ import annotations

import csv
import json
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DPI = 600
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CLUSTERS = 3
CV_STRATIFY_BINS = 10

EXPENSE_COMPONENT_COLUMNS = ["rent_or_mortgage", "essential_spending", "discretionary_spending"]
CLUSTER_FEATURES = ["surplus", "expense_ratio", "debt_pressure"]
CLASSIFICATION_FEATURES = [
    "monthly_income",
    "credit_score",
    "rent_or_mortgage",
    "essential_spending",
    "discretionary_spending",
]
RISK_BANDS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
ORDERED_LABELS = sorted(RISK_BANDS)
ORDERED_NAMES = [RISK_BANDS[i] for i in ORDERED_LABELS]


def project_root() -> Path:
    start = Path(__file__).resolve()
    return next((p for p in [start.parent, *start.parents] if (p / "data").exists()), Path.cwd())


ROOT = project_root()
METRICS_DIR = ROOT / "results" / "metrics"
MODEL_COMPARISON_DIR = ROOT / "results" / "model_comparisons"
IMAGE_DIR = ROOT / "images" / "confusion_matrices"
NOTEBOOK_PATH = ROOT / "notebooks" / "main_research_notebook.ipynb"

DATASET_CANDIDATES = [
    ROOT / "data" / "raw" / "personal_finance_tracker_dataset.csv",
    ROOT / "data" / "sample_inputs" / "personal_finance_tracker_dataset.csv",
    ROOT / "data" / "personal_finance_tracker_dataset.csv",
]


def relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_raw_data() -> pd.DataFrame:
    dataset_path = next((p for p in DATASET_CANDIDATES if p.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Dataset CSV not found in the notebook's expected locations.")
    return pd.read_csv(dataset_path)


def correct_financial_features(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    out = df.copy() if copy else df
    required = ["monthly_income", "loan_payment", *EXPENSE_COMPONENT_COLUMNS]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out["monthly_expense_total"] = out[EXPENSE_COMPONENT_COLUMNS].sum(axis=1)
    out["surplus"] = out["monthly_income"] - out["monthly_expense_total"]

    safe_income = out["monthly_income"].replace(0, np.nan)
    out["expense_ratio"] = (out["monthly_expense_total"] / safe_income) * 100
    out["debt_pressure"] = (out["loan_payment"] / safe_income) * 100
    out[["expense_ratio", "debt_pressure"]] = (
        out[["expense_ratio", "debt_pressure"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(999)
    )
    return out


def stable_quantile_bins(series: pd.Series, q: int = CV_STRATIFY_BINS) -> pd.Series:
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=q, labels=False, duplicates="drop")


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = features.columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        ]
    )


def build_model_pipeline(features: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor(features)), ("model", estimator)])


def get_candidate_estimators() -> dict:
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
        "Multilayer Perceptron": MLPClassifier(
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
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def fit_clusterer(cluster_frame: pd.DataFrame) -> dict:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = imputer.fit_transform(cluster_frame[CLUSTER_FEATURES])
    matrix = scaler.fit_transform(matrix)

    model = KMeans(n_clusters=N_CLUSTERS, n_init=50, random_state=RANDOM_STATE)
    labels = model.fit_predict(matrix)
    return {"imputer": imputer, "scaler": scaler, "model": model, "labels": labels, "k": N_CLUSTERS}


def predict_clusters(clusterer: dict, cluster_frame: pd.DataFrame) -> np.ndarray:
    matrix = clusterer["imputer"].transform(cluster_frame[CLUSTER_FEATURES])
    matrix = clusterer["scaler"].transform(matrix)
    return clusterer["model"].predict(matrix)


def build_risk_map(train_cluster_features: pd.DataFrame, cluster_labels: np.ndarray) -> tuple[dict[int, int], pd.DataFrame]:
    profile = train_cluster_features.copy()
    profile["cluster_id"] = cluster_labels
    profile = profile.groupby("cluster_id", as_index=True).mean()
    profile["cluster_size"] = pd.Series(cluster_labels).value_counts().sort_index()

    surplus_component = -profile["surplus"].rank(method="dense", ascending=True)
    profile["severity_score"] = profile["expense_ratio"] + profile["debt_pressure"] + surplus_component

    ordered_clusters = profile["severity_score"].sort_values().index.tolist()
    risk_map = {int(cluster_id): int(risk_level) for risk_level, cluster_id in enumerate(ordered_clusters)}
    profile["risk_level"] = profile.index.map(risk_map)
    profile["risk_band"] = profile["risk_level"].map(RISK_BANDS)
    return risk_map, profile.sort_values("risk_level")


def build_risk_labels(cluster_labels: np.ndarray, risk_map: dict[int, int], split_name: str = "dataset") -> pd.Series:
    missing = sorted(set(map(int, np.unique(cluster_labels))) - set(risk_map.keys()))
    if missing:
        raise ValueError(f"Missing risk mapping for {split_name}: {missing}")
    return pd.Series(cluster_labels).map(risk_map).astype(int)


def build_holdout_dataset(raw_df: pd.DataFrame) -> dict:
    cluster_features = raw_df[CLUSTER_FEATURES].copy()
    classification_features = raw_df[CLASSIFICATION_FEATURES].copy()
    stratify_target = stable_quantile_bins(classification_features["monthly_income"])
    train_idx, test_idx = train_test_split(
        np.arange(len(classification_features)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_target,
    )

    x_train = classification_features.iloc[train_idx].reset_index(drop=True)
    x_test = classification_features.iloc[test_idx].reset_index(drop=True)
    train_cluster = cluster_features.iloc[train_idx].reset_index(drop=True)
    test_cluster = cluster_features.iloc[test_idx].reset_index(drop=True)

    clusterer = fit_clusterer(train_cluster)
    train_clusters = clusterer["labels"]
    test_clusters = predict_clusters(clusterer, test_cluster)
    risk_map, cluster_profile = build_risk_map(train_cluster, train_clusters)
    y_train = build_risk_labels(train_clusters, risk_map, split_name="holdout train")
    y_test = build_risk_labels(test_clusters, risk_map, split_name="holdout test")

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "risk_map": risk_map,
        "cluster_profile": cluster_profile,
    }


def notebook_model_names() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    candidates = list(get_candidate_estimators())
    return [name for name in candidates if name in source]


def safe_filename(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name.strip().lower()).strip("_")
    return f"confusion_matrix_{slug}"


def load_reported_metrics() -> dict[str, dict[str, float]]:
    path = MODEL_COMPARISON_DIR / "model_comparison.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["model"]: {
                "test_accuracy": float(row["test_accuracy"]),
                "test_macro_f1": float(row["test_macro_f1"]),
            }
            for row in csv.DictReader(handle)
        }


def train_and_predict(holdout_data: dict, model_names: list[str]) -> dict[str, dict]:
    results = {}
    estimators = get_candidate_estimators()
    for model_name in model_names:
        model = build_model_pipeline(holdout_data["x_train"], estimators[model_name])
        model.fit(holdout_data["x_train"], holdout_data["y_train"])
        pred = model.predict(holdout_data["x_test"])
        cm = confusion_matrix(holdout_data["y_test"], pred, labels=ORDERED_LABELS)
        results[model_name] = {
            "model": model,
            "pred": pred,
            "confusion_matrix": cm,
            "accuracy": float(accuracy_score(holdout_data["y_test"], pred)),
            "macro_f1": float(f1_score(holdout_data["y_test"], pred, average="macro", zero_division=0)),
        }
    return results


def plot_confusion_matrix(model_name: str, cm: np.ndarray, output_base: Path, vmax: int) -> dict:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        square=True,
        linewidths=0.8,
        linecolor="white",
        cbar=True,
        cbar_kws={"shrink": 0.86, "label": "Number of Samples"},
        annot_kws={"fontsize": 18, "fontweight": "bold"},
        xticklabels=ORDERED_NAMES,
        yticklabels=ORDERED_NAMES,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontweight="bold", pad=16)
    ax.set_xlabel("Predicted Class", labelpad=12)
    ax.set_ylabel("True Class", labelpad=12)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    ax.grid(False)
    fig.tight_layout()

    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(png_path, dpi=DPI, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)

    with Image.open(png_path) as image:
        dpi = image.info.get("dpi", (None, None))
        width, height = image.size
    return {
        "png": relative_to_root(png_path),
        "svg": relative_to_root(svg_path),
        "width_px": width,
        "height_px": height,
        "dpi": [round(float(dpi[0])), round(float(dpi[1]))],
    }


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = correct_financial_features(load_raw_data())
    model_names = notebook_model_names()
    if not model_names:
        raise RuntimeError("No compared notebook models were detected.")

    holdout_data = build_holdout_dataset(raw_df)
    results = train_and_predict(holdout_data, model_names)
    vmax = max(int(result["confusion_matrix"].max()) for result in results.values())
    reported_metrics = load_reported_metrics()

    summary = {
        "notebook": relative_to_root(NOTEBOOK_PATH),
        "output_dir": relative_to_root(IMAGE_DIR),
        "dpi_required": DPI,
        "class_order": ORDERED_NAMES,
        "test_samples": int(len(holdout_data["y_test"])),
        "models": [],
    }

    for model_name, result in results.items():
        cm = result["confusion_matrix"]
        if cm.shape != (len(ORDERED_LABELS), len(ORDERED_LABELS)):
            raise ValueError(f"{model_name}: unexpected confusion matrix shape {cm.shape}")
        if int(cm.sum()) != len(holdout_data["y_test"]):
            raise ValueError(f"{model_name}: matrix total {cm.sum()} does not match test sample count")

        image_info = plot_confusion_matrix(
            model_name,
            cm,
            IMAGE_DIR / safe_filename(model_name),
            vmax=vmax,
        )
        reported = reported_metrics.get(model_name, {})
        accuracy_delta = None
        macro_f1_delta = None
        if reported:
            accuracy_delta = result["accuracy"] - reported["test_accuracy"]
            macro_f1_delta = result["macro_f1"] - reported["test_macro_f1"]

        summary["models"].append(
            {
                "model": model_name,
                "confusion_matrix": cm.tolist(),
                "matrix_shape": list(cm.shape),
                "matrix_total": int(cm.sum()),
                "accuracy_from_matrix": result["accuracy"],
                "macro_f1_from_predictions": result["macro_f1"],
                "reported_test_accuracy": reported.get("test_accuracy"),
                "reported_test_macro_f1": reported.get("test_macro_f1"),
                "accuracy_delta_vs_reported": accuracy_delta,
                "macro_f1_delta_vs_reported": macro_f1_delta,
                "image": image_info,
            }
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = METRICS_DIR / "confusion_matrix_generation_report.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
