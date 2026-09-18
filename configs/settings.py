from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_INPUTS_DIR = DATA_DIR / "sample_inputs"

MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
CLUSTERING_MODELS_DIR = MODELS_DIR / "clustering"
SCALERS_DIR = MODELS_DIR / "scalers"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
EXPORTS_DIR = ARTIFACTS_DIR / "exports"
RECEIPTS_DIR = ARTIFACTS_DIR / "receipts"

DEFAULT_DATASET_PATH = RAW_DATA_DIR / "personal_finance_tracker_dataset.csv"
DEFAULT_MODEL_BUNDLE_PATH = TRAINED_MODELS_DIR / "finance_risk_bundle.joblib"
DEFAULT_METRICS_PATH = METRICS_DIR / "finance_risk_metrics.json"
