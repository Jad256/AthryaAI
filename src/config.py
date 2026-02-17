# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data locations
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_PATHS = {
    "cc_public": RAW_DIR / "creditcard.csv",
    "cc_2023": RAW_DIR / "creditcard_2023.csv",
    "fraud_2019_2020": RAW_DIR / "fraudTest_2019-2020.csv",
    "synthetic_financial": RAW_DIR / "Synthetic_Financial_Dataset.csv",
}

# Canonical merged dataset path. This is the one every train and CV script should use.
CANONICAL_DATA_PATH = PROCESSED_DIR / "transactions_canonical.parquet"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = ARTIFACTS_DIR / "trained_model.pkl"
STACKED_MODEL_PATH = ARTIFACTS_DIR / "trained_model_stacked.pkl"
GBM_PIPELINE_PATH = ARTIFACTS_DIR / "gbm_pipeline.pkl"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.txt"

# Other constants
TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_RATIO = 0.1
