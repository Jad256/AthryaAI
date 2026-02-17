# src/train/cross_validate.py

import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import RANDOM_STATE, SMOTE_RATIO
from src.data.load_all import load_and_normalize
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean
from src.preprocess.pipeline import build_pipeline

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Loading and preprocessing data…")
    df = load_and_normalize()
    df = add_time_features(df)
    df = add_volume_features(df)
    df = clean(df)

    # === Sanity checks ===
    print("NA count total:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    fraud_ratio = df["Class"].mean()
    print("Rows:", len(df), "Fraud ratio:", fraud_ratio)


    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]
    logging.info(f"Data ready: {X.shape[0]:,} rows × {X.shape[1]} features")

    # Build fresh RF pipeline. No joblib. No artifacts.
    pipe = build_pipeline(smote_ratio=None)

    logging.info("Starting 5-fold cross-validation…")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        error_score="raise"
    )

    logging.info(f"AUC per fold : {[round(s, 4) for s in scores]}")
    logging.info(f"Mean AUC     : {scores.mean():.4f}")
    logging.info(f"Std. Dev.    : {scores.std():.4f}")

if __name__ == "__main__":
    main()
