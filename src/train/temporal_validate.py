# src/train/temporal_validate.py
#
# Temporal hold-out validation: train on the earliest 80% of transactions,
# evaluate on the most recent 20%.
#
# Why temporal? Random splits leak future information into training — the model
# sees transactions that happened *after* some it's being evaluated on.  A
# temporal split mirrors real deployment: the model always predicts on future
# data it has never seen.
#
# Why not just AUC? ROC-AUC is optimistic on imbalanced data and doesn't
# reflect production behaviour.  We report:
#   - PR-AUC  (Average Precision): measures quality across all operating points
#   - ROC-AUC: kept for comparison with prior runs
#   - Precision, Recall, F1 at the calibrated threshold (0.9991)
#   - Confusion matrix: absolute counts of misses and false alarms

import logging
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.config import SMOTE_RATIO, THRESHOLD_PATH
from src.data.load_all import load_and_normalize
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean
from src.preprocess.pipeline import build_pipeline


def main():
    logging.info("Loading and preprocessing data…")
    df = load_and_normalize()
    df = add_time_features(df)
    df = add_volume_features(df)
    df = clean(df)

    # ── Temporal split ────────────────────────────────────────────────────────
    # Sort chronologically, then cut at the 80th percentile of Time.
    # Everything before that boundary is training data; everything after is
    # held-out test data.  This is conservative — 20 % of 7.7 M rows still
    # gives a very large test set (~1.5 M transactions).
    df = df.sort_values("Time").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    cutoff_time = df["Time"].iloc[split_idx]

    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    fraud_in_test = test["Class"].sum()
    logging.info(f"Temporal cut-off: Time = {cutoff_time:.0f}s")
    logging.info(
        f"Train: {len(train):,} rows  |  "
        f"Test:  {len(test):,} rows  |  "
        f"Fraud in test: {fraud_in_test:,} ({100*fraud_in_test/len(test):.3f}%)"
    )

    X_train = train.drop(columns=["Time", "Class"])
    y_train = train["Class"]
    X_test  = test.drop(columns=["Time", "Class"])
    y_test  = test["Class"]

    # ── Train ─────────────────────────────────────────────────────────────────
    pipe = build_pipeline(smote_ratio=SMOTE_RATIO)
    pipe.fit(X_train, y_train)

    # ── Probability predictions ───────────────────────────────────────────────
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # ── Threshold ─────────────────────────────────────────────────────────────
    # Use the threshold calibrated for 90 % recall on the full dataset.
    # If the file isn't available, fall back to 0.5.
    try:
        with open(THRESHOLD_PATH) as f:
            threshold = float(f.read().strip())
        logging.info(f"Using calibrated threshold: {threshold}")
    except FileNotFoundError:
        threshold = 0.5
        logging.warning("threshold.txt not found — falling back to 0.5")

    y_pred = (y_prob >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    roc_auc  = roc_auc_score(y_test, y_prob)
    pr_auc   = average_precision_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    logging.info("─" * 50)
    logging.info(f"ROC-AUC  (all thresholds):  {roc_auc:.4f}")
    logging.info(f"PR-AUC   (all thresholds):  {pr_auc:.4f}  ← more reliable on imbalanced data")
    logging.info("─" * 50)
    logging.info(f"At threshold = {threshold}:")
    logging.info(f"  Precision  : {precision:.4f}  (of flagged txns, this % are real fraud)")
    logging.info(f"  Recall     : {recall:.4f}  (of all fraud, this % were caught)")
    logging.info(f"  F1 (fraud) : {f1:.4f}")
    logging.info("─" * 50)
    logging.info("Confusion matrix:")
    logging.info(f"  True  negatives (correctly cleared):  {tn:,}")
    logging.info(f"  False positives (false alarms):        {fp:,}")
    logging.info(f"  False negatives (missed fraud):        {fn:,}  ← keep this low")
    logging.info(f"  True  positives (caught fraud):        {tp:,}")
    logging.info("─" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
