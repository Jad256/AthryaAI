# src/train/tune_rf.py

import logging
import joblib

from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.config import TEST_SIZE,SMOTE_RATIO, MODEL_PATH
from src.data.load_all import load_and_normalize
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean
from src.preprocess.pipeline import build_pipeline  # SMOTE → Scaler → RandomForest

def main():
    # 1) Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # 2) Load & normalize all CSVs
    df = load_and_normalize()
    logging.info(f"Combined raw data shape: {df.shape}")

    # 3) Feature engineering
    df = add_time_features(df)
    df = add_volume_features(df)
    logging.info("Added time-of-day and tx_per_minute features")

    # 4) Clean
    df = clean(df)
    logging.info(f"After clean() shape: {df.shape}")

    # 5) Train/test split
    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y
    )
    logging.info(f"Train/test split: train={X_train.shape}, test={X_test.shape}")

    # 6) Base pipeline
    pipeline = build_pipeline(smote_ratio=SMOTE_RATIO)

    # 7) Hyperparameter search space
    param_dist = {
        "model__n_estimators": randint(100, 500),
        "model__max_depth": [None] + list(range(5, 31, 5)),
        "model__max_features": ["sqrt", "log2", 0.2, 0.5],
        "model__min_samples_split": randint(2, 11),
        "model__min_samples_leaf": randint(1, 5)
    }

    # 8) RandomizedSearchCV setup
    cv = StratifiedKFold(n_splits=3, shuffle=True)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,             # adjust up/down based on time budget
        scoring="roc_auc",
        cv=cv,
        n_jobs=2,              # keep your machine responsive
        verbose=2
    )

    # 9) Run search
    logging.info("Starting RandomForest hyperparameter search...")
    search.fit(X_train, y_train)

    # 10) Report results
    logging.info(f"Best RF params: {search.best_params_}")
    logging.info(f"Best RF CV AUC: {search.best_score_:.4f}")

    # 11) Evaluate on hold-out
    best_rf = search.best_estimator_
    test_probs = best_rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    logging.info(f"RF Test ROC AUC: {test_auc:.4f}")

    # 12) Persist the tuned pipeline
    joblib.dump(best_rf, MODEL_PATH)
    logging.info(f"Saved tuned RF pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    main()
