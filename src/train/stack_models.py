# src/train/stack_models.py

import logging
import joblib

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import (
    MODEL_PATH,
    GBM_PIPELINE_PATH,
    STACKED_MODEL_PATH,
    TEST_SIZE,
    RANDOM_STATE
)
from src.data.load_all import load_and_normalize
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # 1) Load & prep data
    df = load_and_normalize()
    df = add_time_features(df)
    df = add_volume_features(df)
    df = clean(df)
    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y
    )

    # 3) Load your pre‐trained pipelines
    rf  = joblib.load(MODEL_PATH)
    gbm = joblib.load(GBM_PIPELINE_PATH)

    # 4) Build stacking classifier in prefit mode
    stack = StackingClassifier(
        estimators=[("rf", rf), ("gbm", gbm)],
        final_estimator=LogisticRegression(),
        cv="prefit",       # <<— skip re-fitting rf & gbm
        n_jobs=-1,
        passthrough=False
    )

    # 5) Fit only the meta‐learner & evaluate once
    stack.fit(X_train, y_train)
    probs = stack.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    logging.info(f"Stacked model ROC AUC: {auc:.4f}")

    # 6) Persist the stacked artifact
    joblib.dump(stack, STACKED_MODEL_PATH)
    logging.info(f"Saved stacked model to {STACKED_MODEL_PATH}")

if __name__ == "__main__":
    main()
