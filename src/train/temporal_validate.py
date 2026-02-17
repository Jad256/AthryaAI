# src/train/temporal_validate.py

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import STACKED_MODEL_PATH as MODEL_PATH
from src.data.load_all import load_and_normalize
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean

def main():
    # 1) Load & preprocess
    df = load_and_normalize()
    df = add_time_features(df)
    df = add_volume_features(df)
    df = clean(df)

    # 2) Temporal split at the 80th percentile of Time
    threshold = df["Time"].quantile(0.8)
    train_df = df[df["Time"] <= threshold]
    test_df  = df[df["Time"]  > threshold]

    X_train = train_df.drop(["Class", "Time"], axis=1)
    y_train = train_df["Class"]
    X_test  = test_df.drop(["Class", "Time"], axis=1)
    y_test  = test_df["Class"]

    print(f"Using Time <= {threshold:.0f}s for training, > {threshold:.0f}s for test")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 3) Load & re-fit pipeline
    pipe = joblib.load(MODEL_PATH)
    pipe.fit(X_train, y_train)

    # 4) Evaluate
    probs = pipe.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, probs)
    print(f"Temporal hold-out ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
