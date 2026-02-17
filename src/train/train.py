# src/train/train.py

import logging
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.config import TEST_SIZE, MODEL_PATH, SMOTE_RATIO, RANDOM_STATE
from src.data.load_all import load_and_normalize
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean
from src.preprocess.pipeline import build_pipeline


def main():
    # 0) Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # 1) Load full dataset
    logging.info("Loading data...")
    df = load_and_normalize()
    logging.info(f"Raw data shape: {df.shape}")

    # 2) Feature engineering
    df = add_time_features(df)
    df = add_volume_features(df)
    logging.info("Feature engineering complete")

    # 3) Cleaning
    df = clean(df)
    logging.info(f"After cleaning shape: {df.shape}")

    # === Sanity checks ===
    print("NA count total:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    fraud_ratio = df["Class"].mean()
    print("Rows:", len(df), "Fraud ratio:", fraud_ratio)

    # 4) Split features and target
    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    logging.info(f"Train shape: {X_train.shape}")
    logging.info(f"Test shape: {X_test.shape}")

    # 5) Build RF pipeline
    pipeline = build_pipeline(smote_ratio=None)

    # 6) Train
    logging.info("Training RandomForest pipeline...")
    pipeline.fit(X_train, y_train)
    logging.info("Training complete")

    # 7) Evaluate on holdout
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)

    logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test, preds))
    logging.info("Classification Report:\n%s", classification_report(y_test, preds))
    logging.info(f"ROC AUC Score: {auc:.4f}")

    # 8) Feature importance
    importances = pipeline.named_steps["model"].feature_importances_
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)

    logging.info("Top 10 features:\n%s", feat_imp.head(10).to_string(index=False))

    # 9) Save model
    joblib.dump(pipeline, MODEL_PATH)
    logging.info(f"Saved model to {MODEL_PATH}")

   
    


if __name__ == "__main__":
    main()
