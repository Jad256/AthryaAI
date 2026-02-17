# src/train/train_gbm_mlflow.py

import logging
import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import PROJECT_ROOT, TEST_SIZE, RANDOM_STATE, MODEL_PATH, SMOTE_RATIO
from src.data.load_all import load_and_normalize
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean
from src.preprocess.pipeline import build_gbm_pipeline

def main():
    # 0) Logging config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 1) Load & preprocess
    df = load_and_normalize()
    df = add_time_features(df)
    df = add_volume_features(df)
    df = clean(df)

    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y
    )

    # 2) Enable MLflow autologging
    mlflow.set_experiment("CreditCardFraud_GBM")
    mlflow.sklearn.autolog(log_input_examples=True)

    with mlflow.start_run():
        # 3) Build & train the LightGBM pipeline
        pipeline = build_gbm_pipeline(smote_ratio=SMOTE_RATIO)
        pipeline.fit(X_train, y_train)

        # 4) Save a local copy for stacking later
        gbm_out = PROJECT_ROOT / "models" / "gbm_pipeline.pkl"
        joblib.dump(pipeline, gbm_out)
        logging.info(f"Saved local GBM pipeline to {gbm_out}")

        # 5) Manually log signature & input example to MLflow
        preds_train = pipeline.predict(X_train)
        signature = infer_signature(X_train, preds_train)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(2)
        )

        # 6) Evaluate on hold-out & log metric
        probs = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        mlflow.log_metric("test_auc", float(auc))
        print(f"Logged GBM run with test AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
