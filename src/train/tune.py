# src/train/tune.py
import logging, joblib, pandas as pd
from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.config import DATA_PATH, TEST_SIZE, MODEL_PATH
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean
from src.preprocess.pipeline import build_pipeline

def main():
    # Load & prep
    df = pd.read_csv(DATA_PATH)
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

    # Base pipeline
    pipe = build_pipeline(smote_ratio=0.1)

    # Parameter distributions
    param_dist = {
        "model__n_estimators": randint(100, 500),
        "model__max_depth": [None] + list(range(5, 31, 5)),
        "model__max_features": ["sqrt", "log2", 0.2, 0.5],
        "model__min_samples_split": randint(2, 11),
        "model__min_samples_leaf": randint(1, 5)
    }

    # Randomized search with 5-fold
    cv = StratifiedKFold(5, shuffle=True)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    logging.info("Starting hyperparameter search...")
    search.fit(X_train, y_train)
    logging.info(f"Best params: {search.best_params_}")
    logging.info(f"CV AUC: {search.best_score_:.4f}")

    # Evaluate on hold-out
    best_pipe = search.best_estimator_
    auc = roc_auc_score(y_test, best_pipe.predict_proba(X_test)[:,1])
    logging.info(f"Test ROC AUC: {auc:.4f}")

    # Save tuned pipeline
    joblib.dump(best_pipe, MODEL_PATH)
    logging.info(f"Saved tuned pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
