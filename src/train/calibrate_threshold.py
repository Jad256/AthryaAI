# src/train/calibrate_threshold.py

import joblib
from sklearn.metrics import precision_recall_curve
from src.data.load_all import load_and_normalize
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean
from src.config import STACKED_MODEL_PATH as MODEL_PATH

def main():
    # 1) Load & preprocess
    df = load_and_normalize()
    df = add_time_features(df)
    df = add_volume_features(df)
    df = clean(df)

    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]

    # 2) Load model
    pipeline = joblib.load(MODEL_PATH)

    # 3) Predict probabilities
    probs = pipeline.predict_proba(X)[:, 1]

    # 4) Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, probs)

    # 5) Find threshold for 90% recall
    desired_recall = 0.90
    idx = next(i for i, r in enumerate(recall) if r < desired_recall) - 1
    opt_threshold = thresholds[idx]

    print(f"Optimal threshold for {desired_recall*100:.0f}% recall is {opt_threshold:.4f}")
    # Optionally save this threshold
    with open("models/threshold.txt", "w") as f:
        f.write(str(opt_threshold))

if __name__ == "__main__":
    main()
