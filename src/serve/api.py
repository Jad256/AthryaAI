# src/serve/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean
from src.config import STACKED_MODEL_PATH, THRESHOLD_PATH

# ─── Startup: load your model + threshold ─────────────────────────────────────
pipeline = joblib.load(str(STACKED_MODEL_PATH))
threshold = float(Path(THRESHOLD_PATH).read_text().strip())

app = FastAPI()

# ─── Request & response schemas ──────────────────────────────────────────────
class Transaction(BaseModel):
    Time: float
    V1: float;  V2: float;  V3: float;  V4: float
    V5: float;  V6: float;  V7: float;  V8: float
    V9: float;  V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    Amount: float

class Prediction(BaseModel):
    probability: float
    is_fraud: bool

# ─── Prediction endpoint ────────────────────────────────────────────────────
@app.post("/v1/predict", response_model=Prediction)
def predict(tx: Transaction):
    # A) single‐row DataFrame from the payload
    df = pd.DataFrame([tx.model_dump()])  # use tx.dict() if on Pydantic v1

    # B) re‐apply your train‐time feature engineering & cleaning
    df = add_time_features(df)
    df = add_volume_features(df)
    df = clean(df)

    # C) drop extras (“Time” and possibly “Class”), ignore missing
    X = df.drop(["Time", "Class"], axis=1, errors="ignore")

    # D) predict probability and threshold‐flag
    prob = float(pipeline.predict_proba(X)[:, 1][0])
    is_fraud = prob >= threshold

    return Prediction(probability=prob, is_fraud=is_fraud)
