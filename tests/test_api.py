# tests/test_api.py

import json
import pytest
from fastapi.testclient import TestClient

from src.serve.api import app  # adjust import if your app is elsewhere

client = TestClient(app)

# a minimal “valid” payload – fill in with realistic averages
SAMPLE_TX = {
    "Time": 1000,
    "V1": 0.1,  "V2": -0.05, "V3": 0.2,  "V4": -0.1,
    "V5": 0.0,  "V6": 0.01,  "V7": -0.02, "V8": 0.3,
    "V9": -0.4, "V10": 0.02, "V11": -0.1, "V12": 0.12,
    "V13": -0.08,"V14": 0.05, "V15": 0.03, "V16": -0.07,
    "V17": 0.06, "V18": -0.01,"V19": 0.04, "V20": -0.03,
    "V21": 0.02, "V22": -0.02,"V23": 0.01, "V24": 0.0,
    "V25": 0.0,  "V26": 0.01, "V27": -0.01,"V28": 0.0,
    "Amount": 50.0
}

def test_predict_endpoint_returns_200_and_keys():
    resp = client.post("/v1/predict", json=SAMPLE_TX)
    assert resp.status_code == 200
    body = resp.json()
    assert "probability" in body and "is_fraud" in body

def test_probability_in_range():
    resp = client.post("/v1/predict", json=SAMPLE_TX)
    prob = resp.json()["probability"]
    assert 0.0 <= prob <= 1.0

def test_consistency_at_threshold():
    # force a probability right at your threshold
    # load it from file
    from pathlib import Path
    thr = float(Path("models/threshold.txt").read_text().strip())
    # simulate a “borderline” by stubbing model.predict_proba (optional)
    # here just assert that is_fraud flag matches prob>=thr
    resp = client.post("/v1/predict", json=SAMPLE_TX)
    body = resp.json()
    assert body["is_fraud"] == (body["probability"] >= thr)
