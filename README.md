# Athrya AI – Fraud Detection API

High-performance fraud detection engine exposed as a production-ready REST API.

Athrya AI provides real-time fraud risk scoring using ensemble machine learning models trained on multi-source financial transaction datasets.

 

## Overview

Athrya AI is a fraud detection service built using:

* Stacked ensemble model (RandomForest + LightGBM)
* Canonical feature normalization across heterogeneous datasets
* Temporal and statistical feature engineering
* Threshold-based classification optimized for high recall
* FastAPI inference layer
* Dockerized deployment
* Fully testable and reproducible pipeline

The system supports real-time scoring via a REST endpoint.

 

## Model Architecture

### Base Models

* RandomForestClassifier
* LightGBMClassifier (GPU-enabled optional)

### Stacking

* Logistic Regression meta-learner
* Calibrated threshold for 90% recall

### Feature Engineering

* Canonical schema across multiple datasets
* Time-based cyclical features
* Transaction velocity features
* Data cleaning and normalization pipeline
* SMOTE for class imbalance

### Validation

* Stratified cross-validation
* Temporal hold-out validation
* ROC AUC tracking
* Threshold calibration

 

## API

### POST `/v1/predict`

Predict fraud probability for a transaction.

#### Request Body

```json
{
  "Time": 1000,
  "V1": 0.1,
  "V2": -0.2,
  ...
  "V28": 0.01,
  "Amount": 50.0
}
```

#### Response

```json
{
  "probability": 0.9982,
  "is_fraud": false
}
```

* `probability` = predicted fraud probability
* `is_fraud` = probability >= calibrated threshold

 

## Local Development

### 1. Create virtual environment

```bash
python -m venv myenv
source myenv/bin/activate  # or myenv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run API locally

```bash
uvicorn src.serve.api:app --reload
```

Visit:

```
http://127.0.0.1:8000/docs
```

 

## Docker

### Build image

```bash
docker build -t athrya-fraud-api .
```

### Run container

```bash
docker run --rm -p 8000:8000 athrya-fraud-api
```

Test:

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

 

## Training Pipeline

Training scripts are modular and reproducible:

```bash
python -m src.train.train
python -m src.train.train_gbm_mlflow
python -m src.train.tune_rf
python -m src.train.tune_gbm
python -m src.train.cross_validate
python -m src.train.temporal_validate
python -m src.train.calibrate_threshold
```

Artifacts saved to:

```
models/
  trained_model.pkl
  gbm_pipeline.pkl
  trained_model_stacked.pkl
  threshold.txt
```

 

## Testing

Run full test suite:

```bash
pytest -q
```

Includes:

* API endpoint tests
* Schema validation
* Inference consistency tests

 

## Project Structure

```
src/
  config.py
  data/
  preprocess/
  train/
  serve/
tests/
models/
data/
```

 

## Deployment Roadmap

Planned production enhancements:

* API key authentication
* Rate limiting
* CI/CD pipeline
* Monitoring & logging
* Drift detection
* Stripe subscription integration
* Cloud deployment (AWS / GCP / Azure)

 

## Performance

* Cross-validation ROC AUC: ~0.99+
* Temporal hold-out ROC AUC: ~0.997
* Threshold optimized for 90% recall
* GPU acceleration supported for LightGBM training

 

## Disclaimer

This project is for educational and demonstration purposes. Fraud detection in real financial environments requires regulatory compliance, data governance, and rigorous validation.

