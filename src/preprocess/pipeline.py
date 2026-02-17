# src/preprocess/pipeline.py
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from src.config import SMOTE_RATIO, RANDOM_STATE

def build_pipeline(smote_ratio=None) -> Pipeline:
    steps = []
    if smote_ratio is not None:
        steps.append(("smote", SMOTE(sampling_strategy=smote_ratio, random_state=RANDOM_STATE)))

    steps.append(("model", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )))
    return Pipeline(steps)



def build_gbm_pipeline(smote_ratio: float = SMOTE_RATIO) -> Pipeline:
    return Pipeline([
        ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=RANDOM_STATE)),
        ("model", LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
