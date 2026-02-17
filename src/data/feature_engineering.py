# src/data/feature_engineering.py

import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the raw 'Time' (seconds elapsed since first transaction), derive:
      - hour: 0–24
      - hour_sin, hour_cos: cyclic encoding of time of day
    """
    # seconds in a day
    sec_day = 24 * 3600

    # 1) Hour of day
    #   take Time modulo one day, then divide by 3600 to get hours [0,24)
    df["hour"] = (df["Time"] % sec_day) / 3600.0

    # 2) Cyclical encoding
    radians = 2 * np.pi * df["hour"] / 24
    df["hour_sin"] = np.sin(radians)
    df["hour_cos"] = np.cos(radians)

    return df
