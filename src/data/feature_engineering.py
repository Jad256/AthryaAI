# src/data/feature_engineering.py

import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the raw 'Time' (seconds elapsed since first transaction), derive
    time-of-day signals. Time is the highest-priority feature group because
    fraud has strong circadian patterns (spikes at night, unusual hours).

      - hour:      decimal hour of day [0, 24)
      - hour_sin, hour_cos: cyclic encoding so 23:59 and 00:01 are close
      - is_night:  1 if hour falls in 0–6 (peak fraud window), else 0
    """
    sec_day = 24 * 3600

    df["hour"] = (df["Time"] % sec_day) / 3600.0

    radians = 2 * np.pi * df["hour"] / 24
    df["hour_sin"] = np.sin(radians)
    df["hour_cos"] = np.cos(radians)

    # Night flag: transactions between midnight and 6 am are significantly
    # more likely to be fraudulent.
    df["is_night"] = (df["hour"] < 6).astype(int)

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive amount and frequency features.

      - log_amount:    log1p(Amount) — compresses the heavy right-skew so the
                       model can reason about transaction size proportionally.
                       Safe for single-row API calls.

      - tx_per_minute: number of transactions that fall in the same 1-minute
                       time bin (floor(Time / 60)).  High values indicate busy
                       periods, which correlate with fraud bursts.
                       At training time this is computed from the full dataset.
                       At API inference time (single row) it defaults to 1 —
                       "only transaction known in this minute" — a conservative
                       neutral value until a DB/cache provides real history.

    NOTE — amount_zscore will be added back once training stats (mean, std) are
    persisted to models/feature_stats.json and loaded at API startup.

    NOTE — per-card velocity (e.g. transactions by this card in the last hour)
    is higher-signal than global tx_per_minute but requires a stateful lookup at
    inference time.  Planned for the next iteration.
    """
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])

    if len(df) > 1:
        # Training / batch path: compute real transaction frequency per minute bin.
        minute_bin = (df["Time"] // 60).astype(int)
        df["tx_per_minute"] = minute_bin.map(minute_bin.value_counts())
    else:
        # Single-row inference path: no history available, default to 1.
        df["tx_per_minute"] = 1

    return df
