# src/data/load_all.py

import pandas as pd
from src.config import RAW_PATHS, CANONICAL_COLS

def load_and_normalize():
    dfs = []
    for path in RAW_PATHS:
        df = pd.read_csv(path)

        # 1) Rename to canonical names
        df = df.rename(columns={
            "step": "Time",
            "amount": "Amount",
            "isFraud": "Class",
            "isFlaggedFraud": "Class",
            # add any others you discover…
        })

        # 2) Ensure every canonical column exists
        for col in CANONICAL_COLS:
            if col not in df.columns:
                # PCA comps & Amount → 0.0; Class/Time → NA or 0
                df[col] = 0.0 if col.startswith("V") or col == "Amount" else pd.NA

        # 3) Subset to only the canonical columns (drop extras)
        df = df[CANONICAL_COLS]

        dfs.append(df)

    # 4) Concatenate all dataframes
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    return combined
