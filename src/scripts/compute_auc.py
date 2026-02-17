# src/scripts/count_transactions.py

import pandas as pd
from src.data.load_all import load_and_normalize
from src.config import RAW_PATHS

def main():
    # Load and normalize all your CSVs
    df = load_and_normalize()
    total = len(df)
    print(f"→ Total transactions (combined): {total:,}")

    # Also show per-file raw counts
    print("\nPer-file row counts:")
    for path in RAW_PATHS:
        count = len(pd.read_csv(path))
        print(f"  {path.name:35s}: {count:,}")

if __name__ == "__main__":
    main()
