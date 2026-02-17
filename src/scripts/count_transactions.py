# scripts/count_transactions.py

from src.data.load_all import load_and_normalize
from src.config import RAW_PATHS

def main():
    # This loads and normalizes all your raw CSVs…
    df = load_and_normalize()
    total = len(df)
    print(f"→ Total transactions (all datasets combined): {total:,}")

    # If you also want per‐file counts:
    import pandas as pd
    for path in RAW_PATHS:
        df_i = pd.read_csv(path)
        print(f"  {path.name:35s}: {len(df_i):,}")

if __name__ == "__main__":
    main()
