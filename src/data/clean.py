# src/data/clean.py

import pandas as pd
from sklearn.impute import SimpleImputer
import logging

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values and clip outliers."""
 # 1) Fill any empty spots with the median value   
    imputer = SimpleImputer(strategy="median")
    df[df.columns] = imputer.fit_transform(df)

 # 2) Clip big outliers in the Amount column
    upper = df["Amount"].quantile(0.99)
    df["Amount"] = df["Amount"].clip(0, upper)

    logging.info("Data cleaned: no missing values, outliers clipped.")
    return df
