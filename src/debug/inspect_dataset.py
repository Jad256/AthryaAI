import pandas as pd

df = pd.read_csv("data/raw/creditcard_2023.csv")

print("Is id sorted:", df["id"].is_monotonic_increasing)
print("First 10 ids:", df["id"].head(10).tolist())
