import pandas as pd
from common.io import read_parquet

df = read_parquet("data/processed/AAPL/v1/features.parquet")
print(f"Total rows: {len(df)}")
print(f"Unique options: {df['option_key'].nunique()}")
counts = df.groupby("option_key").size()
print(f"Max rows per option: {counts.max()}")
print(f"Min rows per option: {counts.min()}")
print(f"Mean rows per option: {counts.mean()}")
print(f"Options with >= 32 rows: {(counts >= 32).sum()}")
