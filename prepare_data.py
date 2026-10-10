import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
dtypes = {f"f{i}": "float32" for i in range(12)}
dtypes.update({c: "int8" for c in ["treatment", "visit", "conversion", "exposure"]})

df = pd.concat(
    pd.read_csv("data/criteo-uplift-v2.1.csv.gz", dtype=dtypes, chunksize=1_000_000),
    ignore_index=True,
)
df.to_parquet("data/criteo.parquet")

# 2M random rows from the whole file, then 60/20/20 stratified by treatment x visit
sample = df.sample(n=2_000_000, random_state=SEED)
strata = sample["treatment"].astype(str) + "_" + sample["visit"].astype(str)
train_idx, rest_idx = train_test_split(
    sample.index, test_size=0.4, stratify=strata, random_state=SEED
)
val_idx, test_idx = train_test_split(
    rest_idx, test_size=0.5, stratify=strata.loc[rest_idx], random_state=SEED
)

os.makedirs("splits", exist_ok=True)
for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    np.save(f"splits/{name}_idx.npy", np.asarray(idx))
print(len(train_idx), len(val_idx), len(test_idx))