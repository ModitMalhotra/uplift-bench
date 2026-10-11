import numpy as np, pandas as pd
df = pd.read_parquet("data/criteo.parquet")
idx = np.concatenate([np.load("splits/train_idx.npy"), np.load("splits/val_idx.npy")])
cols = ["treatment", "visit", "conversion"] + [f"f{i}" for i in range(12)]
print(pd.DataFrame({"full": df[cols].mean(), "sample": df.loc[idx, cols].mean()}))