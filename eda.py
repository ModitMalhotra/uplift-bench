import numpy as np
import pandas as pd

df = pd.read_parquet("data/criteo.parquet")
idx = np.concatenate([np.load("splits/train_idx.npy"), np.load("splits/val_idx.npy")])
d = df.loc[idx]  # train + val only; the test split stays closed

feats = [f"f{i}" for i in range(12)]
t, c = d[d.treatment == 1], d[d.treatment == 0]
print("rows:", len(d), "| treated:", len(t), "| control:", len(c))

# 1. Randomization check: standardized mean difference per feature
smd = (t[feats].mean() - c[feats].mean()) / np.sqrt((t[feats].var() + c[feats].var()) / 2)
print("\nStandardized mean differences (want all near 0, under about 0.05):")
print(smd.round(4).to_string())

# 2. Average treatment effect with a 95% CI (difference in proportions)
print()
for y in ["visit", "conversion"]:
    p1, p0 = t[y].mean(), c[y].mean()
    se = np.sqrt(p1 * (1 - p1) / len(t) + p0 * (1 - p0) / len(c))
    ate = p1 - p0
    print(f"{y}: treated {p1:.5f}, control {p0:.5f}, ATE {ate:.5f}, "
          f"95% CI [{ate - 1.96 * se:.5f}, {ate + 1.96 * se:.5f}], lift {ate / p0:.1%}")