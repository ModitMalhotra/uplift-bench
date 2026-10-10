import numpy as np
import pandas as pd
from evaluate import evaluate, show
from learners import s_learner, t_learner, x_learner

df = pd.read_parquet("data/criteo.parquet")
train = df.loc[np.load("splits/train_idx.npy")]
val = df.loc[np.load("splits/val_idx.npy")]
feats = [f"f{i}" for i in range(12)]

rows = []
for name, fn in [("s_learner", s_learner), ("t_learner", t_learner),
                 ("x_learner", x_learner)]:
    score = fn(train[feats], train["visit"], train["treatment"], val[feats])
    res = evaluate(val["visit"], score, val["treatment"])
    show(f"{name} (val)", res)
    rows += [{"model": name, "metric": m, "value": v, "lo": lo, "hi": hi}
             for m, (v, lo, hi) in res.items()]

new = pd.DataFrame(rows)
old = pd.read_csv("results/val_results.csv")
old = old[~old["model"].isin(new["model"])]
pd.concat([old, new]).to_csv("results/val_results.csv", index=False)