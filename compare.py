import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score
from learners import response_model, s_learner, t_learner, x_learner

MODELS = {"response_lgbm": response_model, "s_learner": s_learner,
          "t_learner": t_learner, "x_learner": x_learner}
best = json.load(open("results/best_params.json"))

df = pd.read_parquet("data/criteo.parquet")
train = df.loc[np.load("splits/train_idx.npy")]
val = df.loc[np.load("splits/val_idx.npy")]
feats = [f"f{i}" for i in range(12)]

scores = {n: fn(train[feats], train["visit"], train["treatment"], val[feats], best[n])
          for n, fn in MODELS.items()}

print("\nSpearman correlation between model scores (val):")
print(pd.DataFrame(scores).corr(method="spearman").round(3).to_string())

y, t = val["visit"].to_numpy(), val["treatment"].to_numpy()
rng = np.random.default_rng(0)
boot = {n: [] for n in scores}
for _ in range(200):
    i = rng.integers(0, len(y), len(y))  # one resample, every model scored on it
    for n in scores:
        boot[n].append(qini_auc_score(y[i], scores[n][i], t[i]))

print("\nQini AUC difference vs response model (paired bootstrap, 200 resamples):")
for n in scores:
    if n == "response_lgbm":
        continue
    d = np.array(boot[n]) - np.array(boot["response_lgbm"])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {n:12s} {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")