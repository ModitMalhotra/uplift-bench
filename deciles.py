import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from learners import response_model, s_learner

best = json.load(open("results/best_params.json"))
df = pd.read_parquet("data/criteo.parquet")
train = df.loc[np.load("splits/train_idx.npy")]
val = df.loc[np.load("splits/val_idx.npy")].copy()
feats = [f"f{i}" for i in range(12)]

for name, fn in [("response_lgbm", response_model), ("s_learner", s_learner)]:
    val["score"] = fn(train[feats], train["visit"], train["treatment"], val[feats], best[name])
    # decile 1 = highest scores, decile 10 = lowest
    val["decile"] = 10 - pd.qcut(val["score"].rank(method="first"), 10, labels=False)

    rate = val.groupby(["decile", "treatment"])["visit"].mean().unstack()
    cnt = val.groupby(["decile", "treatment"])["visit"].size().unstack()
    out = pd.DataFrame({"control_rate": rate[0], "treated_rate": rate[1]})
    out["effect"] = rate[1] - rate[0]
    se = np.sqrt(rate[1] * (1 - rate[1]) / cnt[1] + rate[0] * (1 - rate[0]) / cnt[0])
    out["ci_lo"] = out["effect"] - 1.96 * se
    out["ci_hi"] = out["effect"] + 1.96 * se
    print(f"\n{name}: effect of the ad by score decile (val)")
    print(out.round(4).to_string())