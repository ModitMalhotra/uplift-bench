import warnings

warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from lightgbm import LGBMClassifier, LGBMRegressor

from evaluate import evaluate, show

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
TREES = int(sys.argv[2]) if len(sys.argv) > 2 else 100

df = pd.read_parquet("data/criteo.parquet")
train = df.loc[np.load("splits/train_idx.npy")].sample(N, random_state=0)
val = df.loc[np.load("splits/val_idx.npy")]
feats = [f"f{i}" for i in range(12)]

cf = CausalForestDML(
    model_y=LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1),
    model_t=LGBMClassifier(n_estimators=100, learning_rate=0.05, verbose=-1),
    discrete_treatment=True,
    n_estimators=TREES,
    min_samples_leaf=100,
    max_depth=12,
    cv=2,
    random_state=42,
    n_jobs=-1,
)

t0 = time.time()
cf.fit(train["visit"].to_numpy(), train["treatment"].to_numpy(), X=train[feats])
print(f"fit: {time.time() - t0:.0f}s")

score = cf.effect(val[feats]).ravel()
res = evaluate(val["visit"], score, val["treatment"])
name = f"causal_forest_{N // 1000}k"
test_X = df.loc[np.load("splits/test_idx.npy"), feats]
np.save(f"results/scores_val_{name}.npy", score)
np.save(f"results/scores_test_{name}.npy", cf.effect(test_X).ravel())
show(f"{name} (val)", res)

rows = [
    {"model": name, "metric": m, "value": v, "lo": lo, "hi": hi}
    for m, (v, lo, hi) in res.items()
]
old = pd.read_csv("results/val_results.csv")
old = old[old["model"] != name]
pd.concat([old, pd.DataFrame(rows)]).to_csv("results/val_results.csv", index=False)
