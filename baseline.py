import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from evaluate import evaluate, show

df = pd.read_parquet("data/criteo.parquet")
train = df.loc[np.load("splits/train_idx.npy")]
val = df.loc[np.load("splits/val_idx.npy")]
feats = [f"f{i}" for i in range(12)]

# Response model: predicts P(visit) from features only. It ignores treatment.
model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                       random_state=42, verbose=-1)
model.fit(train[feats], train["visit"])
score = model.predict_proba(val[feats])[:, 1]

res = evaluate(val["visit"], score, val["treatment"])
show("response model, LightGBM (val)", res)

os.makedirs("results", exist_ok=True)
pd.DataFrame(
    [{"model": "response_lgbm", "metric": m, "value": v, "lo": lo, "hi": hi}
     for m, (v, lo, hi) in res.items()]
).to_csv("results/val_results.csv", index=False)