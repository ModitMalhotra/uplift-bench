import warnings

warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd

import learners as L

# Fits the four LightGBM models on TRAIN and saves their scores for the validation and test splits.
# Only features of val/test are used here. No val/test outcome labels are read.

FEATS = [f"f{i}" for i in range(12)]

df = pd.read_parquet("data/criteo.parquet")
train = df.loc[np.load("splits/train_idx.npy")]
Xva = df.loc[np.load("splits/val_idx.npy"), FEATS]
Xte = df.loc[np.load("splits/test_idx.npy"), FEATS]  # features only
Xtr, ytr, ttr = train[FEATS], train["visit"], train["treatment"]
with open("results/best_params.json") as f:
    best = json.load(f)

models = {"response": ("response_lgbm", L.response_model), "s_learner": ("s_learner", L.s_learner),
          "t_learner": ("t_learner", L.t_learner), "x_learner": ("x_learner", L.x_learner)}

Xboth = pd.concat([Xva, Xte])  # one fit, scores for both splits (predictions are row-wise)
for name, (key, fn) in models.items():
    s = np.asarray(fn(Xtr, ytr, ttr, Xboth, best[key]))
    np.save(f"results/scores_val_{name}.npy", s[: len(Xva)])
    np.save(f"results/scores_test_{name}.npy", s[len(Xva):])
    print(f"{name}: saved val ({len(Xva)}) and test ({len(Xte)}) scores")