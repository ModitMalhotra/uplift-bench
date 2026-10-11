import warnings

warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

import learners as L

FEATS = [f"f{i}" for i in range(12)]
SEED, N_SHAP = 0, 20000

df = pd.read_parquet("data/criteo.parquet")
train = df.loc[np.load("splits/train_idx.npy")]
val = df.loc[np.load("splits/val_idx.npy")]
with open("results/best_params.json") as f:
    best = json.load(f)

Xtr, ytr, ttr = train[FEATS], train["visit"], train["treatment"]
Xva = val[FEATS]

scores = {
    "uplift (S-learner)": np.asarray(
        L.s_learner(Xtr, ytr, ttr, Xva, best["s_learner"])
    ),
    "response (baseline engagement)": np.asarray(
        L.response_model(Xtr, ytr, ttr, Xva, best["response_lgbm"])
    ),
    "uplift (TARNet seed0)": np.load("results/scores_val_tarnet_seed0.npy"),
    "uplift (DragonNet seed0)": np.load("results/scores_val_dragonnet_seed0.npy"),
}

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(Xva))
fit_idx, hold_idx = perm[: len(perm) // 2], perm[len(perm) // 2 :]
shap_idx = hold_idx[:N_SHAP]

importance = {}
for name, s in scores.items():
    sur = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
    )
    sur.fit(Xva.iloc[fit_idx], s[fit_idx])
    r2 = sur.score(Xva.iloc[hold_idx], s[hold_idx])
    Xs = Xva.iloc[shap_idx]
    sv = shap.TreeExplainer(sur).shap_values(Xs)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATS).sort_values(ascending=False)
    importance[name] = imp
    print(f"\n{name}: surrogate R2 on held-out rows = {r2:.3f}")
    print((imp / imp.sum()).round(3).head(6).to_string())
    plt.figure()
    shap.summary_plot(sv, Xs, show=False, max_display=12)
    plt.title(f"SHAP: {name}")
    plt.tight_layout()
    fname = "shap_" + name.split("(")[1].split(")")[0].split()[0].lower().replace(
        "-", "_"
    )
    plt.savefig(f"results/{fname}_val.png", dpi=150)
    plt.close()

resp = importance["response (baseline engagement)"].reindex(FEATS)
print()
for name, imp in importance.items():
    if name.startswith("uplift"):
        rho = spearmanr(imp.reindex(FEATS), resp)[0]
        print(f"Spearman of feature importance, {name} vs response: {rho:.2f}")
print("saved results/shap_<model>_val.png")
