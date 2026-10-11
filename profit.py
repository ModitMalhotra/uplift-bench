import warnings

warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd

import learners as L

# Profit-vs-budget simulation on VALIDATION. Test split stays closed until Day 5.
#
# Assumptions (Criteo has no prices, so everything is in units of the cost of showing one ad):
#   - showing an ad costs 1 unit
#   - one extra visit is worth V units  (V is varied in a sensitivity table)
#   - we target the top-k fraction of users by score, k = budget
#   - incremental visits in the targeted group = (treated visit rate - control visit rate) * group size,
#     estimated from the randomized data inside that group (no model assumptions)
#   - profit per user of the whole population = (n_k / N) * (uplift_k * V - 1)

V_MAIN = 50
V_GRID = [10, 25, 50, 100, 200]
KS = np.linspace(0.01, 1.0, 100)
N_BOOT = 50
SEED = 0
FEATS = [f"f{i}" for i in range(12)]

df = pd.read_parquet("data/criteo.parquet")
train = df.loc[np.load("splits/train_idx.npy")]
val = df.loc[np.load("splits/val_idx.npy")]
with open("results/best_params.json") as f:
    best = json.load(f)
print("best_params keys:", list(best.keys()))

Xtr, ytr, ttr = train[FEATS], train["visit"], train["treatment"]
Xva, y, t = val[FEATS], val["visit"].to_numpy(), val["treatment"].to_numpy()

MODELS = {
    "response": L.response_model,
    "s_learner": L.s_learner,
    "t_learner": L.t_learner,
    "x_learner": L.x_learner,
}

scores = {}
for name, fn in MODELS.items():
    p = best.get({"response": "response_lgbm"}.get(name, name))
    print(
        f"{name}: {'tuned params' if p else 'DEFAULT params (not found in best_params.json)'}"
    )
    scores[name] = np.asarray(fn(Xtr, ytr, ttr, Xva, p))
scores["random"] = np.random.default_rng(SEED).random(len(y))


def curve(score, y, t, idx=None, V=V_MAIN):
    """profit per user of the whole population, for each budget k in KS"""
    if idx is None:
        order = np.argsort(-score, kind="stable")
    else:  # bootstrap: keep the score ordering, resample rows
        order = idx[np.argsort(-score[idx], kind="stable")]
    yy, tt = y[order], t[order]
    n = len(order)
    c_t, c_ty = np.cumsum(tt), np.cumsum(tt * yy)
    c_c, c_cy = np.cumsum(1 - tt), np.cumsum((1 - tt) * yy)
    ns = np.maximum((KS * n).astype(int), 1) - 1
    up = c_ty[ns] / np.maximum(c_t[ns], 1) - c_cy[ns] / np.maximum(c_c[ns], 1)
    return (ns + 1) / n * (up * V - 1), up


# 1) main curves with bootstrap bands at V_MAIN
rng = np.random.default_rng(SEED)
res = {}
for name, s in scores.items():
    pt, _ = curve(s, y, t)
    boots = np.array(
        [curve(s, y, t, idx=rng.integers(0, len(y), len(y)))[0] for _ in range(N_BOOT)]
    )
    res[name] = (
        pt,
        np.percentile(boots, 2.5, axis=0),
        np.percentile(boots, 97.5, axis=0),
    )

# 2) sensitivity table: best budget and profit vs treating everyone, for several V
rows = []
for V in V_GRID:
    for name, s in scores.items():
        pt, up = curve(s, y, t, V=V)
        all_pt, _ = curve(scores["random"], y, t, V=V)
        i = int(np.argmax(pt))
        rows.append(
            {
                "V": V,
                "model": name,
                "best_budget": round(KS[i], 2),
                "profit_per_1000_users": round(pt[i] * 1000, 2),
                "treat_all_per_1000": round(all_pt[-1] * 1000, 2),
            }
        )
sens = pd.DataFrame(rows)
print("\nSensitivity (profit per 1000 users, units = cost of one ad):")
print(sens.pivot(index="model", columns="V", values="profit_per_1000_users"))
print("\nBest budget share by value of a visit:")
print(sens.pivot(index="model", columns="V", values="best_budget"))
print("\nTreat everyone, per 1000 users:")
print(sens[sens.model == "response"].set_index("V")["treat_all_per_1000"])
sens.to_csv("results/profit_sensitivity_val.csv", index=False)

# 3) chart
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (pt, lo, hi) in res.items():
        ax.plot(
            KS * 100,
            pt * 1000,
            label=name,
            lw=2 if name != "random" else 1,
            ls="--" if name == "random" else "-",
        )
        ax.fill_between(KS * 100, lo * 1000, hi * 1000, alpha=0.12)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Budget: % of users shown the ad (highest score first)")
    ax.set_ylabel("Profit per 1000 users (units of one ad's cost)")
    ax.set_title(f"Profit vs budget (validation, 1 visit = {V_MAIN} ads)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/profit_vs_budget_val.png", dpi=150)
    print("saved results/profit_vs_budget_val.png")
except ImportError:
    print("matplotlib missing: pip install matplotlib")
