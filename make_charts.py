import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import disagreement as D
import qini_curves as Q

# Final README charts, built from SAVED scores. Nothing is fitted or tuned here.
#   1) results/qini_curves_test.png      (test split)
#   2) results/profit_vs_budget_test.png (test split, three values of V)
#   3) results/shap_comparison.png       (validation; SHAP explains the models, makes no outcome claim)

SEEDS = [0, 1, 2]
VS = [25, 50, 100]
N_BOOT = 50

lab = pd.read_parquet("data/criteo.parquet", columns=["visit", "treatment"]).loc[np.load("splits/test_idx.npy")]
y, t = lab["visit"].to_numpy(), lab["treatment"].to_numpy()

scores = {
    "response": [np.load("results/scores_test_response.npy")],
    "s_learner": [np.load("results/scores_test_s_learner.npy")],
    "dragonnet": [np.load(f"results/scores_test_dragonnet_seed{k}.npy") for k in SEEDS],
}

# 1) Qini curves
Q.make_plot(scores, y, t, "results/qini_curves_test.png", split="test")
print("saved results/qini_curves_test.png")

# 2) profit vs budget
names = ["response", "s_learner", "dragonnet"]
rand = [np.random.default_rng(0).random(len(y))]


def curves(idx=None):
    out = {n: D.model_up(scores[n], y, t, idx) for n in names}
    out["random"] = D.model_up(rand, y, t, idx)
    return out


point = curves()
rng = np.random.default_rng(0)
boots = [curves(rng.integers(0, len(y), len(y))) for _ in range(N_BOOT)]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"wspace": 0.25})
for ax, V in zip(axes, VS):
    for n in names + ["random"]:
        share, up = point[n]
        p = share * (up * V - 1) * 1000
        bp = np.array([b[n][0] * (b[n][1] * V - 1) * 1000 for b in boots])
        lo, hi = np.percentile(bp, [2.5, 97.5], axis=0)
        if n == "random":
            ax.plot(D.KS * 100, p, color="#6b6b6b", ls="--", lw=1.2, label="Random targeting")
        else:
            col, _, label = Q.STYLE[n]
            ax.plot(D.KS * 100, p, color=col, lw=2, label=label)
            ax.fill_between(D.KS * 100, lo, hi, color=col, alpha=0.15, lw=0)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_title(f"1 visit = {V} ads", fontsize=11, loc="left")
    ax.set_xlabel("Budget: % of users shown the ad (highest score first)")
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Profit per 1000 users (units of one ad's cost)")
axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
fig.suptitle("Profit vs budget (test split): targeting a small top slice of users beats showing the ad to everyone",
             x=0.06, ha="left", fontsize=12)
fig.savefig("results/profit_vs_budget_test.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved results/profit_vs_budget_test.png")

# 3) SHAP side by side (images made earlier by shap_analysis.py, validation split)
imgs = [("results/shap_baseline_val.png", "Response model: who visits anyway"),
        ("results/shap_s_learner_val.png", "S-learner: who the ad moves")]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, (p, ttl) in zip(axes, imgs):
    ax.imshow(mpimg.imread(p))
    ax.axis("off")
    ax.set_title(ttl, fontsize=12, loc="left")
fig.suptitle("SHAP feature effects (validation split; surrogate R2 >= 0.99)", x=0.06, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig("results/shap_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved results/shap_comparison.png")