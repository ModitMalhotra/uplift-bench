import warnings

warnings.filterwarnings("ignore")

import json
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATS = [f"f{i}" for i in range(12)]
KS = np.linspace(0.005, 1.0, 200)
N_BOOT = 50
SEED = 0

# colour-blind-safe (Okabe-Ito); identity is also carried by direct labels, not colour alone
STYLE = {
    "response": ("#0072B2", "-", "Response model (ignores treatment)"),
    "s_learner": ("#D55E00", "-", "S-learner"),
    "dragonnet": ("#009E73", "-", "DragonNet (mean of 3 seeds)"),
}


def qini(score, y, t, idx=None):
    """cumulative incremental visits per 1000 users at each budget in KS"""
    if idx is None:
        order = np.argsort(-score, kind="stable")
    else:
        order = idx[np.argsort(-score[idx], kind="stable")]
    yy, tt = y[order], t[order]
    n = len(order)
    c_t, c_ty = np.cumsum(tt), np.cumsum(tt * yy)
    c_c, c_cy = np.cumsum(1 - tt), np.cumsum((1 - tt) * yy)
    ns = np.maximum((KS * n).astype(int), 1) - 1
    inc = c_ty[ns] - c_cy[ns] * c_t[ns] / np.maximum(c_c[ns], 1)
    return inc / n * 1000


def mean_qini(slist, y, t, idx=None):
    return np.mean([qini(s, y, t, idx) for s in slist], axis=0)


def make_plot(scores, y, t, path, split="validation"):
    rng = np.random.default_rng(SEED)
    res = {}
    for name, sl in scores.items():
        pt = mean_qini(sl, y, t)
        boots = np.array(
            [
                mean_qini(sl, y, t, rng.integers(0, len(y), len(y)))
                for _ in range(N_BOOT)
            ]
        )
        res[name] = (
            pt,
            np.percentile(boots, 2.5, axis=0),
            np.percentile(boots, 97.5, axis=0),
        )
    total = qini(scores["response"][0], y, t)[
        -1
    ]  # all users targeted: same for every score

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.22})
    for ax, xmax, title in [
        (axes[0], 100, "All budgets"),
        (axes[1], 30, "Zoom: first 30%"),
    ]:
        for name, (pt, lo, hi) in res.items():
            col, ls, label = STYLE[name]
            ax.plot(KS * 100, pt, color=col, ls=ls, lw=2, label=label)
            ax.fill_between(KS * 100, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(
            [0, 100],
            [0, total],
            color="#6b6b6b",
            ls="--",
            lw=1.2,
            label="Random targeting",
        )
        ax.set_xlim(0, xmax)
        if xmax < 100:
            m = KS * 100 <= xmax
            ymax = max(r[2][m].max() for r in res.values()) * 1.08
            ax.set_ylim(0, ymax)
        ax.set_xlabel("Budget: % of users shown the ad (highest score first)")
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Extra visits per 1000 users (cumulative)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle(
        f"Qini curves ({split}): the ad's extra visits, by who is targeted first",
        x=0.06,
        ha="left",
        fontsize=12,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    import learners as L

    df = pd.read_parquet("data/criteo.parquet")
    train = df.loc[np.load("splits/train_idx.npy")]
    val = df.loc[np.load("splits/val_idx.npy")]
    with open("results/best_params.json") as f:
        best = json.load(f)
    Xtr, ytr, ttr = train[FEATS], train["visit"], train["treatment"]
    Xva, y, t = val[FEATS], val["visit"].to_numpy(), val["treatment"].to_numpy()

    scores = {
        "response": [
            np.asarray(L.response_model(Xtr, ytr, ttr, Xva, best["response_lgbm"]))
        ],
        "s_learner": [np.asarray(L.s_learner(Xtr, ytr, ttr, Xva, best["s_learner"]))],
    }
    files = [f"results/scores_val_dragonnet_seed{s}.npy" for s in range(3)]
    if all(os.path.exists(f) for f in files):
        scores["dragonnet"] = [np.load(f) for f in files]
    make_plot(scores, y, t, "results/qini_curves_val.png")
    print("saved results/qini_curves_val.png")


if __name__ == "__main__":
    main()
