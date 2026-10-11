import warnings

warnings.filterwarnings("ignore")

import json
import os
import numpy as np
import pandas as pd


FEATS = [f"f{i}" for i in range(12)]
KS = np.linspace(0.01, 1.0, 100)
V_LIST = [25, 50, 100]
KMAX = 0.40
N_BOOT = 200
SEED = 0


def up_curve(score, y, t, idx=None):
    """share targeted and uplift (treated rate - control rate) among the top-k, for each budget k"""
    if idx is None:
        order = np.argsort(-score, kind="stable")
    else:
        order = idx[np.argsort(-score[idx], kind="stable")]
    yy, tt = y[order], t[order]
    n = len(order)
    c_t, c_ty = np.cumsum(tt), np.cumsum(tt * yy)
    c_c, c_cy = np.cumsum(1 - tt), np.cumsum((1 - tt) * yy)
    ns = np.maximum((KS * n).astype(int), 1) - 1
    up = c_ty[ns] / np.maximum(c_t[ns], 1) - c_cy[ns] / np.maximum(c_c[ns], 1)
    return (ns + 1) / n, up


def model_up(slist, y, t, idx=None):
    """average over seeds (profit is linear in uplift, so averaging uplift = averaging profit)"""
    ups = []
    for s in slist:
        share, up = up_curve(s, y, t, idx)
        ups.append(up)
    return share, np.mean(ups, axis=0)


def mean_profit(share, up, V):
    m = KS <= KMAX
    return float(np.mean((share * (up * V - 1))[m]))


def paired_bootstrap(scores, y, t, base="response"):
    rng = np.random.default_rng(SEED)
    n = len(y)
    others = [m for m in scores if m not in (base, "random")]
    share, up_r = model_up(scores[base], y, t)
    point = {}
    for m in others:
        _, up = model_up(scores[m], y, t)
        for V in V_LIST:
            point[(m, V)] = (mean_profit(share, up, V) - mean_profit(share, up_r, V)) * 1000
    diffs = {k: [] for k in point}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        share, up_r = model_up(scores[base], y, t, idx)
        b = {V: mean_profit(share, up_r, V) for V in V_LIST}
        for m in others:
            _, up = model_up(scores[m], y, t, idx)
            for V in V_LIST:
                diffs[(m, V)].append((mean_profit(share, up, V) - b[V]) * 1000)
    rows = []
    for (m, V), p in point.items():
        lo, hi = np.percentile(diffs[(m, V)], [2.5, 97.5])
        verdict = "better" if lo > 0 else "worse" if hi < 0 else "tie"
        rows.append({"model": m, "V": V, "diff_per_1000": round(p, 2),
                     "ci_lo": round(lo, 2), "ci_hi": round(hi, 2), "verdict": verdict})
    return pd.DataFrame(rows)


def disagree(s_up, s_resp, y, t, q=0.8):
    top_u = s_up >= np.quantile(s_up, q)
    top_r = s_resp >= np.quantile(s_resp, q)
    groups = {
        "both top 20%": top_u & top_r,
        "uplift-only top 20%": top_u & ~top_r,
        "response-only top 20%": top_r & ~top_u,
        "neither": ~top_u & ~top_r,
    }
    rows = []
    for g, m in groups.items():
        yy, tt = y[m], t[m]
        n1, n0 = int(tt.sum()), int((1 - tt).sum())
        if n1 < 2 or n0 < 2:
            continue
        p1, p0 = yy[tt == 1].mean(), yy[tt == 0].mean()
        eff = p1 - p0
        se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        rows.append({"group": g, "n": int(m.sum()), "n_control": n0,
                     "control_rate": round(p0, 4), "treated_rate": round(p1, 4),
                     "effect": round(eff, 4), "ci_lo": round(eff - 1.96 * se, 4),
                     "ci_hi": round(eff + 1.96 * se, 4)})
    return pd.DataFrame(rows)


def main():
    import learners as L

    df = pd.read_parquet("data/criteo.parquet")
    train = df.loc[np.load("splits/train_idx.npy")]
    val = df.loc[np.load("splits/val_idx.npy")]
    with open("results/best_params.json") as f:
        best = json.load(f)
    Xtr, ytr, ttr = train[FEATS], train["visit"], train["treatment"]
    Xva, y, t = val[FEATS], val["visit"].to_numpy(), val["treatment"].to_numpy()

    models = {"response": ("response_lgbm", L.response_model), "s_learner": ("s_learner", L.s_learner),
              "t_learner": ("t_learner", L.t_learner), "x_learner": ("x_learner", L.x_learner)}
    scores = {}
    for name, (key, fn) in models.items():
        scores[name] = [np.asarray(fn(Xtr, ytr, ttr, Xva, best[key]))]
    for v in ["tarnet", "dragonnet"]:
        files = [f"results/scores_val_{v}_seed{s}.npy" for s in range(3)]
        if all(os.path.exists(f) for f in files):
            scores[v] = [np.load(f) for f in files]

    print("PART A: paired bootstrap vs response model")
    print("metric: mean profit per 1000 users over budgets 1-40%; positive = better than response")
    pd.set_option("display.width", 200)
    print(paired_bootstrap(scores, y, t).to_string(index=False))

    print("\nPART B: top-20% disagreement (effect = treated visit rate - control visit rate)")
    for name in ["s_learner", "x_learner", "dragonnet"]:
        if name in scores:
            print(f"\n{name} (seed 0 for deep models) vs response:")
            print(disagree(scores[name][0], scores["response"][0], y, t).to_string(index=False))


if __name__ == "__main__":
    main()