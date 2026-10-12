import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

from evaluate import evaluate, show
import disagreement as D

# Follows the frozen test-split plan in PROTOCOL.md.
#   python test_final.py --split val    dry run: identical code path on validation (do this first)
#   python test_final.py --split test   the one real run. Refuses to run twice.
# Reads saved scores (results/scores_<split>_<model>.npy) and outcome labels. Nothing is fitted here.

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["val", "test"], required=True)
split = ap.parse_args().split

SINGLE = ["response", "s_learner", "t_learner", "x_learner", "causal_forest_1200k"]
DEEP = ["tarnet", "dragonnet"]
SEEDS = [0, 1, 2]
V_PROFIT = [10, 25, 50, 100, 200]
N_BOOT_PROFIT = 100
prefix = "test" if split == "test" else "dryrun_val"


def path(sp, name):
    return f"results/scores_{sp}_{name}.npy"


def needed(sp):
    return ([path(sp, n) for n in SINGLE]
            + [path(sp, f"{v}_seed{k}") for v in DEEP for k in SEEDS])


# 1) guards (checked BEFORE the flag file is written, so a missing file never burns the single run)
missing = [p for sp in ("val", split) for p in needed(sp) if not os.path.exists(p)]
if missing:
    sys.exit("Missing score files:\n  " + "\n  ".join(sorted(set(missing))))
if split == "test":
    if not os.path.exists("results/dryrun_val_ok.txt"):
        sys.exit("Run `python test_final.py --split val` first.")
    if os.path.exists("results/TEST_OPENED.txt"):
        sys.exit("The test split has already been opened (results/TEST_OPENED.txt). Protocol: once only.")
    with open("results/TEST_OPENED.txt", "w") as f:
        f.write(time.strftime("test split opened %Y-%m-%d %H:%M:%S\n"))
    print("TEST SPLIT OPENED. This run is final.\n")
else:
    print("DRY RUN on validation (budgets and evaluation both use val). Not a result.\n")

# 2) labels and scores
lab = pd.read_parquet("data/criteo.parquet", columns=["visit", "treatment"])
lv, le = lab.loc[np.load("splits/val_idx.npy")], lab.loc[np.load(f"splits/{split}_idx.npy")]
yv, tv = lv["visit"].to_numpy(), lv["treatment"].to_numpy()
ye, te = le["visit"].to_numpy(), le["treatment"].to_numpy()


def load(sp, n_rows):
    s = {n: [np.load(path(sp, n))] for n in SINGLE}
    for v in DEEP:
        s[v] = [np.load(path(sp, f"{v}_seed{k}")) for k in SEEDS]
    for n, lst in s.items():
        for a in lst:
            assert len(a) == n_rows, f"{n}: {len(a)} scores for {n_rows} rows"
    return s


sv, se = load("val", len(yv)), load(split, len(ye))
sv["random"] = [np.random.default_rng(0).random(len(yv))]
se["random"] = [np.random.default_rng(0).random(len(ye))]

# 3) ranking metrics with 95% bootstrap intervals
rows = []
for name in SINGLE + ["random"]:
    res = evaluate(ye, se[name][0], te)
    show(f"{name} ({split})", res)
    rows += [{"model": name, "metric": m, "value": v, "lo": lo, "hi": hi} for m, (v, lo, hi) in res.items()]
for v in DEEP:
    qs = []
    for k, s in enumerate(se[v]):
        res = evaluate(ye, s, te)
        show(f"{v}_seed{k} ({split})", res)
        rows += [{"model": f"{v}_seed{k}", "metric": m, "value": x, "lo": lo, "hi": hi}
                 for m, (x, lo, hi) in res.items()]
        qs.append(res["qini_auc"][0])
    print(f"\n{v} Qini AUC over {len(SEEDS)} seeds: mean {np.mean(qs):.4f}, sd {np.std(qs, ddof=1):.4f}")
pd.DataFrame(rows).to_csv(f"results/{prefix}_metrics.csv", index=False)

# 4) profit: budget chosen on VALIDATION per model and V, applied once to the evaluation split
models = SINGLE + DEEP
KS = D.KS
bidx = {}
for n in models:
    share, up = D.model_up(sv[n], yv, tv)
    for V in V_PROFIT:
        bidx[(n, V)] = int(np.argmax(share * (up * V - 1)))


def profits(idx=None):
    out = {}
    curves = {n: D.model_up(se[n], ye, te, idx) for n in models + ["random"]}
    for V in V_PROFIT:
        for n in models:
            share, up = curves[n]
            i = bidx[(n, V)]
            out[(n, V)] = share[i] * (up[i] * V - 1) * 1000
        share, up = curves["random"]
        out[("treat_all", V)] = share[-1] * (up[-1] * V - 1) * 1000
        i = bidx[("response", V)]
        out[("random@response_budget", V)] = share[i] * (up[i] * V - 1) * 1000
    return out


point = profits()
rng = np.random.default_rng(1)
boots = {k: [] for k in point}
for _ in range(N_BOOT_PROFIT):
    for k, val_ in profits(rng.integers(0, len(ye), len(ye))).items():
        boots[k].append(val_)
prow = []
for (n, V), p in point.items():
    lo, hi = np.percentile(boots[(n, V)], [2.5, 97.5])
    prow.append({"model": n, "V": V, "budget": round(float(KS[bidx[(n, V)]]), 2) if (n, V) in bidx else np.nan,
                 "profit_per_1000": round(p, 1), "ci_lo": round(lo, 1), "ci_hi": round(hi, 1)})
prof = pd.DataFrame(prow)
prof.to_csv(f"results/{prefix}_profit.csv", index=False)
print(f"\nPROFIT per 1000 users on {split} (budget chosen on validation; units = cost of one ad)")
for V in V_PROFIT:
    print(f"\nV = {V}")
    print(prof[prof.V == V].drop(columns="V").to_string(index=False))

# 5) paired comparisons against the response model
print("\nPRIMARY (confirmatory): S-learner vs response, V = 50, mean profit over budgets 1-40%")
D.V_LIST = [50]
conf = D.paired_bootstrap({"response": se["response"], "s_learner": se["s_learner"]}, ye, te)
print(conf.to_string(index=False))
print("\nEXPLORATORY: all models vs response")
D.V_LIST = [25, 50, 100]
expl = D.paired_bootstrap({n: se[n] for n in models + ["response"]}, ye, te)
print(expl.to_string(index=False))
pd.concat([conf.assign(kind="primary"), expl.assign(kind="exploratory")]).to_csv(
    f"results/{prefix}_paired.csv", index=False)

if split == "val":
    with open("results/dryrun_val_ok.txt", "w") as f:
        f.write("dry run completed\n")
    print("\nDry run finished OK. Now you may run: python test_final.py --split test")
else:
    print("\nDone. Results saved as results/test_*.csv. Do not re-run.")