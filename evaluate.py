import numpy as np
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k


def evaluate(y, score, t, k=0.2, n_boot=100, seed=0):
    """y: outcome (0/1), score: higher = more uplift, t: treatment (0/1)."""
    y, score, t = map(np.asarray, (y, score, t))

    def metrics(y, score, t):
        return {
            "qini_auc": qini_auc_score(y, score, t),
            "uplift_auc": uplift_auc_score(y, score, t),
            "uplift@20%": uplift_at_k(y, score, t, strategy="overall", k=k),
        }

    point = metrics(y, score, t)
    rng = np.random.default_rng(seed)
    boots = {m: [] for m in point}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        for m, v in metrics(y[i], score[i], t[i]).items():
            boots[m].append(v)
    return {m: (point[m], *np.percentile(boots[m], [2.5, 97.5])) for m in point}


def show(name, res):
    print(f"\n{name}")
    for m, (v, lo, hi) in res.items():
        print(f"  {m:12s} {v:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    # Sanity check: a random score should give Qini/uplift AUC near zero
    import pandas as pd

    df = pd.read_parquet("data/criteo.parquet")
    val = df.loc[np.load("splits/val_idx.npy")]
    rng = np.random.default_rng(1)
    show("random score (val)",
         evaluate(val.visit, rng.random(len(val)), val.treatment, n_boot=50))