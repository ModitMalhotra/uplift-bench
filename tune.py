import json
import os
import sys

import numpy as np
import optuna
import pandas as pd
from sklift.metrics import qini_auc_score

from evaluate import evaluate, show
from learners import response_model, s_learner, t_learner, x_learner

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MODELS = {
    "response_lgbm": response_model,
    "s_learner": s_learner,
    "t_learner": t_learner,
    "x_learner": x_learner,
}

df = pd.read_parquet("data/criteo.parquet")
train = df.loc[np.load("splits/train_idx.npy")]
val = df.loc[np.load("splits/val_idx.npy")]
feats = [f"f{i}" for i in range(12)]
tune_train = train.sample(400_000, random_state=0)  # speed; final fit uses all of train


def suggest(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 500, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
    )


optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs("results", exist_ok=True)
best, rows = {}, []

for name, fn in MODELS.items():

    def objective(trial, fn=fn):
        score = fn(
            tune_train[feats],
            tune_train["visit"],
            tune_train["treatment"],
            val[feats],
            suggest(trial),
        )
        return qini_auc_score(val["visit"], score, val["treatment"])

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.enqueue_trial(
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "reg_lambda": 1e-3,
            "colsample_bytree": 1.0,
            "subsample": 1.0,
        }
    )
    study.optimize(objective, n_trials=N_TRIALS)
    best[name] = {**study.best_params, "subsample_freq": 1}
    print(f"{name}: best tuning Qini AUC {study.best_value:.4f}")
    study.trials_dataframe().to_csv(f"results/trials_{name}.csv", index=False)

    # Final fit on all of train with the best settings, scored with bootstrap CIs
    score = fn(train[feats], train["visit"], train["treatment"], val[feats], best[name])
    res = evaluate(val["visit"], score, val["treatment"])
    show(f"{name}, tuned (val)", res)
    rows += [
        {"model": f"{name}_tuned", "metric": m, "value": v, "lo": lo, "hi": hi}
        for m, (v, lo, hi) in res.items()
    ]

json.dump(best, open("results/best_params.json", "w"), indent=2)
old = pd.read_csv("results/val_results.csv")
old = old[~old["model"].str.endswith("_tuned")]
pd.concat([old, pd.DataFrame(rows)]).to_csv("results/val_results.csv", index=False)
