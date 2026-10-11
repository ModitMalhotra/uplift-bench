# Protocol (written 10 Oct 2026, before any model was trained)

## Data
Criteo Uplift v2.1: 13,979,592 rows, 12 features, treatment, visit, conversion, exposure.
License CC BY-NC-SA 4.0. Cite Diemert et al., AdKDD 2018.

## Outcome and treatment
- Primary outcome: visit. Secondary outcome: conversion.
- Treatment column: `treatment` (randomized). `exposure` is not used as treatment.

## Splits
- 2M rows drawn at random from the whole file (seed 42).
- 60/20/20 train/val/test, stratified by treatment x visit. Row indices saved in `splits/`.
- The test split is opened once, on Day 5 (Wed 14 Oct).

## Metrics
- Primary: Qini AUC (scikit-uplift `qini_auc_score`), for every model.
- Secondary: uplift AUC (AUUC) and uplift@20%, and the same metrics on conversion.
- 95% bootstrap intervals on every metric, 3 seeds for the PyTorch model.
- Overlapping intervals are reported as a tie.
- Secondary analysis added on 10 Oct after seeing validation results: paired bootstrap of Qini AUC differences against the response model, plus score correlations. The primary rule (overlapping intervals count as a tie) is unchanged.
- Deviation (11 Oct 2026): conversion is not evaluated. At 0.29% it is too sparse for stable intervals; visit is the only outcome reported.

## Rules
- Tuning (Optuna) uses validation only, with a capped number of trials.
- Nothing is changed or re-run after the test split is opened.
- Sanity check: a random score must give AUUC near zero.
- Profit simulation assumptions are stated in the README.
- Weak or negative results are reported as they are.
- Optuna: 20 trials per model (the first is the default settings), same search space and seed for every model, tuned on a 400k-row slice of train and scored by validation Qini AUC. Final fits use all of train.
- Causal forest is trained on a random subsample of train (size stated in the results table) because of compute cost. It is scored on the same full validation set as every other model.
- DragonNet/TARNet: early stopping on validation factual loss (patience 3), never on Qini. Three seeds, each reported separately plus mean and SD.
- Profit simulation: the budget (share of users targeted) is chosen on validation only. On the test split that fixed budget is applied once; the test split is never used to choose it. Value of a visit (V) is an assumption, reported as a sensitivity table (V = 10, 25, 50, 100, 200). (added 11 Oct 2026, before the test split was opened)
- Confirmatory test (declared 11 Oct 2026, before the test split was opened): the one primary
  hypothesis is that the S-learner earns more profit than the response model at V = 50
  (mean profit over budgets 1-40%, paired bootstrap, 95% interval). All other comparisons on
  the test split are exploratory and reported as such.

## Test-split plan (frozen 11 Oct 2026, before the test split was opened)

Models evaluated, each fit on the full train split with the tuned settings in best_params.json
(fixed as of this date; no re-tuning):
- response model (LightGBM, ignores treatment)
- S-learner, T-learner, X-learner (LightGBM)
- Causal forest: the final model is fit on the full train split. Runs on 400k-row subsamples (200 and 800 trees) were sensitivity checks on validation only.
- TARNet and DragonNet (3 seeds each; seeds 0, 1, 2; early stopping on validation factual loss)

Metrics on test (95% bootstrap intervals): Qini AUC (primary), uplift AUC, uplift@20%.
Overlapping intervals are reported as a tie.

Profit: for each model and each V in {10, 25, 50, 100, 200}, the budget is the one that maximizes
validation profit for that model. That fixed budget is applied once to the test split.
Deep models: profit curves are averaged over their 3 seeds before choosing the budget.
Treat-all and random targeting are reported as references.

Primary (confirmatory) comparison: S-learner vs response model, V = 50, mean profit over
budgets 1-40%, paired bootstrap. All other comparisons are exploratory.

Test is opened once. Whatever the result, it is reported.