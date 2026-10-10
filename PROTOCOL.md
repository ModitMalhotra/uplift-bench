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

## Rules
- Tuning (Optuna) uses validation only, with a capped number of trials.
- Nothing is changed or re-run after the test split is opened.
- Sanity check: a random score must give AUUC near zero.
- Profit simulation assumptions are stated in the README.
- Weak or negative results are reported as they are.
- Optuna: 20 trials per model (the first is the default settings), same search space and seed for every model, tuned on a 400k-row slice of train and scored by validation Qini AUC. Final fits use all of train.