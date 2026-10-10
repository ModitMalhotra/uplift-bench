from lightgbm import LGBMClassifier, LGBMRegressor

PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
)


def s_learner(Xtr, ytr, ttr, Xva):
    m = LGBMClassifier(**PARAMS).fit(Xtr.assign(t=ttr.to_numpy()), ytr)
    return (
        m.predict_proba(Xva.assign(t=1))[:, 1] - m.predict_proba(Xva.assign(t=0))[:, 1]
    )


def t_learner(Xtr, ytr, ttr, Xva):
    m1 = LGBMClassifier(**PARAMS).fit(Xtr[ttr == 1], ytr[ttr == 1])
    m0 = LGBMClassifier(**PARAMS).fit(Xtr[ttr == 0], ytr[ttr == 0])
    return m1.predict_proba(Xva)[:, 1] - m0.predict_proba(Xva)[:, 1]


def x_learner(Xtr, ytr, ttr, Xva):
    X1, y1 = Xtr[ttr == 1], ytr[ttr == 1]
    X0, y0 = Xtr[ttr == 0], ytr[ttr == 0]
    m1 = LGBMClassifier(**PARAMS).fit(X1, y1)
    m0 = LGBMClassifier(**PARAMS).fit(X0, y0)
    d1 = y1 - m0.predict_proba(X1)[:, 1]  # imputed effect for treated
    d0 = m1.predict_proba(X0)[:, 1] - y0  # imputed effect for control
    tau1 = LGBMRegressor(**PARAMS).fit(X1, d1)
    tau0 = LGBMRegressor(**PARAMS).fit(X0, d0)
    g = ttr.mean()  # share treated; constant because treatment was randomized
    return g * tau0.predict(Xva) + (1 - g) * tau1.predict(Xva)
