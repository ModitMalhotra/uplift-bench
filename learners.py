from lightgbm import LGBMClassifier, LGBMRegressor

PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              random_state=42, verbose=-1)


def _p(params):
    return {**PARAMS, **(params or {})}


def response_model(Xtr, ytr, ttr, Xva, params=None):
    m = LGBMClassifier(**_p(params)).fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


def s_learner(Xtr, ytr, ttr, Xva, params=None):
    m = LGBMClassifier(**_p(params)).fit(Xtr.assign(t=ttr.to_numpy()), ytr)
    return (m.predict_proba(Xva.assign(t=1))[:, 1]
            - m.predict_proba(Xva.assign(t=0))[:, 1])


def t_learner(Xtr, ytr, ttr, Xva, params=None):
    P = _p(params)
    m1 = LGBMClassifier(**P).fit(Xtr[ttr == 1], ytr[ttr == 1])
    m0 = LGBMClassifier(**P).fit(Xtr[ttr == 0], ytr[ttr == 0])
    return m1.predict_proba(Xva)[:, 1] - m0.predict_proba(Xva)[:, 1]


def x_learner(Xtr, ytr, ttr, Xva, params=None):
    P = _p(params)
    X1, y1 = Xtr[ttr == 1], ytr[ttr == 1]
    X0, y0 = Xtr[ttr == 0], ytr[ttr == 0]
    m1 = LGBMClassifier(**P).fit(X1, y1)
    m0 = LGBMClassifier(**P).fit(X0, y0)
    d1 = y1 - m0.predict_proba(X1)[:, 1]
    d0 = m1.predict_proba(X0)[:, 1] - y0
    tau1 = LGBMRegressor(**P).fit(X1, d1)
    tau0 = LGBMRegressor(**P).fit(X0, d0)
    g = ttr.mean()
    return g * tau0.predict(Xva) + (1 - g) * tau1.predict(Xva)