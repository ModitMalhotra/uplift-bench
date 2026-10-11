import warnings
warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate import evaluate, show

# usage: python dragonnet.py [tarnet|dragonnet] [max_epochs]
#   tarnet     outcome heads only
#   dragonnet  + propensity head + targeted regularization
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "dragonnet"
MAX_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ALPHA = 1.0 if VARIANT == "dragonnet" else 0.0  # weight of the propensity loss
BETA = 1.0 if VARIANT == "dragonnet" else 0.0  # weight of targeted regularization
SEEDS = [0, 1, 2]
BATCH, LR, PATIENCE = 4096, 1e-3, 3
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| variant:", VARIANT)

df = pd.read_parquet("data/criteo.parquet")
train = df.loc[np.load("splits/train_idx.npy")]
val = df.loc[np.load("splits/val_idx.npy")]
feats = [f"f{i}" for i in range(12)]

mu, sd = train[feats].mean(), train[feats].std() + 1e-6  # scaling from train only


def tens(d):
    x = torch.tensor(((d[feats] - mu) / sd).to_numpy(np.float32))
    return (x.to(dev), torch.tensor(d["treatment"].to_numpy(np.float32)).to(dev),
            torch.tensor(d["visit"].to_numpy(np.float32)).to(dev))


Xtr, Ttr, Ytr = tens(train)
Xva, Tva, Yva = tens(val)


class DragonNet(nn.Module):
    def __init__(self, d_in=12, h=128, hh=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.ELU(), nn.Linear(h, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU())
        self.head1 = nn.Sequential(nn.Linear(h, hh), nn.ELU(), nn.Linear(hh, 1))  # treated
        self.head0 = nn.Sequential(nn.Linear(h, hh), nn.ELU(), nn.Linear(hh, 1))  # control
        self.prop = nn.Linear(h, 1)  # propensity head
        self.eps = nn.Parameter(torch.zeros(1))  # targeted-regularization correction

    def forward(self, x):
        z = self.enc(x)
        return self.head1(z).squeeze(1), self.head0(z).squeeze(1), self.prop(z).squeeze(1)


def loss_fn(model, x, t, y):
    l1, l0, lp = model(x)
    logit_y = torch.where(t > 0.5, l1, l0)  # outcome head matching the real treatment
    loss = F.binary_cross_entropy_with_logits(logit_y, y)
    if ALPHA > 0:
        loss = loss + ALPHA * F.binary_cross_entropy_with_logits(lp, t)
    if BETA > 0:
        # clever covariate H = t/g - (1-t)/(1-g); the corrected logit is logit_y + eps * H.
        # g is detached so this term only trains the outcome side and eps.
        g = torch.sigmoid(lp).detach().clamp(0.01, 0.99)
        h = t / g - (1 - t) / (1 - g)
        loss = loss + BETA * F.binary_cross_entropy_with_logits(logit_y + model.eps * h, y)
    return loss


@torch.no_grad()
def val_loss(model):
    model.eval()
    tot = 0.0
    for i in range(0, len(Xva), 65536):
        s = slice(i, i + 65536)
        tot += loss_fn(model, Xva[s], Tva[s], Yva[s]).item() * len(Xva[s])
    return tot / len(Xva)


@torch.no_grad()
def uplift(model):
    model.eval()
    out = []
    for i in range(0, len(Xva), 65536):
        l1, l0, lp = model(Xva[i:i + 65536])
        if BETA > 0:  # use the corrected predictions: H = 1/g for treated, -1/(1-g) for control
            g = torch.sigmoid(lp).clamp(0.01, 0.99)
            l1 = l1 + model.eps / g
            l0 = l0 - model.eps / (1 - g)
        out.append((torch.sigmoid(l1) - torch.sigmoid(l0)).cpu())
    return torch.cat(out).numpy()


rows = []
for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DragonNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    best, best_state, bad, t0 = 1e9, None, 0, time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model, Xtr[b], Ttr[b], Ytr[b]).backward()
            opt.step()
        vl = val_loss(model)
        print(f"  seed {seed} epoch {epoch + 1}: val loss {vl:.5f}")
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    print(f"seed {seed}: {epoch + 1} epochs, {time.time() - t0:.0f}s, eps {model.eps.item():+.4f}")
    res = evaluate(val["visit"], uplift(model), val["treatment"])
    name = f"{VARIANT}_seed{seed}"
    np.save(f"results/scores_val_{name}.npy", uplift(model))
    show(f"{name} (val)", res)
    rows += [{"model": name, "metric": m, "value": v, "lo": lo, "hi": hi}
             for m, (v, lo, hi) in res.items()]

new = pd.DataFrame(rows)
old = pd.read_csv("results/val_results.csv")
old = old[~old["model"].isin(new["model"])]
pd.concat([old, new]).to_csv("results/val_results.csv", index=False)

q = new[new.metric == "qini_auc"]["value"]
print(f"\n{VARIANT} Qini AUC over {len(SEEDS)} seeds: mean {q.mean():.4f}, sd {q.std():.4f}")