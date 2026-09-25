"""Asset-wise predictors and the return-unit prediction map."""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LOOK = 100
HID1, HID2 = 512, 256


class PerAsset(nn.Module):
    """Shared asset-wise multilayer perceptron, optionally using dropout."""

    def __init__(self, nin=LOOK, h1=HID1, h2=HID2, init=1.0, drop=0.0):
        super().__init__()
        layers = [nn.Linear(nin, h1), nn.ReLU()]
        if drop:
            layers.append(nn.Dropout(drop))
        layers += [nn.Linear(h1, h2), nn.ReLU()]
        if drop:
            layers.append(nn.Dropout(drop))
        layers.append(nn.Linear(h2, 1))
        self.f = nn.Sequential(*layers)
        if init != 1.0:
            last = [m for m in self.f.modules() if isinstance(m, nn.Linear)][-1]
            with torch.no_grad():
                last.weight.mul_(init)
                if last.bias is not None:
                    last.bias.zero_()

    def forward(self, x):
        B, N, L = x.shape
        return self.f(x.reshape(B * N, L)).view(B, N)


def zt(a, eps=1e-08):
    """Cross-sectional standardization using the sample standard deviation."""
    return (a - a.mean(-1, keepdim=True)) / (a.std(-1, keepdim=True) + eps)


def to_returns(g, anchor, mode="anchored", scale=1.0):
    """Map a predictor score to return units around the rolling sample mean."""
    sd = anchor.std(-1, keepdim=True)
    base = anchor if mode == "anchored" else anchor.mean(-1, keepdim=True)
    return base + scale * sd * zt(g)


def rank_ic(pred, real):
    """Mean date-wise Spearman rank correlation, excluding constant cross-sections."""
    out = []
    for p, r in zip(np.asarray(pred), np.asarray(real)):
        if np.std(p) < 1e-14 or np.std(r) < 1e-14:
            continue
        out.append(
            np.corrcoef(pd.Series(p).rank().values, pd.Series(r).rank().values)[0, 1]
        )
    return float(np.mean(out)) if out else float("nan")


class Conditioned(nn.Module):
    """Asset-wise predictor conditioned on a learned market embedding."""

    def __init__(self, nin, nmk, h1, h2, drop, init, dim=None):
        dim = dim if dim is not None else max(4, int(np.ceil(nin / 2.75)))
        super().__init__()
        self.emb = nn.Embedding(nmk, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.net = PerAsset(nin + dim, h1, h2, init=init, drop=drop)

    def forward(self, x, mkt):
        e = self.emb(torch.as_tensor(mkt, device=x.device)).view(1, 1, -1)
        return self.net(torch.cat([x, e.expand(x.shape[0], x.shape[1], -1)], -1))
