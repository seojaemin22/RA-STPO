"""Window-derived asset features with date-wise normal-score normalization.

Input arrays have shape (windows, lookback, assets). Cross-sectional normalization
uses the complete input universe before any decision-face restriction.
"""

from __future__ import annotations
import numpy as np
from scipy.special import ndtri

NAMES = (
    "return_1",
    "return_2_5",
    "return_6_21",
    "return_22_63",
    "return_64_T",
    "mean",
    "volatility",
    "volatility_21",
    "downside_volatility",
    "maximum_21",
    "minimum_21",
    "skewness",
    "excess_kurtosis",
    "positive_fraction",
    "terminal_drawdown",
    "autocorrelation_1",
    "mean_to_volatility",
    "volatility_ratio",
    "market_beta",
    "residual_volatility",
    "residual_mean",
    "market_correlation",
)


def normal_scores(A, eps=1e-12):
    """Cross-sectional rank -> normal score, per date and per feature."""
    r = A.argsort(axis=1).argsort(axis=1).astype(np.float64)
    n = A.shape[1]
    return ndtri((r + 0.5) / n)


def build(W):
    """Compute summaries using NumPy's ordinal-rank convention at ties."""
    B, T, N = W.shape
    m = W.mean(1)
    sd = W.std(1) + 1e-14
    neg = np.minimum(W, 0.0)
    cum = np.cumprod(1.0 + W, axis=1)
    peak = np.maximum.accumulate(cum, axis=1)
    c = W - m[:, None, :]
    sk = (c**3).mean(1) / sd**3
    ku = (c**4).mean(1) / sd**4 - 3.0
    a = c[:, :-1, :]
    b = c[:, 1:, :]
    ac1 = (a * b).mean(1) / (a.std(1) * b.std(1) + 1e-14)
    mk = W.mean(2, keepdims=True)
    mkc = mk - mk.mean(1, keepdims=True)
    vm = (mkc**2).mean(1) + 1e-18
    beta = (c * mkc).mean(1) / vm
    res = W - beta[:, None, :] * mk
    idio = res.std(1)
    residmom = res.mean(1)
    corrmkt = (c * mkc).mean(1) / (sd * np.sqrt(vm))
    f = np.stack(
        [
            W[:, -1, :],
            W[:, -5:-1, :].mean(1),
            W[:, -21:-5, :].mean(1),
            W[:, -63:-21, :].mean(1),
            W[:, :-63, :].mean(1),
            m,
            sd,
            W[:, -21:, :].std(1),
            neg.std(1),
            W[:, -21:, :].max(1),
            W[:, -21:, :].min(1),
            sk,
            ku,
            (W > 0).mean(1),
            1.0 - cum[:, -1, :] / peak[:, -1, :],
            ac1,
            m / sd,
            W[:, -21:, :].std(1) / sd,
            beta,
            idio,
            residmom,
            corrmkt,
        ],
        axis=-1,
    )
    out = np.empty_like(f)
    for j in range(f.shape[-1]):
        out[:, :, j] = normal_scores(f[:, :, j])
    return out.astype(np.float32)
