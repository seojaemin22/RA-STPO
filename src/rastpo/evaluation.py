"""Portfolio paths, seed-level metrics, calibrated ensembles, and member resampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import Dataset, file_hash, save_json
from .layers import exact_layer, budgeted_tangency, cholesky
from . import numerics
from .runtime import environment, measured, synchronize
from .selection import greedy_face_safe_draws

MEAN_ABSOLUTE_NORMAL = 0.7978845608


def metrics(weights, targets, annualization=252.0):
    weights, targets = np.asarray(weights, float), np.asarray(targets, float)
    returns = (weights * targets).sum(axis=1)
    volatility = returns.std(ddof=1)
    wealth_path = np.r_[1.0, np.cumprod(1 + returns)]
    return dict(
        sharpe=(
            float(returns.mean() / volatility * np.sqrt(annualization))
            if volatility > 0
            else None
        ),
        wealth=float(np.prod(1 + returns)),
        mdd=float(np.max(1 - wealth_path / np.maximum.accumulate(wealth_path))),
        card=float((np.abs(weights) > 1e-10).sum(1).mean()),
        turnover=(
            float(0.5 * np.abs(np.diff(weights, axis=0)).sum(1).mean())
            if len(weights) > 1
            else 0.0
        ),
        mean=float(returns.mean()),
        vol=float(volatility),
        net=float(weights.sum(1).mean()),
        gross=float(np.abs(weights).sum(1).mean()),
    )


def calibrate(anchors, corrected):
    """Per-member/date cross-sectional moment matching in float64."""
    anchors, corrected = np.asarray(anchors, np.float64), np.asarray(
        corrected, np.float64
    )
    sd = corrected.std(axis=-1, keepdims=True, ddof=1)
    z = (corrected - corrected.mean(axis=-1, keepdims=True)) / np.maximum(sd, 1e-300)
    return (
        anchors.mean(axis=-1, keepdims=True)
        + anchors.std(axis=-1, keepdims=True, ddof=1) * z
    )


def aggregate(members, diagonal, lookback, uncertainty, mean_precision="float64"):
    if mean_precision not in {"float32", "float64"}:
        raise ValueError("unknown mean reduction precision")
    center = (
        np.asarray(members, dtype=np.dtype(mean_precision))
        .mean(axis=0)
        .astype(np.float64)
    )
    variance = np.zeros_like(center)
    if uncertainty == "mean":
        variance += diagonal / lookback
    if uncertainty == "mean" and len(members) > 1:
        variance += members.std(axis=0, ddof=1) ** 2 / len(members)
    if uncertainty not in {"none", "mean"}:
        raise ValueError("unknown uncertainty model")
    return center - MEAN_ABSOLUTE_NORMAL * np.sqrt(variance), variance


@dataclass(frozen=True)
class DecisionConfig:
    optimizer: str = "alpha"
    cardinality: int = 10
    aggregation: str = "calibrated"
    uncertainty: str = "mean"
    credibility: str = "none"
    position: str = "long-only"
    ordering: str = "stored"
    mean_precision: str = "float64"
    annualization: float = 252.0
    batch_size: int | None = None
    pga_ridge: float = 1e-3
    pga_tolerance: float = 1e-6
    pga_iterations: int = 3000
    sdp_tolerance: float = 1e-4
    sdp_iterations: int = 100000
    sdp_max_seconds: float = 600.0

    def validate(self):
        if self.uncertainty not in {"none", "mean"}:
            raise ValueError("unknown uncertainty model")
        if self.mean_precision not in {"float32", "float64"}:
            raise ValueError("unknown mean reduction precision")
        if self.aggregation not in {"individual", "mean", "calibrated", "anchor-mean"}:
            raise ValueError("unknown forecast aggregation")
        if self.credibility not in {"none", "sharpe"}:
            raise ValueError("unknown credibility statistic")
        if self.credibility != "none" and self.aggregation not in {"calibrated", "anchor-mean"}:
            raise ValueError("credibility needs corrected and anchor forecast legs")
        if self.credibility != "none" and (
            self.optimizer != "alpha" or self.position != "long-only"
        ):
            raise ValueError("credibility requires exact long-only asset-face allocation")
        if self.position not in {"long-only", "long-short"}:
            raise ValueError("unknown position constraint")
        if self.position == "long-short" and self.optimizer not in {
            "oscar",
            "dfstpo",
            "sdp",
            "alpha",
        }:
            raise ValueError("this optimizer is intrinsically long-only")
        if self.ordering not in {"seeded", "stored", "stream"} or self.cardinality < 1:
            raise ValueError("invalid ordering or cardinality")
        if self.optimizer == "afba" and self.cardinality < 4:
            raise ValueError(
                "the released ASMP-AFBA adaptation requires cardinality >= 4"
            )
        if self.optimizer not in {"alpha", "oscar", "dfstpo", "pga", "afba", "sdp"}:
            raise ValueError("unknown allocation optimizer")
        if (
            self.sdp_tolerance <= 0
            or self.sdp_iterations < 1
            or self.sdp_max_seconds < 0
            or self.annualization <= 0
            or self.pga_ridge < 0
            or self.pga_tolerance <= 0
            or self.pga_iterations < 1
            or (self.batch_size is not None and self.batch_size < 1)
        ):
            raise ValueError("invalid allocation or reporting settings")


def chunk_size(assets, requested=None):
    return (
        requested
        if requested is not None
        else max(2, min(96, int(4e6 / (assets * assets))))
    )


def long_short_face(S, signal, face):
    Sff, muf = numerics.gather_face(S, signal, face)
    ray = torch.linalg.solve(Sff, muf.unsqueeze(-1)).squeeze(-1)
    gross = ray.abs().sum(-1, keepdim=True)
    wf = torch.where(
        gross > 1e-14,
        ray / gross.clamp_min(1e-14),
        torch.full_like(ray, 1 / face.shape[-1]),
    )
    return torch.zeros_like(signal).scatter(1, face, wf)


def permutation(assets, seed, ordering):
    """Independent seeded orders, stored order, or indexed draws from one stream."""
    if ordering == "stored":
        return None
    if ordering == "seeded":
        return numerics.asset_order(assets, seed)
    if ordering == "stream":
        if seed < 0:
            raise ValueError("stream draw indices must be nonnegative")
        generator = torch.Generator().manual_seed(0)
        for _ in range(seed + 1):
            order = torch.randperm(assets, generator=generator)
        return order
    raise ValueError("unknown asset ordering")


@torch.no_grad()
def allocate(S, signal, config, seed=0, ranking=None):
    k = config.cardinality
    if config.optimizer == "dfstpo":
        order = permutation(signal.shape[-1], seed, config.ordering)
        q = (torch.arange(signal.shape[-1], device=signal.device)
             if order is None else order.to(signal.device))
        sp, mp = S[:, q][:, :, q], signal[:, q]
        L = cholesky(sp)
        direction = budgeted_tangency(mp, chol=L)
        score = torch.abs(torch.bmm(L.transpose(1, 2), direction.unsqueeze(-1)).squeeze(-1))
        face = q[numerics.topk_idx(score, k)]
        if config.position == "long-only":
            return numerics.tangency_on_support(S, signal, face)
        Sff, muf = numerics.gather_face(S, signal, face)
        ray = budgeted_tangency(muf, Sff)
        gross = ray.abs().sum(-1, keepdim=True)
        wf = torch.where(gross > 1e-14, ray / gross.clamp_min(1e-14),
                         torch.full_like(ray, 1 / face.shape[-1]))
        return torch.zeros_like(signal).scatter(1, face, wf)
    if config.optimizer == "alpha":
        nonnegative = torch.relu(signal)
        face, mask = greedy_face_safe_draws(S, nonnegative[:, None], k)
        face, mask = face[:, 0], mask[:, 0]
        if config.position == "long-short":
            return long_short_face(S, signal, face)
        return exact_layer(S, nonnegative, k, face=face, face_mask=mask)[0]
    if config.optimizer == "oscar":
        order = permutation(signal.shape[-1], seed, config.ordering)
        if config.position == "long-only":
            return numerics.oscar(signal, S, k, perm=order)
        q = (
            torch.arange(signal.shape[-1], device=signal.device)
            if order is None
            else order.to(signal.device)
        )
        sp, mp = S[:, q][:, :, q], signal[:, q]
        L = numerics._chol(sp)
        ray = torch.cholesky_solve(mp.unsqueeze(-1), L).squeeze(-1)
        score = torch.abs(torch.bmm(L.transpose(1, 2), ray.unsqueeze(-1)).squeeze(-1))
        face = q[numerics.topk_idx(score, k)]
        return long_short_face(S, signal, face)
    if config.optimizer == "pga":
        return numerics.mssrm_pga(
            signal,
            S,
            k,
            eps=config.pga_ridge,
            tol=config.pga_tolerance,
            iters=config.pga_iterations,
        )
    if config.optimizer == "afba":
        return numerics.asmp_afba(signal, S, k)
    if config.optimizer == "sdp":
        if ranking is None:
            raise ValueError("SDP allocation requires a hash-verified ranking cache")
        face = numerics.topk_idx(ranking, k)
        return (
            numerics.tangency_on_support(S, signal, face)
            if config.position == "long-only"
            else long_short_face(S, signal, face)
        )
    raise ValueError(f"unknown optimizer: {config.optimizer}")


def load_members(dataset, market, forecasts, seeds, *, anchor=False, since=None):
    arrays, hashes = [], []
    for seed in seeds:
        directory = Path(forecasts) / f"seed_{seed}"
        record = json.loads((directory / "fit.json").read_text())
        if record["seed"] != seed or record["dataset_identity"] != dataset.hash:
            raise ValueError("forecast identity does not match evaluation inputs")
        if anchor:
            if record["anchor_directory"] is None:
                raise ValueError("calibration requires an anchored correction fit")
            expected_anchor = record["anchor_identity"]
            anchor_path = Path(record["anchor_directory"])
            directory = (
                anchor_path if anchor_path.is_absolute() else directory / anchor_path
            )
            record = json.loads((directory / "fit.json").read_text())
            if (record["seed"] != seed or record["dataset_identity"] != dataset.hash
                    or record["identity"] != expected_anchor):
                raise ValueError("anchor identity does not match evaluation inputs")
        path = directory / f"{market.name}.npy"
        digest = file_hash(path)
        if digest != record["files_sha256"][path.name]:
            raise ValueError(f"forecast hash mismatch: {path}")
        value = np.load(path, allow_pickle=False)
        offset = record.get("forecast_start", {}).get(market.name, 0)
        start = market.test.start if since is None else since
        if not 0 <= offset <= start:
            raise ValueError(f"forecast start does not cover evaluation: {path}")
        if value.shape != (market.n - offset, market.N) or not np.isfinite(value).all():
            raise ValueError(f"invalid forecast shape/values: {path}")
        arrays.append(value[start - offset :].astype(np.float64))
        hashes.append(dict(seed=seed, sha256=digest))
    return np.stack(arrays), hashes


def credibility_face(corrected, anchor, credibility, cardinality):
    """Select actual assets from a mixture of separately normalized books."""
    def normalized(book):
        maximum = book.max(axis=1, keepdims=True)
        return np.divide(book, maximum, out=np.zeros_like(book), where=maximum > 0)
    score = (credibility[:, None] * normalized(corrected)
             + (1.0 - credibility[:, None]) * normalized(anchor))
    return np.argsort(-score, axis=1)[:, :cardinality]


class _Split:
    """Presents one split of a market as if it were the evaluation window.

    portfolio_path is written against `market.test`; the credibility prior has
    to run the same solver over the validation block.  Nothing else about the
    market changes, so a thin view is enough.
    """

    def __init__(self, market, span):
        self._market, self._span = market, span
        self.name, self.N, self.root = market.name, market.N, market.root
        self.metadata = market.metadata
        self.n = span.stop if span.stop is not None else market.n

    @property
    def test(self):
        return self._span

    def array(self, name, mode="r"):
        return self._market.array(name, mode)


def _leg_weights(market, span, members, config, device, lookback, seed):
    """Sparse portfolio path for one leg over one split."""
    view = _Split(market, span)
    covariance = np.asarray(market.array("covariances")[span], np.float64)
    diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
    signal, _ = aggregate(members, diagonal, lookback,
                          config.uncertainty, config.mean_precision)
    weights, _ = portfolio_path(view, signal, config, device, seed=seed)
    return weights


def credible_weights(market, members, anchors, config, device, lookback, base,
                     *, seed=0, return_trace=False):
    """Causal credibility selects a face and blends its robust return signal.

    The two books contribute relative allocation scores; credibility is not a
    final exposure fraction. Final weights exactly solve the selected face.
    Mean signals retain their own q^2/M terms and are clipped before blending.
    """
    from scipy.stats import norm

    val = market.validation
    span = market.test
    local = lambda sp: slice(sp.start - base, (sp.stop or market.n) - base)
    lv, ls = local(val), local(span)
    wc_v = _leg_weights(market, val, members[:, lv], config, device,
                        lookback, seed)
    plain = config
    wa_v = _leg_weights(market, val, anchors[:, lv], plain, device,
                        lookback, seed)
    y_v = np.asarray(market.array("targets")[val], np.float64)
    rc_v, ra_v = (wc_v * y_v).sum(1), (wa_v * y_v).sum(1)
    sc, sa = (max(rc_v.std(ddof=1), 1e-12), max(ra_v.std(ddof=1), 1e-12)) if config.credibility == "sharpe" \
        else (1.0, 1.0)
    prior = rc_v / sc - ra_v / sa
    mu0, s2, n0 = float(prior.mean()), float(prior.var(ddof=1)), float(len(prior))

    wc = _leg_weights(market, span, members[:, ls], config, device,
                      lookback, seed)
    wa = _leg_weights(market, span, anchors[:, ls], plain, device,
                      lookback, seed)
    y = np.asarray(market.array("targets")[span], np.float64)
    rc, ra = (wc * y).sum(1), (wa * y).sum(1)

    lam = np.empty(len(y))
    acc = count = 0.0
    cc, aa = (sc ** 2) * n0, (sa ** 2) * n0
    cn = an = n0
    for t in range(len(y)):
        weight = n0 / (1.0 + count / n0)
        precision = weight / max(s2, 1e-18) + count / max(s2, 1e-18)
        mean = ((weight / max(s2, 1e-18)) * mu0 + acc / max(s2, 1e-18)) / precision
        lam[t] = float(norm.cdf(mean * np.sqrt(precision)))
        if config.credibility == "sharpe":
            acc += rc[t] / np.sqrt(cc / cn) - ra[t] / np.sqrt(aa / an)
            cc += rc[t] ** 2; aa += ra[t] ** 2; cn += 1.0; an += 1.0
        else:
            acc += rc[t] - ra[t]
        count += 1.0

    covariance = np.asarray(market.array("covariances")[span], np.float64)
    diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
    corrected_signal, _ = aggregate(members[:, ls], diagonal, lookback,
                                     config.uncertainty, config.mean_precision)
    anchor_signal, _ = aggregate(anchors[:, ls], diagonal, lookback,
                                  config.uncertainty, config.mean_precision)
    blend = (lam[:, None] * np.maximum(corrected_signal, 0.0)
             + (1.0 - lam)[:, None] * np.maximum(anchor_signal, 0.0))
    face = credibility_face(wc, wa, lam, config.cardinality)
    weights, _ = portfolio_path(_Split(market, span), blend, config, device,
                                seed=seed, faces=face)
    if return_trace:
        trace = dict(lam=lam, face=face, signal=blend, w_corrected=wc, w_anchor=wa,
                     validation_corrected=rc_v, validation_anchor=ra_v,
                     corrected_signal=np.maximum(corrected_signal, 0),
                     anchor_signal=np.maximum(anchor_signal, 0))
        return weights, lam, trace
    return weights, lam


def portfolio_path(market, signal, config, device, *, seed=0, ranking=None,
                   faces=None):
    config.validate()
    if signal.shape != (market.n - market.test.start, market.N):
        raise ValueError("signal must cover exactly the declared test windows")
    if faces is not None:
        if config.optimizer != "alpha" or config.position != "long-only":
            raise ValueError("prescribed faces require exact long-only allocation")
        if (faces.shape != (len(signal), min(config.cardinality, market.N))
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= market.N
                or np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0)):
            raise ValueError("invalid prescribed asset face")
    covariance = market.array("covariances")[market.test]
    outputs = []
    with measured(device) as resources:
        for start in range(0, len(signal), chunk_size(market.N, config.batch_size)):
            sl = slice(
                start, min(start + chunk_size(market.N, config.batch_size), len(signal))
            )
            S = torch.tensor(
                np.asarray(covariance[sl]), device=device, dtype=torch.float64
            )
            mu = torch.tensor(
                np.ascontiguousarray(signal[sl]), device=device, dtype=torch.float64
            )
            d = (
                None
                if ranking is None
                else torch.tensor(
                    np.ascontiguousarray(ranking[sl]),
                    device=device,
                    dtype=torch.float64,
                )
            )
            if faces is None:
                weights = allocate(S, mu, config, seed, d)
            else:
                face = torch.tensor(faces[sl], device=device, dtype=torch.long)
                with torch.no_grad():
                    weights = exact_layer(S, torch.relu(mu), config.cardinality,
                                          face=face)[0]
            outputs.append(weights.cpu().numpy())
    weights = np.concatenate(outputs)
    if not np.isfinite(weights).all():
        raise RuntimeError("nonfinite portfolio weights")
    if config.position == "long-only" and (
        weights.min() < -1e-8 or not np.allclose(weights.sum(1), 1, atol=1e-7)
    ):
        raise RuntimeError("long-only feasibility check failed")
    if (np.abs(weights) > 1e-10).sum(1).max() > config.cardinality:
        raise RuntimeError("cardinality check failed")
    return weights, resources


def evaluate(
    dataset,
    config,
    output,
    device,
    *,
    forecasts=None,
    seeds=(0,),
    markets=None,
    ranking_cache=None,
):
    """Save ordinary weights, daily returns, metrics, input hashes, and timing."""
    config.validate()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    result = dict(
        dataset_identity=dataset.hash,
        deployment_version="fresh-paths-mean-signals-v1",
        implementation_sha256={name: file_hash(Path(__file__).with_name(name))
                               for name in ["evaluation.py", "selection.py", "numerics.py",
                                            "nnqp_pruned.py", "layers.py", "sdp.py"]},
        configuration=asdict(config),
        seeds=list(seeds),
        environment=environment(device),
        markets=[],
    )
    selected = (
        dataset.markets
        if markets is None
        else [dataset.by_name[name] for name in markets]
    )
    marker = root / "evaluation.json"
    previous = None
    if marker.exists():
        previous = json.loads(marker.read_text())
        for key in ["dataset_identity", "configuration", "seeds", "implementation_sha256"]:
            if previous.get(key) != result.get(key):
                raise ValueError(f"{marker}: incompatible completed evaluation ({key})")
        verify_evaluation(dataset, marker)
    old_markets = {} if previous is None else {m["name"]: m for m in previous["markets"]}
    result["markets"] = [row for name, row in old_markets.items()
                         if name not in {m.name for m in selected}]
    for market in selected:
        synchronize(device)
        market_started = time.perf_counter()
        covariance = market.array("covariances")[market.test]
        diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
        targets = market.array("targets")[market.test]
        base = (market.validation.start if config.credibility != "none"
                else market.test.start)
        if forecasts is None:
            members = np.stack([market.array("means")[base:]] * len(seeds))
            hashes = []
        else:
            members, hashes = load_members(dataset, market, forecasts, seeds,
                                           since=base)
        if config.aggregation in {"calibrated", "anchor-mean"}:
            if forecasts is None:
                raise ValueError(
                    "calibrated/anchor aggregation requires prediction files"
                )
            anchors, anchor_hashes = load_members(
                dataset, market, forecasts, seeds, anchor=True, since=base
            )
            members = (
                anchors
                if config.aggregation == "anchor-mean"
                else calibrate(anchors, members)
            )
        else:
            anchors, anchor_hashes = None, []
        if market.name in old_markets:
            old = old_markets[market.name]
            if old["predictions"] != hashes or old["anchors"] != anchor_hashes:
                raise ValueError(f"{market.name}: completed evaluation uses different forecasts")
            result["markets"].append(old)
            continue
        if config.credibility == "none":
            members = members[:, market.test.start - base:]
        groups = (
            [(str(seed), members[j : j + 1], seed) for j, seed in enumerate(seeds)]
            if config.aggregation == "individual"
            else [("ensemble", members, 0)]
        )
        rows = []
        for label, group, seed in groups:
            signal, ranking, selection_resources = None, None, None
            trace = None
            if config.credibility == "none":
                signal, _ = aggregate(
                    group,
                    diagonal,
                    dataset.lookback,
                    config.uncertainty,
                    config.mean_precision,
                )
                if config.optimizer == "sdp":
                    from .sdp import ensure_ranking
                    ranking, selection_resources = ensure_ranking(
                        ranking_cache or root / "sdp_cache", market, signal, config.cardinality,
                        tolerance=config.sdp_tolerance, max_iterations=config.sdp_iterations,
                        max_seconds=config.sdp_max_seconds)

            if config.credibility != "none":
                with measured(device) as resources:
                    weights, _, trace = credible_weights(
                        market, group, anchors, config, device,
                        dataset.lookback, base, seed=seed, return_trace=True)
            else:
                weights, resources = portfolio_path(
                    market, signal, config, device, seed=seed, ranking=ranking
                )
            weights_file = root / f"{market.name}_{label}_weights.npy"
            returns_file = root / f"{market.name}_{label}_returns.npy"
            np.save(weights_file, weights)
            np.save(returns_file, (weights * targets).sum(1))
            saved_paths = [weights_file, returns_file]
            if trace is not None:
                trace_file = root / f"{market.name}_credibility.npz"
                np.savez_compressed(trace_file, **trace)
                saved_paths.append(trace_file)
            rows.append(
                dict(
                    label=label,
                    metrics=metrics(weights, targets, config.annualization),
                    resources=resources,
                    selection_resources=selection_resources,
                    files_sha256={
                        p.name: file_hash(p) for p in saved_paths
                    },
                )
            )
        average = {
            key: (
                float(np.mean([row["metrics"][key] for row in rows]))
                if all(row["metrics"][key] is not None for row in rows)
                else None
            )
            for key in rows[0]["metrics"]
        }
        result["markets"].append(
            dict(
                name=market.name,
                predictions=hashes,
                anchors=anchor_hashes,
                mean=average,
                results=rows,
                end_to_end_resources=dict(
                    seconds=time.perf_counter() - market_started,
                    scope="forecast I/O, aggregation, allocation, metrics, and portfolio-file output",
                    gpu_peak_allocated_mib=max(
                        r["resources"]["gpu_peak_allocated_mib"] for r in rows
                    ),
                    gpu_peak_reserved_mib=max(
                        r["resources"]["gpu_peak_reserved_mib"] for r in rows
                    ),
                    cpu_process_peak_rss_mib=max(
                        r["resources"]["cpu_process_peak_rss_mib"] for r in rows
                    ),
                ),
            )
        )
        save_json(root / "evaluation.json", result)
        print(market.name, average, flush=True)
    save_json(root / "evaluation.json", result)
    return result


def verify_evaluation(dataset, evaluation_path):
    """Independently recompute every saved metric from hash-verified portfolio paths."""
    path = Path(evaluation_path)
    record = json.loads(path.read_text())
    if record["dataset_identity"] != dataset.hash:
        raise ValueError("dataset identity mismatch")
    checks = []
    for group in record["markets"]:
        market = dataset.by_name[group["name"]]
        targets = market.array("targets")[market.test]
        for row in group["results"]:
            for filename, digest in row["files_sha256"].items():
                if file_hash(path.parent / filename) != digest:
                    raise ValueError(f"changed path: {filename}")
            weightfile = next(
                name for name in row["files_sha256"] if name.endswith("_weights.npy")
            )
            weights = np.load(path.parent / weightfile)
            returnfile = next(
                name for name in row["files_sha256"] if name.endswith("_returns.npy")
            )
            if not np.array_equal(
                np.load(path.parent / returnfile), (weights * targets).sum(1)
            ):
                raise ValueError("saved daily returns do not match portfolio weights")
            actual = metrics(weights, targets, record["configuration"]["annualization"])
            for key, value in actual.items():
                if value is None:
                    if row["metrics"][key] is not None:
                        raise ValueError("undefined metric mismatch")
                elif not np.isclose(value, row["metrics"][key], atol=1e-12, rtol=1e-12):
                    raise ValueError(
                        f"metric mismatch: {market.name}/{row['label']}/{key}"
                    )
            checks.append(dict(market=market.name, label=row["label"], passed=True))
    return checks


def member_bootstrap(*args, **kwargs):
    """Resample paired member trajectories and recompute the complete decisions."""
    from .bootstrap import member_bootstrap as run
    return run(*args, **kwargs)
