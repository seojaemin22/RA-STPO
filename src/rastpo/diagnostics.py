"""Greedy, final-face, and calibration diagnostics."""

from pathlib import Path
import numpy as np
import torch

from .data import file_hash, save_json
from .evaluation import (DecisionConfig, aggregate, calibrate, evaluate, load_members,
                         verify_evaluation)
from .selection import greedy_face_safe_draws, greedy_face_long_only, certificate_upper


def summary(values):
    values = np.asarray(values, float)
    return dict(min=float(values.min()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), mean=float(values.mean()),
                max=float(values.max()))


def diagnose(dataset, forecasts, seeds, output, device, *, cardinality=10,
             samples=24, evaluation=None, markets=None):
    if samples < 1:
        raise ValueError("samples must be positive")
    output = Path(output)
    if evaluation is None:
        directory = output.parent / "diagnostic_portfolios"
        evaluate(dataset, DecisionConfig(cardinality=cardinality, credibility="sharpe"),
                 directory, device, forecasts=forecasts, seeds=seeds, markets=markets)
        evaluation = directory / "evaluation.json"
    evaluation = Path(evaluation)
    verify_evaluation(dataset, evaluation)
    import json
    point = json.loads(evaluation.read_text())
    cfg = point["configuration"]
    if (cfg["cardinality"] != cardinality or point["seeds"] != list(seeds)
            or cfg["credibility"] != "sharpe" or cfg["aggregation"] != "calibrated"):
        raise ValueError("diagnostics require the matching complete RA-STPO evaluation")
    report = dict(dataset_identity=dataset.hash, cardinality=cardinality, seeds=list(seeds),
                  evaluation_sha256=file_hash(evaluation), markets=[])
    selected = dataset.markets if markets is None else [dataset.by_name[n] for n in markets]
    for market in selected:
        a, hashes = load_members(dataset, market, forecasts, seeds)
        m, anchor_hashes = load_members(dataset, market, forecasts, seeds, anchor=True)
        saved = next(row for row in point["markets"] if row["name"] == market.name)
        if hashes != saved["predictions"] or anchor_hashes != saved["anchors"]:
            raise ValueError("diagnostics and evaluation use different forecasts")
        covariance = market.array("covariances")[market.test]
        diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
        sigma = np.sqrt(diagonal / dataset.lookback)
        x = calibrate(m, a)
        delta = a - m
        centered = delta - delta.mean(-1, keepdims=True)
        displacement = np.linalg.norm(x - m, axis=-1)
        norm_bound = 2 * np.linalg.norm(centered, axis=-1)
        trust_bound = 2 * np.linalg.norm(sigma, axis=-1)[None]
        calibration = dict(
            before=summary(np.abs(delta) / sigma), after=summary(np.abs(x-m) / sigma),
            coordinate_box_exceeded=float(np.mean(np.abs(x-m) > sigma * (1+1e-5))),
            norm_bound_violations=int(np.sum(displacement > norm_bound + 1e-10 + 1e-9*norm_bound)),
            measured_bound_ratio=summary(displacement / np.maximum(trust_bound, 1e-300)),
        )
        indices = np.linspace(0, len(covariance)-1, min(samples, len(covariance)), dtype=int)
        corrected = aggregate(x, diagonal, dataset.lookback, "mean")[0]
        anchor = aggregate(m, diagonal, dataset.lookback, "mean")[0]
        signal = np.maximum(np.stack([corrected[indices], anchor[indices]], axis=1), 0)
        S = torch.tensor(np.asarray(covariance[indices]), device=device, dtype=torch.float64)
        z = torch.tensor(signal, device=device, dtype=torch.float64)
        face, mask, stats = greedy_face_safe_draws(S, z, cardinality, return_stats=True)
        components = []
        def supports(f, allowed):
            return [set(row[keep].cpu().tolist()) for row, keep in zip(f, allowed)]
        for j, label in enumerate(["corrected", "anchor"]):
            ref, ref_mask = greedy_face_long_only(S, z[:, j], cardinality)
            f32, m32 = greedy_face_safe_draws(S.float(), z[:, j:j+1].float(), cardinality)
            expected = supports(face[:, j], mask[:, j])
            upper, value, _, _ = certificate_upper(
                S, z[:, j], face[:, j], mask[:, j],
                delta=dataset.specification["covariance_shrinkage"])
            components.append(dict(
                component=label,
                solver_agreement=float(np.mean([a == b for a,b in zip(expected, supports(ref, ref_mask))])),
                precision_agreement=float(np.mean([a == b for a,b in zip(expected, supports(f32[:, 0], m32[:, 0]))])),
                value=value.cpu().tolist(), upper=upper.cpu().tolist(),
                ratio=(value / upper.clamp_min(1e-300)).cpu().tolist(),
            ))
        trace = np.load(evaluation.parent / f"{market.name}_credibility.npz")
        weights = np.load(evaluation.parent / f"{market.name}_ensemble_weights.npy")
        final_signal, final_face = trace["signal"], trace["face"]
        Sw = np.einsum("tij,tj->ti", covariance, weights)
        scale = (final_signal*weights).sum(1) / (weights*Sw).sum(1)
        v = weights * scale[:, None]
        g = np.einsum("tij,tj->ti", covariance, v) - final_signal
        gf, vf = [np.take_along_axis(q, final_face, axis=1) for q in [g,v]]
        residual = float(max(np.maximum(-gf,0).max(), np.abs(vf*gf).max()))
        upper, value, _, _ = certificate_upper(
            S, torch.tensor(final_signal[indices], device=device, dtype=torch.float64),
            torch.tensor(final_face[indices], device=device),
            torch.ones((len(indices), cardinality), device=device, dtype=torch.bool),
            delta=dataset.specification["covariance_shrinkage"])
        report["markets"].append(dict(
            name=market.name, dates=indices.tolist(), calibration=calibration,
            components=components, signals=2*len(indices), fallback_count=stats["fallback_count"],
            final_face=dict(kkt_residual=residual, value_upper_ratio=summary(
                (value / upper.clamp_min(1e-300)).cpu().numpy())),
        ))
        save_json(output, report)
        print(f"diagnostics {market.name}: final KKT residual {residual:.3g}", flush=True)
    return report
