"""Resident-input allocation latency and working-memory measurements."""

from dataclasses import asdict

import numpy as np
import torch

from .data import save_json
from .evaluation import aggregate, allocate, calibrate, load_members, credible_weights, credibility_face
from .runtime import environment, measured, synchronize


@torch.no_grad()
def benchmark(
    dataset,
    config,
    output,
    device,
    *,
    forecasts=None,
    seeds=(0,),
    markets=None,
    window_index=0,
    warmup=3,
    repetitions=10,
    sdp_tolerance=1e-4,
    sdp_iterations=100000
):
    """Benchmark one supplied window per market, without input I/O or inference.

    SDP measurements always perform fresh solves. There is no ranking-cache lookup.
    CPU RSS is the process lifetime maximum, not an incremental allocation estimate.
    """
    config.validate()
    if config.credibility != "none":
        return benchmark_policy(dataset, config, output, device, forecasts=forecasts,
                                seeds=seeds, markets=markets, window_index=window_index,
                                warmup=warmup, repetitions=repetitions)
    if warmup < 0 or repetitions < 1 or window_index < 0 or not seeds:
        raise ValueError("invalid benchmark counts")
    selected = (
        dataset.markets
        if markets is None
        else [dataset.by_name[name] for name in markets]
    )
    report = dict(
        dataset_identity=dataset.hash,
        configuration=asdict(config),
        seeds=list(seeds),
        environment=environment(device),
        window_index=window_index,
        warmup=warmup,
        repetitions=repetitions,
        scope="resident covariance and post-aggregation signal to one allocated portfolio",
        markets=[],
    )
    for market in selected:
        if window_index >= market.n - market.test.start:
            raise ValueError("benchmark window lies outside the test block")
        covariance = np.asarray(
            market.array("covariances")[market.test.start + window_index]
        )
        if forecasts is None:
            members = np.stack(
                [market.array("means")[market.test.start + window_index]] * len(seeds)
            )[:, None]
            hashes, anchor_hashes = [], []
        else:
            members, hashes = load_members(dataset, market, forecasts, seeds)
            members = members[:, window_index : window_index + 1]
            anchor_hashes = []
        if config.aggregation in {"calibrated", "anchor-mean"}:
            if forecasts is None:
                raise ValueError("calibrated benchmarks need anchored forecasts")
            anchors, anchor_hashes = load_members(
                dataset, market, forecasts, seeds, anchor=True
            )
            anchors = anchors[:, window_index : window_index + 1]
            members = (
                anchors
                if config.aggregation == "anchor-mean"
                else calibrate(anchors, members)
            )
        groups = (
            [(str(seed), members[j : j + 1], seed) for j, seed in enumerate(seeds)]
            if config.aggregation == "individual"
            else [("ensemble", members, 0)]
        )
        rows = []
        for label, group, seed in groups:
            signal, _ = aggregate(
                group,
                covariance.diagonal()[None],
                dataset.lookback,
                config.uncertainty,
                config.mean_precision,
            )
            S = torch.tensor(covariance[None], dtype=torch.float64, device=device)
            mu = torch.tensor(signal, dtype=torch.float64, device=device)

            def decision():
                ranking = None
                if config.optimizer == "sdp":
                    from .sdp import timed_sdp

                    diagonal, _, _, _ = timed_sdp(
                        signal[0],
                        covariance,
                        config.cardinality,
                        eps=sdp_tolerance,
                        max_iters=sdp_iterations,
                        max_seconds=config.sdp_max_seconds,
                    )
                    # Match the numerical ranking interchange precision.
                    ranking = torch.tensor(
                        diagonal.astype(np.float32)[None],
                        dtype=torch.float64,
                        device=device,
                    )
                return allocate(S, mu, config, seed, ranking)

            for _ in range(warmup):
                decision()
            synchronize(device)
            baseline = (
                torch.cuda.memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            )
            timings = []
            for _ in range(repetitions):
                with measured(device) as timing:
                    weights = decision()
                timings.append(
                    dict(
                        timing,
                        gpu_incremental_peak_mib=max(
                            0.0, timing["gpu_peak_allocated_mib"] - baseline
                        ),
                    )
                )
                del weights
            rows.append(
                dict(
                    label=label,
                    measurements=timings,
                    median_seconds=float(
                        np.median([entry["seconds"] for entry in timings])
                    ),
                    max_incremental_gpu_mib=max(
                        entry["gpu_incremental_peak_mib"] for entry in timings
                    ),
                )
            )
        report["markets"].append(
            dict(
                name=market.name,
                predictions=hashes,
                anchors=anchor_hashes,
                results=rows,
            )
        )
        save_json(output, report)
    return report


@torch.no_grad()
def benchmark_policy(dataset, config, output, device, *, forecasts, seeds, markets=None,
                     window_index=0, warmup=3, repetitions=10):
    """Time calibration through final allocation from resident forecasts and state."""
    from scipy.special import ndtr
    from .layers import exact_layer

    if warmup < 0 or repetitions < 1 or window_index < 0 or forecasts is None:
        raise ValueError("invalid full-policy benchmark settings")
    selected = dataset.markets if markets is None else [dataset.by_name[n] for n in markets]
    report = dict(configuration=asdict(config), dataset_identity=dataset.hash,
                  seeds=list(seeds), environment=environment(device), warmup=warmup,
                  repetitions=repetitions, window_index=window_index,
                  scope="resident forecasts, covariance and past state through final allocation",
                  markets=[])
    for market in selected:
        base = market.validation.start
        a, hashes = load_members(dataset, market, forecasts, seeds, since=base)
        m, anchor_hashes = load_members(dataset, market, forecasts, seeds, anchor=True, since=base)
        x = m if config.aggregation == "anchor-mean" else calibrate(m, a)
        expected, _, trace = credible_weights(
            market, x, m, config, device, dataset.lookback, base, return_trace=True)
        if window_index >= len(expected):
            raise ValueError("benchmark window lies outside test block")
        rc_v, ra_v = trace["validation_corrected"], trace["validation_anchor"]
        sc, sa = max(rc_v.std(ddof=1), 1e-12), max(ra_v.std(ddof=1), 1e-12)
        prior = rc_v/sc - ra_v/sa
        n0, mu0, s2 = len(prior), float(prior.mean()), max(float(prior.var(ddof=1)), 1e-18)
        y = market.array("targets")[market.test]
        rc = (trace["w_corrected"]*y).sum(1)
        ra = (trace["w_anchor"]*y).sum(1)
        acc, cc, aa = 0.0, n0*sc**2, n0*sa**2
        for t in range(window_index):
            acc += rc[t]/np.sqrt(cc/(n0+t)) - ra[t]/np.sqrt(aa/(n0+t))
            cc += rc[t]**2
            aa += ra[t]**2
        offset = market.test.start-base+window_index
        a, m = a[:, offset:offset+1], m[:, offset:offset+1]
        covariance = np.asarray(market.array("covariances")[
            market.test.start+window_index:market.test.start+window_index+1])
        diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
        S = torch.tensor(covariance, device=device, dtype=torch.float64)

        def decision():
            x = m if config.aggregation == "anchor-mean" else calibrate(m, a)
            c = aggregate(x, diagonal, dataset.lookback, config.uncertainty, config.mean_precision)[0]
            b = aggregate(m, diagonal, dataset.lookback, config.uncertainty, config.mean_precision)[0]
            wc = allocate(S, torch.tensor(c, device=device, dtype=torch.float64), config).cpu().numpy()
            wa = allocate(S, torch.tensor(b, device=device, dtype=torch.float64), config).cpu().numpy()
            prior_count = n0/(1+window_index/n0)
            precision = (prior_count+window_index)/s2
            mean = (prior_count*mu0+acc)/(prior_count+window_index)
            lam = float(ndtr(mean*np.sqrt(precision)))
            face = credibility_face(wc, wa, np.array([lam]), config.cardinality)
            signal = lam*np.maximum(c,0)+(1-lam)*np.maximum(b,0)
            return exact_layer(S, torch.tensor(signal, device=device, dtype=torch.float64),
                               config.cardinality,
                               face=torch.tensor(face, device=device, dtype=torch.long))[0].cpu().numpy()

        np.testing.assert_allclose(decision(), expected[window_index:window_index+1],
                                   atol=1e-10, rtol=1e-9)
        for _ in range(warmup):
            decision()
        synchronize(device)
        baseline = torch.cuda.memory_allocated(device)/1024**2 if device.type == "cuda" else 0
        timings = []
        for _ in range(repetitions):
            with measured(device) as timing:
                decision()
            timing["gpu_incremental_peak_mib"] = max(0, timing["gpu_peak_allocated_mib"]-baseline)
            timings.append(timing)
        report["markets"].append(dict(
            name=market.name, predictions=hashes, anchors=anchor_hashes,
            results=[dict(label="ensemble", measurements=timings,
                          median_seconds=float(np.median([t["seconds"] for t in timings])),
                          max_incremental_gpu_mib=max(t["gpu_incremental_peak_mib"] for t in timings))],
        ))
        save_json(output, report)
    return report
