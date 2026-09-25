"""Conditional bootstrap of fitted members, with complete re-optimization."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .data import file_hash, save_json
from .evaluation import aggregate, calibrate, credible_weights, load_members, metrics, portfolio_path
from .runtime import environment, measured


def _contract(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"{path}: incompatible bootstrap inputs or settings")
    else:
        save_json(path, value)


def _sdp_rankings(market, members, indices, config, root, start, stop):
    from .sdp import RepeatedSDP, timed_sdp

    covariance = market.array("covariances")[market.test]
    result = []
    for day, risk in enumerate(covariance):
        path = root / f"date_{day}_{start}_{stop}.npz"
        marker = path.with_suffix(".json")
        if marker.exists():
            record = json.loads(marker.read_text())
            if file_hash(path) != record["sha256"]:
                raise ValueError(f"{path}: damaged bootstrap ranking")
            result.append(np.load(path)["ranking"])
            continue
        means = np.stack([
            members[index, day].astype(config.mean_precision).mean(0).astype(np.float64)
            for index in indices[start:stop]
        ])
        center = members[:, day].mean(0)
        solver = RepeatedSDP(risk, center, config.cardinality,
                             config.sdp_tolerance, config.sdp_iterations)
        ranks, records = [], []
        for replicate, signal in enumerate(means):
            try:
                rank, record = solver.solve(signal, warm=replicate > 0)
            except RuntimeError as error:
                begun = time.perf_counter()
                rank, _, value, status = timed_sdp(
                    signal, risk, config.cardinality, max_seconds=config.sdp_max_seconds,
                    eps=config.sdp_tolerance, max_iters=config.sdp_iterations,
                    acceleration_lookback=0,
                )
                rank = rank.astype(np.float32)
                record = dict(seconds=time.perf_counter()-begun, status=status,
                              objective=value, recovery=str(error))
            ranks.append(rank)
            records.append(record)
        np.savez_compressed(path, ranking=np.stack(ranks))
        save_json(marker, dict(sha256=file_hash(path), solves=records))
        result.append(np.stack(ranks))
        print(f"SD-relaxation bootstrap {market.name}: date {day + 1}/{len(covariance)}",
              flush=True)
    return np.stack(result, axis=1)


def member_bootstrap(dataset, config, output, device, *, forecasts, seeds,
                     replicates=100, resampling_seed=20260920, start=0, stop=None,
                     markets=None, ranking_cache=None, evaluation_path=None):
    """Resample forecasts (paired anchors/correctors for RA-STPO), never metrics."""
    config.validate()
    stop = replicates if stop is None else stop
    if forecasts is None or config.aggregation == "individual":
        raise ValueError("member bootstrap requires fitted, mean-aggregated forecasts")
    if len(set(seeds)) != len(seeds) or len(seeds) < 2 or not 0 <= start < stop <= replicates:
        raise ValueError("need at least two distinct members and a valid replicate range")
    if config.optimizer == "sdp" and (
        config.uncertainty != "none" or config.aggregation != "mean"
    ):
        raise ValueError("SD-relaxation bootstrap uses nominal mean forecasts")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    indices = np.random.default_rng(resampling_seed).integers(
        0, len(seeds), size=(replicates, len(seeds))
    )
    selected = dataset.markets if markets is None else [dataset.by_name[m] for m in markets]
    report = dict(dataset_identity=dataset.hash, configuration=asdict(config),
                  seeds=list(seeds), replicates=replicates, range=[start, stop],
                  resampling_seed=resampling_seed, environment=environment(device), markets=[])
    for market in selected:
        directory = root / market.name
        directory.mkdir(parents=True, exist_ok=True)
        base = market.validation.start if config.credibility != "none" else market.test.start
        members, hashes = load_members(dataset, market, forecasts, seeds, since=base)
        anchors, anchor_hashes = None, []
        if config.aggregation in {"calibrated", "anchor-mean"}:
            anchors, anchor_hashes = load_members(
                dataset, market, forecasts, seeds, anchor=True, since=base)
            members = anchors if config.aggregation == "anchor-mean" else calibrate(anchors, members)
        contract = dict(dataset_identity=dataset.hash, configuration=asdict(config),
                        seeds=list(seeds), resampling_seed=resampling_seed,
                        predictions=hashes, anchors=anchor_hashes,
                        implementation=file_hash(Path(__file__)),
                        evaluation_implementation=file_hash(Path(__file__).with_name("evaluation.py")))
        _contract(directory / "inputs.json", contract)
        covariance = market.array("covariances")[market.test]
        diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
        targets = market.array("targets")[market.test]
        rankings = None
        if config.optimizer == "sdp":
            rankings = _sdp_rankings(market, members, indices, config, directory, start, stop)
        rows = []
        for replicate in range(start, stop):
            marker = directory / f"replicate_{replicate}.json"
            sample = indices[replicate]
            index_hash = hashlib.sha256(sample.tobytes()).hexdigest()
            if marker.exists():
                row = json.loads(marker.read_text())
                if row["index_sha256"] != index_hash:
                    raise ValueError(f"{marker}: different resampled members")
            else:
                group = members[sample]
                with measured(device) as resources:
                    if config.credibility != "none":
                        weights, _ = credible_weights(
                            market, group, anchors[sample], config, device, dataset.lookback, base)
                    else:
                        signal, _ = aggregate(group, diagonal, dataset.lookback,
                                              config.uncertainty, config.mean_precision)
                        ranking = None if rankings is None else rankings[replicate - start]
                        weights, _ = portfolio_path(market, signal, config, device, ranking=ranking)
                row = dict(replicate=replicate, index_sha256=index_hash,
                           metrics=metrics(weights, targets, config.annualization),
                           resources=resources)
                save_json(marker, row)
            rows.append(row)
            if (replicate + 1) % 10 == 0 or replicate + 1 == stop:
                print(f"bootstrap {market.name}: {replicate + 1}/{stop}", flush=True)
        deviation = {}
        for key in rows[0]["metrics"]:
            values = [r["metrics"][key] for r in rows]
            deviation[key] = (float(np.std(values, ddof=1))
                              if len(values) > 1 and all(v is not None for v in values) else None)
        report["markets"].append(dict(name=market.name, results=rows, standard_deviation=deviation))
        save_json(root / f"resampling_{start}_{stop}.json", report)
    return report
