"""Train and evaluate sparse portfolio models and run the paper's experiments."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from rastpo.data import Dataset, save_json
from rastpo.evaluation import DecisionConfig
from rastpo.runtime import device_from_name
from rastpo.training import TrainConfig


def execution_arguments(p, *, device="auto"):
    p.add_argument("--device", default=device)
    p.add_argument("--threads", type=int, default=2)


def allocation_arguments(p):
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--forecasts", help="Completed fits; omit for historical means")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--markets", nargs="+")
    p.add_argument("--optimizer", choices=["alpha","oscar","dfstpo","pga","afba","sdp"], default="alpha")
    p.add_argument("--cardinality", type=int, default=10)
    p.add_argument("--aggregation", choices=["individual","mean","calibrated","anchor-mean"], default="mean")
    p.add_argument("--uncertainty", choices=["none","mean"], default="none")
    p.add_argument("--credibility", choices=["none","sharpe"], default="none")
    p.add_argument("--position", choices=["long-only","long-short"], default="long-only")
    p.add_argument("--ordering", choices=["stored","seeded","stream"], default="stored")
    p.add_argument("--mean-precision", choices=["float32","float64"], default="float64")
    p.add_argument("--annualization", type=float, default=252.)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--pga-ridge", type=float, default=1e-3)
    p.add_argument("--pga-tolerance", type=float, default=1e-6)
    p.add_argument("--pga-iterations", type=int, default=3000)
    p.add_argument("--sdp-tolerance", type=float, default=1e-4)
    p.add_argument("--sdp-iterations", type=int, default=100000)
    p.add_argument("--sdp-max-seconds", type=float, default=600., help="Per cold SDP attempt; 0 removes the time limit")
    p.add_argument("--ranking-cache")
    execution_arguments(p)


def parser():
    root = argparse.ArgumentParser(
        description="Regularization-Aligned Sparse Tangent Portfolio Optimization",
        epilog="Use python run.py COMMAND --help for individual experiment options.",
    )
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train", help="Train PFL, DF-STPO, or a RA-STPO corrector")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--objective", choices=["prediction","soft","hard"], default="hard")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--cardinality", type=int, default=10)
    p.add_argument("--faces", type=int, default=8, help="Fresh independent faces per epoch, including J=1")
    p.add_argument("--hidden", type=int, nargs=2, help="Default: 64 32 for window features, 512 256 for raw")
    p.add_argument("--dropout", type=float)
    p.add_argument("--embedding", type=int, help="Default: max(4, ceil(input_features/2.75))")
    p.add_argument("--initialization", type=float, default=.01)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, help="Default: 128 for correction, 64 otherwise")
    p.add_argument("--anchor-batch-size", type=int, default=64)
    p.add_argument("--proposal-batch-size", type=int, default=256)
    p.add_argument("--face-solver", choices=["certified","reference"], default="certified")
    p.add_argument("--loss-floor", type=float, default=1e-8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--gradient-clip", type=float, default=5.)
    p.add_argument("--decision-weight", type=float, default=.5)
    p.add_argument("--validation-cache", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resident-inputs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--anchor-cache", help="Reusable PFL fits with matching inputs and configuration")
    execution_arguments(p)
    for name, help_text in [
        ("evaluate","Save portfolio paths, metrics, and resources"),
        ("rank","Compute SD-relaxation scores for the specified forecast aggregation"),
        ("resample","Bootstrap fitted members and re-optimize every resampled mean"),
        ("benchmark","Measure resident-input allocation latency, including full RA-STPO"),
    ]:
        p = commands.add_parser(name, help=help_text)
        allocation_arguments(p)
        if name == "rank":
            p.set_defaults(optimizer="sdp")
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--stop", type=int)
            p.add_argument("--stride", type=int, default=1)
        if name == "resample":
            p.add_argument("--replicates", type=int, default=100)
            p.add_argument("--resampling-seed", type=int, default=20260920)
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--stop", type=int)
        if name == "benchmark":
            p.add_argument("--window-index", type=int, default=0)
            p.add_argument("--warmup", type=int, default=3)
            p.add_argument("--repetitions", type=int, default=10)
    p = commands.add_parser("diagnose", help="Greedy, final-face KKT, and calibration diagnostics")
    p.add_argument("--data", required=True)
    p.add_argument("--forecasts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--evaluation", help="Matching complete RA-STPO evaluation.json")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--markets", nargs="+")
    p.add_argument("--cardinality", type=int, default=10)
    p.add_argument("--samples", type=int, default=24)
    execution_arguments(p)
    p = commands.add_parser("verify", help="Verify data hashes and recompute saved portfolio metrics")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--evaluations", nargs="*", default=[])
    execution_arguments(p, device="cpu")
    p = commands.add_parser("certify", help="Run the rational-arithmetic certifier in Appendix B")
    p.add_argument("--returns", required=True, help="T by N NumPy return array")
    p.add_argument("--output", required=True)
    p.add_argument("--cardinality", type=int, default=10)
    p.add_argument("--max-nodes", type=int, default=200)
    p.add_argument("--max-seconds", type=float, default=60.)
    p.add_argument("--covariance-shrinkage", default="0")
    execution_arguments(p, device="cpu")
    p = commands.add_parser("summarize", help="Regenerate CSV and TeX tables from reproduction outputs")
    p.add_argument("--output", required=True, help="Root containing the experiment results")
    p.add_argument("--data", default=str(ROOT / "data/processed/window"))
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--baseline-members", type=int, default=32)
    p.add_argument("--faces", type=int, default=8)
    p.add_argument("--bootstrap-replicates", type=int, default=100)
    p.add_argument("--sensitivity-members", type=int, nargs="+", default=[1,4,8,16,32])
    p.add_argument("--face-counts", type=int, nargs="+", default=[1,2,4,8,16])
    p.add_argument("--benchmark-markets", nargs="+")
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    if args.command == "summarize":
        from rastpo.reporting import summarize
        return summarize(args.output, args.data, members=args.members,
                         baseline_members=args.baseline_members, faces=args.faces,
                         bootstrap_replicates=args.bootstrap_replicates,
                         sensitivity_members=args.sensitivity_members, face_counts=args.face_counts,
                         benchmark_markets=args.benchmark_markets)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    device = device_from_name(args.device)
    if args.command == "train":
        from rastpo.training import train_member
        dataset = Dataset(args.data)
        raw = dataset.specification["representation"] == "raw"
        args.hidden = tuple(args.hidden or ([512,256] if raw else [64,32]))
        args.dropout = (0. if raw else .1) if args.dropout is None else args.dropout
        args.batch_size = args.batch_size or (128 if args.objective == "hard" else 64)
        config = TrainConfig(**{name:getattr(args,name) for name in TrainConfig.__dataclass_fields__})
        if config.objective == "prediction":
            # A direct PFL fit and a subsequently reused anchor have one identity.
            config = TrainConfig(**{**asdict(config), "anchor_batch_size":config.batch_size}).anchor()
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("training seeds must be distinct")
        tensors = [m.tensors() for m in dataset.markets]
        for seed in args.seeds:
            train_member(dataset, config, seed, args.output, device, tensors=tensors,
                         anchor_root=args.anchor_cache, resident_inputs=args.resident_inputs)
        return
    if args.command in {"evaluate","rank","resample","benchmark"}:
        from rastpo.evaluation import evaluate, load_members, aggregate, member_bootstrap
        config = DecisionConfig(**{name:getattr(args,name) for name in DecisionConfig.__dataclass_fields__})
        config.validate()
        dataset = Dataset(args.data)
        common = dict(forecasts=args.forecasts, seeds=args.seeds, markets=args.markets)
        if args.command == "evaluate":
            return evaluate(dataset, config, args.output, device, ranking_cache=args.ranking_cache, **common)
        if args.command == "resample":
            return member_bootstrap(dataset, config, args.output, device, **common,
                                    replicates=args.replicates, resampling_seed=args.resampling_seed,
                                    start=args.start, stop=args.stop)
        if args.command == "benchmark":
            from rastpo.benchmark import benchmark
            return benchmark(dataset, config, args.output, device, **common,
                             window_index=args.window_index, warmup=args.warmup,
                             repetitions=args.repetitions, sdp_tolerance=args.sdp_tolerance,
                             sdp_iterations=args.sdp_iterations)
        if config.credibility != "none" or config.aggregation not in {"mean","individual"}:
            raise ValueError("rank accepts nominal mean or individual forecasts")
        if args.start < 0 or args.stride < 1:
            raise ValueError("invalid ranking range")
        from rastpo.sdp import solve_cached
        import numpy as np
        selected = dataset.markets if args.markets is None else [dataset.by_name[n] for n in args.markets]
        for market in selected:
            members = (market.array("means")[market.test][None] if args.forecasts is None
                       else load_members(dataset, market, args.forecasts, args.seeds)[0])
            covariance = market.array("covariances")[market.test]
            groups = [members] if config.aggregation == "mean" else [m[None] for m in members]
            for group in groups:
                signal = aggregate(group, np.diagonal(covariance,axis1=-2,axis2=-1),
                                   dataset.lookback, config.uncertainty, config.mean_precision)[0]
                stop = len(signal) if args.stop is None else min(args.stop,len(signal))
                for day in range(args.start, stop, args.stride):
                    solve_cached(signal[day], covariance[day], config.cardinality, args.output,
                                 tolerance=config.sdp_tolerance, max_iterations=config.sdp_iterations,
                                 max_seconds=config.sdp_max_seconds)
                    print(f"SD-relaxation {market.name}: {day+1}/{stop}", flush=True)
        return
    if args.command == "diagnose":
        from rastpo.diagnostics import diagnose
        return diagnose(Dataset(args.data),args.forecasts,args.seeds,args.output,device,
                        cardinality=args.cardinality,samples=args.samples,
                        evaluation=args.evaluation,markets=args.markets)
    if args.command == "verify":
        from rastpo.verification import verify_artifacts
        result = verify_artifacts(Dataset(args.data), args.evaluations)
        save_json(args.output,result)
        return result
    if args.command == "certify":
        import math
        import numpy as np
        from fractions import Fraction
        from rastpo.certification import build_exact_model, certify, SaferConfig
        from rastpo.certificate_verifier import verify
        returns = np.load(args.returns,allow_pickle=False)
        model = build_exact_model(returns)
        if Fraction(args.covariance_shrinkage):
            model = model.shrink_covariance(Fraction(args.covariance_shrinkage),toward="IDENTITY")
        certificate = certify(returns,SaferConfig(cardinality=args.cardinality,
                              max_nodes=args.max_nodes,max_seconds=args.max_seconds),model=model)
        record = certificate.as_record()
        verdict = verify(returns,record,args.cardinality)
        def encode(v):
            if isinstance(v,float) and not math.isfinite(v): return str(v)
            if isinstance(v,dict): return {k:encode(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)): return [encode(x) for x in v]
            return v
        save_json(args.output,encode(dict(certificate=record,verification=verdict.__dict__)))


if __name__ == "__main__":
    main()
