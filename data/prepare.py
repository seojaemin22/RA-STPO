"""Preprocess adjusted prices into the window and raw-input datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

DATA_ROOT = Path(__file__).resolve().parent
ROOT = DATA_ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from rastpo import features
from rastpo.data import file_hash, save_json


def resolve_price_paths(prices: list[str | Path] | None = None,
                        raw_dir: Path = DATA_ROOT / "raw") -> list[Path]:
    """Use the bundled price snapshot by default, or resolve explicit input paths."""
    project_root = ROOT
    records = None
    if prices is None:
        manifest = raw_dir.expanduser().resolve() / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Price manifest not found in {manifest.parent}. "
                "Restore data/raw/ from the repository, or supply --prices PATH ..."
            )
        records = json.loads(manifest.read_text())["markets"]
        prices = [manifest.parent / record["file"] for record in records]
    if not prices:
        raise ValueError("provide at least one price panel")
    paths = []
    for value in prices:
        path = Path(value).expanduser()
        if not path.is_absolute() and not path.exists():
            candidate = project_root / path
            if candidate.is_file():
                path = candidate
        paths.append(path.resolve())
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Price files not found:\n  " + "\n  ".join(missing)
            + "\nSupply existing files with --prices, or omit --prices to use "
            "the bundled price snapshot."
        )
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("each input panel must have a distinct filename stem")
    if records is not None:
        for path, record in zip(paths, records):
            if file_hash(path) != record["sha256"]:
                raise ValueError(
                    f"Price data failed its checksum: {path}. "
                    "Restore the original file from the repository, "
                    "or use --prices to provide a custom dataset."
                )
    return paths


def standardize_assets(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-12
    )


def prepare(
    prices: list[str | Path] | None,
    output: str,
    *,
    representation: str = "window",
    lookback: int = 100,
    test_fraction: float = 0.2,
    validation_window: int = 280,
    shrinkage: float = 0.1,
    calendar_aligned: bool = True,
) -> dict:
    """Create a reusable cache from arbitrary dated price-parquet panels.

    The input list determines the market-embedding order; None selects the bundled
    price snapshot. Existing caches are accepted only when all input hashes and
    preparation settings match.
    """
    if representation not in {"window", "raw"}:
        raise ValueError("representation must be window or raw")
    if lookback < 2 or (representation == "window" and lookback < 64):
        raise ValueError("raw inputs need lookback >= 2; window summaries need >= 64")
    if not 0 < test_fraction < 1 or not 0 <= shrinkage <= 1 or validation_window < 1:
        raise ValueError("invalid split or covariance settings")
    paths = resolve_price_paths(prices)
    specification = dict(
        schema_version=2,
        representation=representation,
        lookback=lookback,
        test_fraction=test_fraction,
        validation_window=validation_window,
        covariance_shrinkage=shrinkage,
        calendar_aligned=calendar_aligned,
        inputs=[dict(name=p.stem, sha256=file_hash(p)) for p in paths],
    )
    root = Path(output)
    manifest_path = root / "dataset.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["specification"] != specification:
            raise ValueError("output contains a cache with different inputs/settings")
        for market in previous["markets"]:
            for name, digest in market["arrays_sha256"].items():
                if file_hash(root / market["name"] / name) != digest:
                    raise ValueError(
                        f"cache integrity check failed: {market['name']}/{name}"
                    )
        return previous
    pending = root / "preparing.json"
    if root.exists() and any(root.iterdir()):
        if not pending.exists() or json.loads(pending.read_text()) != specification:
            raise ValueError("output is nonempty without a matching preparation record")
    root.mkdir(parents=True, exist_ok=True)
    save_json(pending, specification)
    started = time.perf_counter()
    panels = []
    for path in paths:
        frame = pd.read_parquet(path)
        if (
            not isinstance(frame.index, pd.DatetimeIndex)
            or not frame.index.is_monotonic_increasing
        ):
            raise ValueError(f"{path}: require a chronologically sorted DatetimeIndex")
        if (
            frame.index.has_duplicates
            or frame.columns.has_duplicates
            or frame.isna().any().any()
        ):
            raise ValueError(f"{path}: duplicate labels or missing prices")
        if len(frame.columns) < 2:
            raise ValueError(
                f"{path}: cross-sectional predictors require at least two assets"
            )
        if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
            raise ValueError(f"{path}: prices must be positive and finite")
        returns = frame.pct_change(fill_method=None).dropna()
        count = len(returns) - lookback
        split = int(count * (1 - test_fraction))
        train_end = split - validation_window
        if train_end < 2 or count - split < 2:
            raise ValueError(f"{path}: insufficient history for the requested split")
        panels.append((path.stem, frame, returns, split, train_end))
    cutoff = min(item[2].index[lookback + item[3] - 1] for item in panels)
    markets = []
    for name, frame, returns, split, train_end in panels:
        started_market = time.perf_counter()
        target_dates = returns.index[lookback:]
        val_end = (
            min(split, int(target_dates.searchsorted(cutoff, side="left")))
            if calendar_aligned
            else split
        )
        if val_end <= train_end:
            raise ValueError(f"{name}: the common calendar leaves no validation period")
        directory = root / name
        directory.mkdir(exist_ok=True)
        values = returns.to_numpy(dtype=np.float64)
        windows = np.stack(
            [values[t - lookback : t] for t in range(lookback, len(values))]
        )
        means = windows.mean(axis=1)
        covariances = []
        for window in windows:
            empirical = np.atleast_2d(np.cov(window, rowvar=False))
            count = empirical.shape[0]
            covariances.append(
                (1 - shrinkage) * empirical
                + shrinkage * (np.trace(empirical) / count) * np.eye(count)
            )
        covariances = np.stack(covariances)
        targets = values[lookback:]
        if representation == "window":
            inputs = features.build(windows)
            feature_names = list(features.NAMES)
        else:
            lagged = windows.transpose(0, 2, 1)
            inputs = np.stack(
                [standardize_assets(lagged[:, :, j]) for j in range(lookback)], axis=2
            )
            feature_names = [f"return_lag_{lookback-j}" for j in range(lookback)]
        arrays = {
            "returns.npy": values,
            "targets.npy": targets,
            "means.npy": means,
            "covariances.npy": covariances,
            "features.npy": np.asarray(inputs, np.float32),
            "targets32.npy": targets.astype(np.float32),
            "standardized_targets.npy": standardize_assets(targets).astype(np.float32),
            "means32.npy": means.astype(np.float32),
            "covariances32.npy": covariances.astype(np.float32),
        }
        for filename, array in arrays.items():
            np.save(directory / filename, array, allow_pickle=False)
        market = dict(
            name=name,
            assets=[str(c) for c in frame.columns],
            windows=len(targets),
            train_end=train_end,
            validation_end=val_end,
            test_start=split,
            target_dates=[d.isoformat() for d in target_dates],
            input_dates=[d.isoformat() for d in returns.index[lookback - 1 : -1]],
            feature_names=feature_names,
            preparation_seconds=time.perf_counter() - started_market,
            arrays_sha256={
                filename: file_hash(directory / filename) for filename in arrays
            },
        )
        markets.append(market)
        print(
            f"prepared {name}: {len(targets)} windows, {len(frame.columns)} assets",
            flush=True,
        )
        del windows, covariances, arrays, inputs
    manifest = dict(
        specification=specification,
        markets=markets,
        common_validation_cutoff=cutoff.isoformat() if calendar_aligned else None,
        preparation_seconds=time.perf_counter() - started,
    )
    save_json(manifest_path, manifest)
    pending.unlink(missing_ok=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", nargs="+", help="Adjusted-price Parquet panels in market-embedding order; default: data/raw/")
    parser.add_argument("--raw-dir", type=Path, default=DATA_ROOT / "raw", help="Directory containing the price manifest")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "processed", help="Directory for window/ and raw/ datasets")
    parser.add_argument("--representation", choices=["window", "raw", "all"], default="all")
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-window", type=int, default=280)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.1)
    parser.add_argument("--calendar-aligned", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    try:
        prices = resolve_price_paths(args.prices, args.raw_dir)
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.error(str(error))
    representations = ["window", "raw"] if args.representation == "all" else [args.representation]
    for representation in representations:
        destination = args.output.expanduser().resolve() / representation
        prepare(prices, str(destination), representation=representation,
                lookback=args.lookback, test_fraction=args.test_fraction,
                validation_window=args.validation_window, shrinkage=args.covariance_shrinkage,
                calendar_aligned=args.calendar_aligned)
        print(f"{representation} dataset: {destination}", flush=True)


if __name__ == "__main__":
    main()
