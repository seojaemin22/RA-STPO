"""Download adjusted daily prices from Yahoo Finance for a fixed asset universe."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_panel(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)


def close_prices(frame, tickers, start, end):
    """Handle both multi-ticker and single-ticker yfinance responses."""
    if frame is None or frame.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers, dtype=float)
    if isinstance(frame.columns, pd.MultiIndex):
        levels = [i for i in range(frame.columns.nlevels)
                  if "Close" in frame.columns.get_level_values(i)]
        if not levels:
            raise ValueError("Yahoo response has no Close prices")
        close = frame.xs("Close", axis=1, level=levels[0])
    elif "Close" in frame.columns and len(tickers) == 1:
        close = frame[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raise ValueError("Unexpected Yahoo price columns")
    close = close.copy()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    if close.index.has_duplicates or close.columns.has_duplicates:
        raise ValueError("Yahoo response contains duplicate dates or tickers")
    close = close.reindex(columns=tickers).apply(pd.to_numeric, errors="coerce")
    close = close.where(np.isfinite(close) & (close > 0))
    return close.loc[(close.index >= start) & (close.index < end)].sort_index().dropna(how="all")


def download_market(market, output, *, start, end, fetch, version,
                    batch_size=40, retries=4, pause=2.0, timeout=30.0, refresh=False):
    name, tickers = market["name"], market["tickers"]
    coverage = float(market.get("min_coverage", 0.995))
    path = output / f"{name}.parquet"
    state = output / ".downloads" / name
    request_path = state / "request.json"
    partial = state / "prices.parquet"
    receipt = state / "panel.json"
    request = dict(name=name, tickers=tickers, start=start, end=end,
                   min_coverage=coverage, interval="1d", auto_adjust=True,
                   yfinance_version=version)
    if not refresh:
        if request_path.exists():
            if json.loads(request_path.read_text()) != request:
                raise ValueError(f"{name}: cached download uses different settings; use a new --output or --refresh")
        elif path.exists():
            raise ValueError(f"{path}: price file has no download record; use a new --output or --refresh")
        if receipt.exists() and path.exists():
            record = json.loads(receipt.read_text())
            if file_hash(path) != record["sha256"]:
                raise ValueError(f"{path}: checksum mismatch; use --refresh to download again")
            print(f"cached {name}: {record['assets']} assets", flush=True)
            return record
    state.mkdir(parents=True, exist_ok=True)
    if refresh:
        partial.unlink(missing_ok=True)
        receipt.unlink(missing_ok=True)
    write_json(request_path, request)
    panel = pd.read_parquet(partial) if partial.exists() else pd.DataFrame()
    available = {symbol for symbol in panel.columns if panel[symbol].notna().any()}
    missing = [symbol for symbol in tickers if symbol not in available]
    for offset in range(0, len(missing), batch_size):
        pending = missing[offset:offset + batch_size]
        for attempt in range(retries):
            if offset or attempt:
                time.sleep(min(30.0, pause * 2**attempt))
            try:
                frame = fetch(pending, start=start, end=end, interval="1d",
                              auto_adjust=True, back_adjust=False, repair=False,
                              progress=False, threads=4, group_by="column",
                              ignore_tz=True, timeout=timeout, multi_level_index=True)
                prices = close_prices(frame, pending, start, end)
            except Exception as error:
                print(f"{name}: request {attempt + 1}/{retries} failed: {error}", flush=True)
                continue
            received = [symbol for symbol in pending if prices[symbol].notna().any()]
            if received:
                panel = pd.concat([panel.drop(columns=received, errors="ignore"),
                                   prices[received]], axis=1).sort_index()
                panel = panel.reindex(columns=[symbol for symbol in tickers if symbol in panel.columns])
                write_panel(partial, panel)
                pending = [symbol for symbol in pending if symbol not in received]
            if not pending:
                break
        if pending:
            raise RuntimeError(
                f"{name}: no prices for {', '.join(pending)} after {retries} attempts. "
                "Rerun this command to retry; successful series are cached."
            )
        print(f"{name}: {len(panel.columns)}/{len(tickers)} assets downloaded", flush=True)
    panel = panel.reindex(columns=tickers).dropna(how="all")
    observed = panel.notna().mean()
    inadequate = observed[observed < coverage]
    if not inadequate.empty:
        details = ", ".join(f"{symbol} ({value:.1%})" for symbol, value in inadequate.items())
        raise RuntimeError(
            f"{name}: coverage below {coverage:.1%}: {details}. "
            "No assets were silently dropped. Use --refresh to retry the download."
        )
    panel = panel.ffill().dropna(how="any")
    if len(panel) < 3 or not np.isfinite(panel.to_numpy()).all() or (panel <= 0).any().any():
        raise RuntimeError(f"{name}: insufficient valid prices")
    write_panel(path, panel)
    record = dict(file=path.name, sha256=file_hash(path), assets=len(tickers),
                  observations=len(panel), first_date=str(panel.index[0].date()),
                  last_date=str(panel.index[-1].date()), min_coverage=coverage,
                  downloaded_at=datetime.now(timezone.utc).isoformat())
    write_json(receipt, record)
    print(f"saved {name}: {len(panel)} dates, {len(tickers)} assets", flush=True)
    return record


def load_universe(path, selected=None):
    universe = json.loads(path.read_text())
    names = []
    for market in universe["markets"]:
        name, tickers = market["name"], market["tickers"]
        if not name or not all(c.isalnum() or c in "_-" for c in name):
            raise ValueError("market names must contain only letters, digits, underscores, or hyphens")
        if not isinstance(tickers, list) or len(tickers) < 2:
            raise ValueError(f"{name}: provide a list with at least two Yahoo symbols")
        if any(not isinstance(t, str) or not t.strip() for t in tickers):
            raise ValueError(f"{name}: ticker symbols must be nonempty strings")
        if len(set(tickers)) != len(tickers):
            raise ValueError(f"{name}: provide at least two distinct Yahoo symbols")
        if not 0 < float(market.get("min_coverage", 0.995)) <= 1:
            raise ValueError(f"{name}: min_coverage must be in (0, 1]")
        names.append(name)
    if not names or len(set(names)) != len(names):
        raise ValueError("the universe must contain distinct market names")
    if selected is not None and (len(set(selected)) != len(selected) or set(selected) - set(names)):
        raise ValueError(f"choose distinct markets from: {', '.join(names)}")
    # Keep the configured market order, which defines the embedding indices.
    markets = [m for m in universe["markets"] if selected is None or m["name"] in selected]
    return universe, markets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=DATA_ROOT / "universe.json")
    parser.add_argument("--markets", nargs="+", help="Download selected markets in the configured order")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "raw")
    parser.add_argument("--start", help="Inclusive start date; default: universe setting")
    parser.add_argument("--end", help="Exclusive end date; default: universe setting")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--pause", type=float, default=2.0, help="Delay between request batches and retries (seconds)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout per Yahoo request (seconds)")
    parser.add_argument("--refresh", action="store_true", help="Replace cached downloads for the selected markets")
    args = parser.parse_args(argv)
    try:
        universe, markets = load_universe(args.universe.expanduser(), args.markets)
        start = date.fromisoformat(args.start or universe["start"]).isoformat()
        end = date.fromisoformat(args.end or universe["end"]).isoformat()
        if (start >= end or min(args.batch_size, args.retries) < 1
                or not np.isfinite([args.pause, args.timeout]).all()
                or args.pause < 0 or args.timeout <= 0):
            raise ValueError("invalid date range, batch size, retry count, pause, or timeout")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    try:
        import yfinance as yf
    except ImportError:
        parser.error("yfinance is required; run: python -m pip install -r requirements.txt")
    output = args.output.expanduser().resolve()
    yf.set_tz_cache_location(str(output / ".downloads" / "yfinance"))
    records = []
    try:
        for market in markets:
            records.append(download_market(
                market, output, start=start, end=end, fetch=yf.download, version=yf.__version__,
                batch_size=args.batch_size, retries=args.retries, pause=args.pause,
                timeout=args.timeout, refresh=args.refresh))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Download failed: {error}\n")
    write_json(output / "manifest.json", dict(
        description="Adjusted daily closing prices downloaded from Yahoo Finance.",
        source="Yahoo Finance", yfinance_version=yf.__version__, start=start, end=end,
        auto_adjust=True, universe_sha256=file_hash(args.universe.expanduser()), markets=records))
    print(f"Download complete: {output}\nNext: python data/prepare.py --raw-dir \"{output}\"", flush=True)


if __name__ == "__main__":
    main()
