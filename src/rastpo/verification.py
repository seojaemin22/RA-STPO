"""Read-only integrity checks for portable prepared inputs and evaluation artifacts."""

import pandas as pd

from .data import file_hash
from .evaluation import verify_evaluation


def verify_dataset(dataset):
    checks = []
    cutoff = dataset.manifest["common_validation_cutoff"]
    for market in dataset.markets:
        if (
            not 0
            < market.train_end
            < market.validation.stop
            <= market.test.start
            < market.n
        ):
            raise ValueError(f"invalid chronological boundaries: {market.name}")
        targets = pd.DatetimeIndex(market.metadata["target_dates"])
        inputs = pd.DatetimeIndex(market.metadata["input_dates"])
        if (
            len(targets) != market.n
            or not targets.is_monotonic_increasing
            or targets.has_duplicates
        ):
            raise ValueError(f"invalid target dates: {market.name}")
        if len(inputs) != market.n or not (inputs < targets).all():
            raise ValueError(f"input/target dates overlap: {market.name}")
        if (
            cutoff is not None
            and not (targets[market.validation] < pd.Timestamp(cutoff)).all()
        ):
            raise ValueError(
                f"pooled validation crosses the common cutoff: {market.name}"
            )
        for name, digest in market.metadata["arrays_sha256"].items():
            if file_hash(market.root / market.name / name) != digest:
                raise ValueError(f"prepared array was changed: {market.name}/{name}")
        checks.append(
            dict(
                market=market.name,
                arrays=len(market.metadata["arrays_sha256"]),
                chronology=True,
                hashes=True,
            )
        )
    return checks


def verify_artifacts(dataset, evaluations=()):
    return dict(
        dataset_identity=dataset.hash,
        dataset=verify_dataset(dataset),
        evaluations=[
            dict(path=str(path), portfolios=verify_evaluation(dataset, path))
            for path in evaluations
        ],
    )
