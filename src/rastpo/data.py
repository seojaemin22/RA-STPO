"""Load prepared market datasets and record artifact checksums."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass
class Market:
    root: Path
    metadata: dict

    @property
    def name(self):
        return self.metadata["name"]

    @property
    def n(self):
        return self.metadata["windows"]

    @property
    def N(self):
        return len(self.metadata["assets"])

    @property
    def train_end(self):
        return self.metadata["train_end"]

    @property
    def validation(self):
        return slice(self.train_end, self.metadata["validation_end"])

    @property
    def test(self):
        return slice(self.metadata["test_start"], self.n)

    def array(self, name, mode="r"):
        return np.load(
            self.root / self.name / f"{name}.npy", mmap_mode=mode, allow_pickle=False
        )

    def tensors(self):
        names = (
            "features",
            "standardized_targets",
            "targets32",
            "means32",
            "covariances32",
        )
        return tuple(torch.from_numpy(self.array(name, mode="c")) for name in names)


class Dataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "dataset.json").read_text())
        self.hash = file_hash(self.root / "dataset.json")
        self.specification = self.manifest["specification"]
        self.markets = [
            Market(self.root, record) for record in self.manifest["markets"]
        ]
        self.by_name = {market.name: market for market in self.markets}

    @property
    def lookback(self):
        return self.specification["lookback"]
