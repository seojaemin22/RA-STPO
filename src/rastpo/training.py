"""Pooled prediction, soft-decision, and anchored hard-face training.

Every member has its own initialization, dropout stream, optimizer, minibatch
permutations, and early stopping. Neither ensemble size nor test returns enter a fit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .data import Dataset, file_hash, save_json
from .layers import exact_layer, oscar_soft
from .models import Conditioned, rank_ic, to_returns
from .numerics import asset_order
from .faces import FacePool
from .runtime import environment, measured, synchronize
from .selection import greedy_face_safe_draws


@dataclass(frozen=True)
class TrainConfig:
    objective: str = "hard"
    cardinality: int = 10
    faces: int = 8
    hidden: tuple[int, int] = (64, 32)
    dropout: float = 0.1
    embedding: int | None = None
    initialization: float = 0.01
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    anchor_batch_size: int = 64
    proposal_batch_size: int = 256
    face_solver: str = "certified"
    loss_floor: float = 1e-8
    epochs: int = 40
    patience: int = 8
    gradient_clip: float = 5.0
    decision_weight: float = 0.5
    validation_cache: bool = False

    def validate(self):
        if self.objective not in {"prediction", "soft", "hard"}:
            raise ValueError("unknown training objective")
        if self.face_solver not in {"reference", "certified"}:
            raise ValueError("unknown face solver")
        if self.anchor_batch_size < 2 or self.proposal_batch_size < 1 or self.loss_floor <= 0:
            raise ValueError("invalid anchor batch, proposal batch, or loss floor")
        if (
            min(
                self.faces,
                self.cardinality,
                self.epochs,
                self.patience,
                self.batch_size,
            )
            < 1
        ):
            raise ValueError("counts must be positive")
        if self.batch_size < 2 or not 0 <= self.dropout < 1:
            raise ValueError("invalid batch size or dropout")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.gradient_clip <= 0
            or not 0 <= self.decision_weight <= 1
            or min(self.hidden) < 1
            or (self.embedding is not None and self.embedding < 1)
        ):
            raise ValueError("invalid optimizer or model settings")

    def anchor(self):
        return replace(
            self,
            objective="prediction",
            faces=1,
            batch_size=self.anchor_batch_size,
            proposal_batch_size=256,
            face_solver="certified",
            loss_floor=1e-8,
            validation_cache=False,
        )


def identity(dataset: Dataset, config: TrainConfig, seed: int) -> str:
    payload = dict(dataset=dataset.hash, configuration=asdict(config), seed=int(seed))
    if config.objective == "soft":
        payload["decision_layer"] = "dfstpo_eq5_algorithm1_raw_mse_v1"
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def make_model(dataset, tensors, config, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Conditioned(
        tensors[0][0].shape[-1],
        len(dataset.markets),
        *config.hidden,
        config.dropout,
        config.initialization,
        dim=config.embedding,
    ).to(device)
    model.log_scale = nn.Parameter(
        torch.zeros((), device=device), requires_grad=config.objective == "prediction"
    )
    model.forecast_objective = config.objective
    return model


def mean_signal(model, score, mean, sigma, lookback, anchor=None):
    if anchor is not None:
        return anchor + torch.sqrt(sigma / float(lookback)) * torch.tanh(score)
    if getattr(model, "forecast_objective", None) == "soft":
        return score
    return to_returns(score, mean, "anchored", torch.exp(model.log_scale))


def fit(dataset, tensors, config, seed, device, anchors=None, *, resident_inputs=False):
    """Fit one independent member and return its selected model and diagnostics."""
    config.validate()
    if config.cardinality > min(m.N for m in dataset.markets):
        raise ValueError("cardinality exceeds the smallest market")
    if (config.objective == "hard") != (anchors is not None):
        raise ValueError(
            "hard training requires frozen anchors; other objectives do not"
        )
    model = make_model(dataset, tensors, config, seed, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    stages = {}
    if resident_inputs:
        with measured(device, reset_peak=False) as stages["input_transfer"]:
            tensors = [
                tuple(t.to(device) if index < 4 else t for index, t in enumerate(row))
                for row in tensors
            ]
            anchors = None if anchors is None else [t.to(device) for t in anchors]
    pool = (FacePool(dataset, tensors, anchors, config, seed, device)
            if config.objective == "hard" else None)
    diagonals = [torch.diagonal(t[4], dim1=-2, dim2=-1).clone() for t in tensors]
    permutations = [asset_order(m.N, seed) for m in dataset.markets]
    validation_tensors = []
    if config.validation_cache:
        for market, items in zip(dataset.markets, tensors):
            validation_tensors.append(
                tuple(t[market.validation].to(device) for t in items)
            )

    @torch.no_grad()
    def validation():
        model.eval()
        scores = []
        for mi, market in enumerate(dataset.markets):
            sl = market.validation
            if validation_tensors:
                X, _, Yr, MU, S = validation_tensors[mi]
            else:
                X, _, Yr, MU, S = (t[sl] for t in tensors[mi])
            g = model(X.to(device), mi)
            base = None if anchors is None else anchors[mi][sl].to(device)
            mu = mean_signal(
                model,
                g,
                MU.to(device),
                diagonals[mi][sl].to(device),
                dataset.lookback,
                base,
            )
            if config.objective != "hard":
                scores.append(rank_ic(mu.cpu().numpy(), market.array("targets")[sl]))
            else:
                S = S.to(device)
                f, mask = greedy_face_safe_draws(
                    S, torch.relu(mu)[:, None], config.cardinality
                )
                w, _ = exact_layer(
                    S, mu, config.cardinality, face=f[:, 0], face_mask=mask[:, 0]
                )
                r = (Yr.to(device) * w).sum(-1)
                scores.append(float(r.mean() / r.std().clamp_min(1e-12)))
        finite = [v for v in scores if np.isfinite(v)]
        return float(np.mean(finite)) if finite else -1e18

    with measured(device, reset_peak=False) as stages["initial_validation"]:
        best = validation()
    initial = best
    best_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    best_epoch, bad, history = 0, 0, []
    batches = []
    for mi, market in enumerate(dataset.markets):
        starts = list(range(0, market.train_end, config.batch_size))
        if len(starts) > 1 and market.train_end - starts[-1] == 1:
            starts.pop()
        for index, start in enumerate(starts):
            stop = starts[index + 1] if index + 1 < len(starts) else market.train_end
            batches.append((mi, start, stop))
    for epoch in range(config.epochs):
        synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        order = np.random.permutation(len(batches))
        refresh = pool.refresh(epoch) if pool is not None else None
        for bi in order:
            mi, start, stop = batches[bi]
            market = dataset.markets[mi]
            X, Yz, Yr, MU, SIG = tensors[mi]
            b = torch.arange(start, stop)
            g = model(X[b].to(device), mi)
            lmse = ((g - Yz[b].to(device)) ** 2).mean()
            if config.objective == "prediction":
                loss = lmse
            else:
                base = None if anchors is None else anchors[mi][b].to(device)
                mu = mean_signal(
                    model,
                    g,
                    MU[b].to(device),
                    diagonals[mi][b].to(device),
                    dataset.lookback,
                    base,
                )
                if config.objective == "soft":
                    w = oscar_soft(
                        mu.double(), SIG[b].to(device, dtype=torch.float64),
                        config.cardinality, permutations[mi]
                    )
                    realized = Yr[b].to(device, dtype=torch.float64)
                    r = (realized * w).sum(-1)
                    # Published Eq. (14): squared Euclidean error per example,
                    # averaged over examples, in the same return units as mu.
                    lmse = ((mu.double() - realized) ** 2).sum(-1).mean()
                    loss = (
                        config.decision_weight * (-r.mean())
                        + (1 - config.decision_weight) * lmse
                    )
                else:
                    loss, _ = pool.loss(mi, b.to(device), mu, Yr[b].to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        synchronize(device)
        optimization_seconds = time.perf_counter() - epoch_start
        with measured(device, reset_peak=False) as val_time:
            score = validation()
        history.append(
            dict(
                epoch=epoch + 1,
                validation_score=score,
                optimization_seconds=optimization_seconds,
                validation_seconds=val_time["seconds"],
                face_refresh=refresh,
            )
        )
        print(
            f"seed={seed} objective={config.objective} faces={config.faces} "
            f"epoch={epoch+1} validation={score:.10g}",
            flush=True,
        )
        if score > best:
            best, bad, best_epoch = score, 0, epoch + 1
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(
        initial_validation=initial,
        best_validation=best,
        best_epoch=best_epoch,
        epochs=len(history),
        history=history,
        stages=stages,
        face_sampling="fresh_each_epoch" if pool is not None else None,
        loss_reduction="mean_path_sharpe_losses" if pool is not None else None,
    )


@torch.no_grad()
def predict(dataset, tensors, model, device, anchors=None, batch_size=128):
    model.eval()
    result = []
    for mi, market in enumerate(dataset.markets):
        X, _, _, MU, SIG = tensors[mi]
        pieces = []
        for start in range(0, market.n, batch_size):
            sl = slice(start, min(start + batch_size, market.n))
            g = model(X[sl].to(device), mi)
            diagonal = torch.diagonal(SIG[sl], dim1=-2, dim2=-1).to(device)
            base = None if anchors is None else anchors[mi][sl].to(device)
            pieces.append(
                mean_signal(
                    model, g, MU[sl].to(device), diagonal, dataset.lookback, base
                ).cpu()
            )
        result.append(torch.cat(pieces))
    return result


def train_member(
    dataset: Dataset,
    config: TrainConfig,
    seed: int,
    output: str | Path,
    device,
    *,
    anchor_root: str | Path | None = None,
    tensors=None,
    resident_inputs=False,
):
    """Train and export a member. A complete hash-checked cache can be reused."""
    config.validate()
    directory = Path(output) / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "fit.json"
    implementation = {
        name: file_hash(Path(__file__).with_name(name))
        for name in [
            "training.py",
            "models.py",
            "layers.py",
            "numerics.py",
            "selection.py",
            "faces.py",
            "nnqp_pruned.py",
        ]
    }
    fingerprint = identity(dataset, config, seed)
    if marker.exists():
        record = json.loads(marker.read_text())
        if record.get("implementation_sha256") != implementation:
            raise ValueError(f"{directory}: completed fit uses a different implementation")
        if record["identity"] != fingerprint:
            raise ValueError(f"{directory}: incompatible completed fit")
        for filename, digest in record["files_sha256"].items():
            if file_hash(directory / filename) != digest:
                raise ValueError(f"{directory/filename}: cache hash mismatch")
        return record
    pending = directory / "fit.pending.json"
    contract = dict(identity=fingerprint, implementation_sha256=implementation)
    if any(directory.iterdir()):
        if not pending.exists() or json.loads(pending.read_text()) != contract:
            raise ValueError(f"{directory}: incomplete output with different or unknown settings")
    save_json(pending, contract)
    if tensors is None:
        tensors = [market.tensors() for market in dataset.markets]
    anchor_record, anchors = None, None
    if config.objective == "hard":
        anchor_root = Path(anchor_root) if anchor_root else Path(output) / "anchors"
        anchor_record = train_member(
            dataset, config.anchor(), seed, anchor_root, device, tensors=tensors,
            resident_inputs=resident_inputs
        )
        anchors = [
            torch.from_numpy(np.load(anchor_root / f"seed_{seed}" / f"{m.name}.npy"))
            for m in dataset.markets
        ]
    with measured(device) as fit_resources:
        model, diagnostics = fit(
            dataset,
            tensors,
            config,
            seed,
            device,
            anchors,
            resident_inputs=resident_inputs,
        )
    with measured(device) as prediction_resources:
        predictions = predict(dataset, tensors, model, device, anchors)
    paths = []
    for market, prediction in zip(dataset.markets, predictions):
        path = directory / f"{market.name}.npy"
        np.save(path, prediction.numpy(), allow_pickle=False)
        paths.append(path)
    torch.save(
        {name: t.detach().cpu() for name, t in model.state_dict().items()},
        directory / "model.pt",
    )
    paths.append(directory / "model.pt")
    record = dict(
        identity=fingerprint,
        seed=seed,
        dataset_identity=dataset.hash,
        configuration=asdict(config),
        execution=dict(resident_inputs=bool(resident_inputs)),
        implementation_sha256=implementation,
        environment=environment(device),
        fit=diagnostics,
        training_resources=fit_resources,
        prediction_resources=prediction_resources,
        anchor_identity=None if anchor_record is None else anchor_record["identity"],
        anchor_directory=(
            None
            if anchor_record is None
            else os.path.relpath(
                Path(anchor_root).resolve() / f"seed_{seed}", directory.resolve()
            )
        ),
        files_sha256={path.name: file_hash(path) for path in paths},
    )
    record["configuration"]["hidden"] = list(config.hidden)
    save_json(marker, record)
    pending.unlink(missing_ok=True)
    return record
