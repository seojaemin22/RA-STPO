"""Fresh anchor-based faces and KKT-checked local allocation gradients."""
from __future__ import annotations

import time
import torch

from .numerics import nnqp
from .selection import greedy_face_safe_draws
from .runtime import synchronize


@torch.no_grad()
def local_inverse(covariance, active):
    keep = active.to(covariance.dtype)
    system = (covariance * keep[..., :, None] * keep[..., None, :]
              + torch.diag_embed(1 - keep))
    inverse = torch.linalg.inv(system)
    return inverse * keep[..., :, None] * keep[..., None, :]


@torch.no_grad()
def kkt_certificate(covariance, rhs, allowed, active, inverse):
    v = torch.matmul(inverse, rhs[..., None]).squeeze(-1)
    fitted = torch.matmul(covariance, v[..., None]).squeeze(-1)
    dual = fitted - rhs
    eps = torch.finfo(v.dtype).eps
    scale = torch.maximum(rhs.abs().amax(-1), fitted.abs().amax(-1)).clamp_min(1e-30)
    tol = 128 * eps * scale
    vscale = v.abs().amax(-1).clamp_min(1e-30)
    primal = torch.where(active, v, torch.full_like(v, float("inf"))).amin(-1)
    stationarity = torch.where(active, dual.abs(), torch.zeros_like(dual)).amax(-1)
    inactive = allowed & ~active
    slack = torch.where(inactive, dual, torch.full_like(dual, float("inf"))).amin(-1)
    # Strict cell interiors only. Boundary rows use the reference active-set solve.
    primal_margin = torch.maximum(128 * eps * vscale, torch.full_like(vscale, 1e-12))
    accepted = ((primal > primal_margin) & (stationarity <= tol)
                & (slack > tol) & torch.isfinite(v).all(-1))
    return accepted, v


def exact_local_response(signal, covariance, faces, masks, inverse, active,
                         *, solver="certified"):
    """Return v on the current active face, keeping its exact local derivative."""
    B, J, k = faces.shape
    rhs = signal[:, None, :].expand(-1, J, -1).gather(-1, faces).double()
    C = covariance.double()
    with torch.no_grad():
        if solver == "certified":
            accepted, _ = kkt_certificate(C, rhs, masks, active, inverse)
        elif solver == "reference":
            accepted = torch.zeros((B, J), dtype=torch.bool, device=signal.device)
        else:
            raise ValueError(solver)
        bad = ~accepted
        new_inverse, new_active = inverse.clone(), active.clone()
        if bool(bad.any()):
            allowed = masks[bad].to(C.dtype)
            system = C[bad] * allowed[..., :, None] * allowed[..., None, :]
            system = system + torch.diag_embed(1 - allowed)
            solution = nnqp(system, rhs[bad] * allowed) * allowed
            keep = (solution > 1e-12) & masks[bad]
            new_active[bad] = keep
            new_inverse[bad] = local_inverse(C[bad], keep)
    # No differentiation through discrete active-set discovery is required on
    # a locally constant active cell. All signal derivatives pass through M a.
    v = torch.matmul(new_inverse, rhs[..., None]).squeeze(-1)
    return v, rhs, new_inverse, new_active, accepted


def portfolio_returns(v, realized, faces, masks):
    """Return one sparse portfolio return per date and sampled path (B x J)."""
    _, draws, cardinality = faces.shape
    y = realized[:, None, :].expand(-1, draws, -1).gather(-1, faces).double()
    count = masks.sum(-1, keepdim=True)
    fallback = torch.where(count > 0, masks.double() / count.clamp_min(1),
                           torch.full_like(v, 1.0 / cardinality))
    budget = v.sum(-1, keepdim=True)
    weights = torch.where(budget > 1e-12, v / budget.clamp_min(1e-12), fallback)
    return (weights * y).sum(-1)


class FacePool:
    """Redraw every candidate face at each epoch, including when J=1."""

    def __init__(self, dataset, tensors, anchors, config, seed, device):
        self.dataset, self.tensors, self.anchors = dataset, tensors, anchors
        self.config, self.device = config, device
        self.generator = torch.Generator(device=device).manual_seed(170141 + 1000003 * seed)
        self.diagonals = [row[4].diagonal(dim1=-2, dim2=-1).clone().to(device)
                          for row in tensors]
        self.banks = []
        for market in dataset.markets:
            n, draws, k = market.train_end, config.faces, config.cardinality
            self.banks.append({
                "faces": torch.empty(n, draws, k, dtype=torch.long, device=device),
                "masks": torch.empty(n, draws, k, dtype=torch.bool, device=device),
                "covariance": torch.empty(n, draws, k, k, dtype=torch.float64, device=device),
                "inverse": torch.empty(n, draws, k, k, dtype=torch.float64, device=device),
                "active": torch.empty(n, draws, k, dtype=torch.bool, device=device),
            })

    @torch.no_grad()
    def refresh(self, epoch):
        synchronize(self.device)
        started = time.perf_counter()
        cfg = self.config
        proposals = 0
        for mi, market in enumerate(self.dataset.markets):
            bank = self.banks[mi]
            for start in range(0, market.train_end, cfg.proposal_batch_size):
                end = min(start + cfg.proposal_batch_size, market.train_end)
                anchor = self.anchors[mi][start:end].to(self.device)
                radius = (self.diagonals[mi][start:end] / self.dataset.lookback).sqrt()
                xi = torch.randn(len(anchor), cfg.faces, market.N, device=self.device,
                                 dtype=anchor.dtype, generator=self.generator)
                signal = (anchor[:, None, :] + radius[:, None, :] * xi).relu()
                covariance = self.tensors[mi][4][start:end].to(self.device)
                face, mask = greedy_face_safe_draws(covariance, signal, cfg.cardinality)
                rows = torch.arange(len(anchor), device=self.device)[:, None, None, None]
                blocks = covariance[rows, face[..., :, None], face[..., None, :]].double()
                bank["faces"][start:end] = face
                bank["masks"][start:end] = mask
                bank["active"][start:end] = mask
                bank["covariance"][start:end] = blocks
                bank["inverse"][start:end] = local_inverse(blocks, mask)
                proposals += len(anchor) * cfg.faces
        synchronize(self.device)
        return dict(epoch=epoch + 1, proposals=proposals, seconds=time.perf_counter()-started)

    def loss(self, market_index, indices, signal, realized):
        bank = self.banks[market_index]
        C, faces, masks, inverse, active = [bank[key][indices] for key in
                                         ["covariance", "faces", "masks", "inverse", "active"]]
        v, _, inverse, active, accepted = exact_local_response(
            signal.relu(), C, faces, masks, inverse, active, solver=self.config.face_solver)
        with torch.no_grad():
            bank["inverse"][indices] = inverse
            bank["active"][indices] = active
        paths = portfolio_returns(v, realized, faces, masks)
        # Each term evaluates an actual k-sparse return path. Average losses last.
        loss = (-paths.mean(dim=0) / paths.std(dim=0).clamp_min(self.config.loss_floor)).mean()
        return loss, accepted.double().mean()
