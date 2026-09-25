"""Batched nonnegative quadratic solves and sparse allocation baselines.

OSCAR selects transformed coordinates and then applies a long-only face solve.
mSSRM-PGA and ASMP-AFBA retain their nonnegative allocation algorithms.
See Appendix A of the paper for the baseline formulations and adaptations.
"""

from __future__ import annotations
import torch


def _chol(S, jitter=1e-12):
    """Cholesky factorization with the numerical safeguards used by the baseline."""
    N = S.shape[-1]
    eye = torch.eye(N, dtype=S.dtype, device=S.device)
    for j in (0.0, jitter, 1e-10, 1e-08, 1e-06):
        try:
            return torch.linalg.cholesky(S + j * eye)
        except Exception:
            continue
    try:
        return torch.linalg.cholesky(
            S + 0.0001 * torch.diagonal(S, dim1=-2, dim2=-1).mean() * eye
        )
    except Exception:
        return torch.linalg.cholesky(S.cpu() + jitter * eye.cpu()).to(S.device)


def gather_face(S, mu, idx):
    """Sub-matrix and sub-vector on the k indices in `idx` (B, k)."""
    B, k = idx.shape
    b = torch.arange(B, device=mu.device).unsqueeze(-1)
    Sff = S[b.unsqueeze(-1), idx.unsqueeze(-1), idx.unsqueeze(1)]
    muf = torch.gather(mu, 1, idx)
    return (Sff, muf)


def _lipschitz(Sff):
    """Gershgorin upper bound on the largest eigenvalue of a symmetric matrix."""
    return Sff.abs().sum(-1).amax(-1).clamp_min(1e-14).unsqueeze(-1)


def _nnqp_active_set(Sff, muf, max_outer=None, max_inner=None):
    """Lawson--Hanson active-set solve for a batch of small strictly convex NNQPs.

    The passive coordinates solve the equality-constrained system exactly.  A variable with
    negative reduced cost enters, and the ratio test removes a passive variable before it can
    become negative.  The returned flag is a KKT audit; callers retain a first-order fallback
    for any row that exhausts the finite numerical safeguard.
    """
    B, k, _ = Sff.shape
    max_outer = max_outer or 20 * k + 20
    max_inner = max_inner or 10 * k + 10
    eps = torch.finfo(Sff.dtype).eps
    tiny = torch.finfo(Sff.dtype).tiny
    rows_all = torch.arange(B, device=Sff.device)
    x = torch.zeros_like(muf)
    passive = torch.zeros(B, k, dtype=torch.bool, device=Sff.device)
    unfinished = torch.ones(B, dtype=torch.bool, device=Sff.device)
    for _ in range(max_outer):
        if not bool(unfinished.any()):
            break
        ur = rows_all[unfinished]
        fitted = torch.bmm(Sff[ur], x[ur].unsqueeze(-1)).squeeze(-1)
        residual = muf[ur] - fitted
        scale = torch.maximum(muf[ur].abs().amax(-1), fitted.abs().amax(-1)).clamp_min(
            tiny
        )
        tol = 64.0 * eps * scale
        score = torch.where(
            ~passive[ur], residual, torch.full_like(residual, -float("inf"))
        )
        enter_value, enter_index = score.max(-1)
        entering_local = enter_value > tol
        unfinished[ur[~entering_local]] = False
        if not bool(entering_local.any()):
            continue
        enter_rows = ur[entering_local]
        passive[enter_rows, enter_index[entering_local]] = True
        working = torch.zeros(B, dtype=torch.bool, device=Sff.device)
        working[enter_rows] = True
        for _ in range(max_inner):
            if not bool(working.any()):
                break
            wr = rows_all[working]
            keep_bool = passive[working]
            keep = keep_bool.to(Sff.dtype)
            outer = keep.unsqueeze(-1) * keep.unsqueeze(-2)
            system = Sff[working] * outer + torch.diag_embed(1.0 - keep)
            z = (
                torch.linalg.solve(system, (muf[working] * keep).unsqueeze(-1)).squeeze(
                    -1
                )
                * keep
            )
            nonpositive = keep_bool & (z <= 0)
            good = ~nonpositive.any(-1)
            if bool(good.any()):
                x[wr[good]] = z[good]
                working[wr[good]] = False
            bad = ~good
            if not bool(bad.any()):
                continue
            br = wr[bad]
            xb = x[br]
            zb = z[bad]
            pb = passive[br]
            crossing = pb & (zb <= 0)
            ratios = torch.where(
                crossing,
                xb / (xb - zb).clamp_min(tiny),
                torch.full_like(xb, float("inf")),
            )
            alpha = ratios.amin(-1, keepdim=True).clamp(0.0, 1.0)
            moved = xb + alpha * (zb - xb)
            moved_scale = torch.maximum(xb.abs().amax(-1), zb.abs().amax(-1)).clamp_min(
                tiny
            )
            zero_tol = (64.0 * eps * moved_scale).unsqueeze(-1)
            leaving = pb & (moved <= zero_tol)
            x[br] = torch.where(leaving, torch.zeros_like(moved), moved.clamp_min(0))
            passive[br] &= ~leaving
    fitted = torch.bmm(Sff, x.unsqueeze(-1)).squeeze(-1)
    gradient = fitted - muf
    scale = torch.maximum(muf.abs().amax(-1), fitted.abs().amax(-1)).clamp_min(tiny)
    tol = 128.0 * eps * scale
    active_residual = torch.where(
        passive, gradient.abs(), torch.zeros_like(gradient)
    ).amax(-1)
    inactive_min = torch.where(
        ~passive, gradient, torch.full_like(gradient, float("inf"))
    ).amin(-1)
    converged = (active_residual <= tol) & (inactive_min >= -tol)
    return (x, converged)


def nnqp(Sff, muf, iters=300, polish=8, cap=8192, interior_fast=True, active_fast=True):
    if interior_fast and active_fast:
        from .nnqp_pruned import solve
        return solve(Sff,muf,_nnqp_reference,iters=iters,polish=polish,cap=cap,
                     interior_fast=interior_fast,active_fast=active_fast)
    return _nnqp_reference(Sff,muf,iters,polish,cap,interior_fast,active_fast)


def _nnqp_reference(Sff, muf, iters=300, polish=8, cap=8192, interior_fast=True, active_fast=True):
    """min 1/2 v' Sff v - muf' v  s.t.  v >= 0, batched, k x k.

    Try an interior solve and an active-set solve first. The first-order fallback is
    accepted only after checking all KKT conditions, including excluded coordinates.
    Removing negative coordinates alone is not a certificate of the NNQP optimum."""
    if Sff.shape[0] > cap:
        return torch.cat(
            [
                nnqp(
                    Sff[i : i + cap],
                    muf[i : i + cap],
                    iters,
                    polish,
                    cap,
                    interior_fast,
                    active_fast,
                )
                for i in range(0, Sff.shape[0], cap)
            ],
            0,
        )
    B, k, _ = Sff.shape
    if interior_fast:
        full = torch.linalg.solve(Sff, muf.unsqueeze(-1)).squeeze(-1)
        interior = (full >= 0).all(-1)
        if bool(interior.all()):
            return full
        if bool(interior.any()):
            result = torch.empty_like(muf)
            result[interior] = full[interior]
            result[~interior] = nnqp(
                Sff[~interior],
                muf[~interior],
                iters,
                polish,
                cap,
                interior_fast=False,
                active_fast=active_fast,
            )
            return result
    if active_fast:
        active_solution, active_converged = _nnqp_active_set(Sff, muf)
        if bool(active_converged.all()):
            return active_solution
        if bool(active_converged.any()):
            result = torch.empty_like(muf)
            result[active_converged] = active_solution[active_converged]
            result[~active_converged] = nnqp(
                Sff[~active_converged],
                muf[~active_converged],
                iters,
                polish,
                cap,
                interior_fast=False,
                active_fast=False,
            )
            return result
    L = _lipschitz(Sff)
    v = torch.relu(muf) / L
    y, t = (v.clone(), torch.ones(B, 1, dtype=Sff.dtype, device=Sff.device))
    for _ in range(iters):
        g = torch.bmm(Sff, y.unsqueeze(-1)).squeeze(-1) - muf
        vn = torch.relu(y - g / L)
        tn = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
        y = vn + (t - 1.0) / tn * (vn - v)
        v, t = (vn, tn)
    act = v > 1e-14
    act = torch.where(
        act.any(-1, keepdim=True), act, muf >= muf.max(-1, keepdim=True).values
    )
    sol = v
    for _ in range(polish):
        keep = act.to(Sff.dtype)
        outer = keep.unsqueeze(-1) * keep.unsqueeze(-2)
        A = Sff * outer + torch.diag_embed(1.0 - keep)
        sol = torch.linalg.solve(A, (muf * keep).unsqueeze(-1)).squeeze(-1) * keep
        bad = act & (sol < -1e-16)
        if not bool(bad.any()):
            break
        act = act & ~bad
        act = torch.where(
            act.any(-1, keepdim=True), act, muf >= muf.max(-1, keepdim=True).values
        )
    answer = torch.relu(sol)
    fitted = torch.bmm(Sff, answer.unsqueeze(-1)).squeeze(-1)
    gradient = fitted - muf
    scale = torch.maximum(muf.abs().amax(-1), fitted.abs().amax(-1)).clamp_min(
        torch.finfo(Sff.dtype).tiny
    )
    tolerance = 256 * torch.finfo(Sff.dtype).eps * scale
    residual = torch.where(answer > 0, gradient.abs(), torch.relu(-gradient)).amax(-1)
    failed = ~torch.isfinite(residual) | (residual > tolerance)
    if bool(failed.any()):
        refined, passed = _nnqp_active_set(
            Sff[failed].double(),
            muf[failed].double(),
            max_outer=100 * k + 100,
            max_inner=50 * k + 50,
        )
        if not bool(passed.all()):
            raise RuntimeError(
                "NNQP numerical safeguard exhausted without a KKT-valid solution"
            )
        answer[failed] = refined.to(answer.dtype)
    return answer


def tangency_on_support(S, mu, idx):
    """Exact long-only tangency restricted to `idx`, returned on the simplex."""
    B, k = idx.shape
    Sff, muf = gather_face(S, mu, idx)
    v = nnqp(Sff, muf)
    s = v.sum(-1, keepdim=True)
    fallback = torch.full_like(v, 1.0 / k)
    v = torch.where(s > 1e-14, v / s.clamp_min(1e-14), fallback)
    return torch.zeros_like(mu).scatter_(1, idx, v)


def sharpe_of(w, mu, S):
    num = (mu * w).sum(-1)
    den = torch.sqrt(
        torch.relu((w.unsqueeze(1) @ S @ w.unsqueeze(-1)).squeeze(-1).squeeze(-1))
    )
    return num / den.clamp_min(1e-14)


def topk_idx(score, k):
    """Top-k indices after replacing NaN and infinities with finite sentinels."""
    if not torch.isfinite(score).all():
        score = torch.nan_to_num(score, nan=-1e30, posinf=1e30, neginf=-1e30)
    return score.topk(min(k, score.shape[-1]), dim=-1).indices


def asset_order(N, seed, device=None):
    """A reproducible asset ordering, one per seed."""
    g = torch.Generator().manual_seed(1000 + int(seed))
    q = torch.randperm(N, generator=g)
    return q.to(device) if device is not None else q


def oscar(mu, S, k, perm=None, **_):
    """Select by the absolute Cholesky-transformed tangency score, then allocate long-only."""
    if perm is not None:
        q = perm.to(mu.device)
        inv = torch.empty_like(q).scatter_(
            0, q, torch.arange(q.numel(), device=q.device)
        )
        return oscar(mu[:, q], S[:, q][:, :, q], k)[:, inv]
    Lc = _chol(S)
    y = torch.cholesky_solve(mu.unsqueeze(-1), Lc).squeeze(-1)
    sc = torch.abs(torch.bmm(Lc.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1))
    return tangency_on_support(S, mu, topk_idx(sc, k))


def mssrm_pga(mu, S, k, eps=0.001, iters=3000, tol=1e-06, **_):
    """Nonnegative sparse Sharpe-ratio proximal-gradient allocation."""
    B, N = mu.shape
    Se = S + eps * torch.eye(N, dtype=S.dtype, device=S.device)
    L = torch.linalg.eigvalsh(Se)[:, -1].clamp_min(1e-14).view(B, 1)
    step = 0.999 / L
    v = mu.clone()
    for _ in range(iters):
        prev = v
        c = torch.relu(v - step * (torch.bmm(Se, v.unsqueeze(-1)).squeeze(-1) - mu))
        if k < N:
            values, indices = c.topk(k, dim=-1)
            v = torch.zeros_like(c).scatter_(1, indices, values)
        else:
            v = c
        if torch.all(
            torch.linalg.vector_norm(v - prev, dim=-1)
            <= tol * torch.linalg.vector_norm(prev, dim=-1).clamp_min(1e-30)
        ):
            break
    s = v.sum(-1, keepdim=True)
    best = (mu / torch.diagonal(S, dim1=-2, dim2=-1).sqrt()).argmax(-1, keepdim=True)
    fallback = torch.zeros_like(v).scatter_(1, best, 1.0)
    return torch.where(s > 1e-14, v / s.clamp_min(1e-14), fallback)


def _project_simplex(x, total=1.0):
    u, _ = torch.sort(x, dim=-1, descending=True)
    css = torch.cumsum(u, dim=-1)
    idx = torch.arange(1, x.shape[-1] + 1, dtype=x.dtype, device=x.device)
    valid = u > (css - total) / idx
    rho = valid.to(x.dtype).cumsum(-1).argmax(-1, keepdim=True)
    theta = (torch.gather(css, 1, rho) - total) / (rho.to(x.dtype) + 1)
    return torch.relu(x - theta)


def _asmp_stage(
    M2,
    mu,
    lam=6.0,
    beta_c=0.5,
    tol=0.0001,
    iters=3000,
    rho_lo=0.01,
    rho_hi=0.1,
    a=1.0 / 2.01,
    b=1.0,
):
    """One accelerated forward-backward stage of ASMP-AFBA, as released."""
    B, n = mu.shape
    dt, dv = (mu.dtype, mu.device)
    A = torch.zeros(B, n + 1, n + 1, dtype=dt, device=dv)
    A[:, :n, :n] = M2
    ma = torch.cat([mu, -torch.ones(B, 1, dtype=dt, device=dv)], dim=-1)
    A = A + lam * ma.unsqueeze(-1) * ma.unsqueeze(1)
    nrm = torch.linalg.eigvalsh(A)[:, -1].clamp_min(1e-30).view(B, 1)
    beta = beta_c / nrm
    prev = torch.cat(
        [
            torch.full((B, n), 1.0 / n, dtype=dt, device=dv),
            torch.full((B, 1), 0.5 * (rho_lo + rho_hi), dtype=dt, device=dv),
        ],
        -1,
    )
    cur = prev.clone()
    for it in range(1, iters + 1):
        num = a * (it - 2) + b - 1.0
        den = a * (it - 1) + b
        ex = cur + num / den * (cur - prev)
        prev = cur
        d = ex - 2.0 * beta * torch.bmm(A, ex.unsqueeze(-1)).squeeze(-1)
        cur = torch.cat(
            [_project_simplex(d[:, :n]), d[:, n:].clamp(rho_lo, rho_hi)], dim=-1
        )
        if it % 50 == 0:
            r = torch.linalg.vector_norm(cur - prev, dim=-1) / torch.linalg.vector_norm(
                prev, dim=-1
            ).clamp_min(1e-30)
            if bool((r <= tol).all()):
                break
    return cur[:, :n]


def asmp_afba(mu, S, k, **_):
    """Two-stage accelerated forward-backward allocation on a selected asset face."""
    B, N = mu.shape
    M2 = S + mu.unsqueeze(-1) * mu.unsqueeze(1)
    s1 = _asmp_stage(M2, mu)
    m = min(N, max(4, k))
    idx = topk_idx(s1, m)
    idx, _ = torch.sort(idx, dim=-1)
    Sff, muf = gather_face(S, mu, idx)
    M2f = Sff + muf.unsqueeze(-1) * muf.unsqueeze(1)
    s2 = _asmp_stage(M2f, muf)
    ssum = s2.sum(-1, keepdim=True)
    s2 = torch.where(
        ssum > 1e-14, s2 / ssum.clamp_min(1e-14), torch.full_like(s2, 1.0 / m)
    )
    return torch.zeros_like(mu).scatter_(1, idx, s2)
