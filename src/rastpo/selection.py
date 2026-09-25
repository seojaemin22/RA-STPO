"""Residual-alpha selection, exact-prefix reference, and residual upper bounds."""

from __future__ import annotations
import torch
from . import numerics as rules


def _correlation_problem(S: torch.Tensor, z: torch.Tensor):
    """Return unit-diagonal covariance, normalized signal, and marginal scales."""
    scale = torch.sqrt(
        torch.diagonal(S, dim1=-2, dim2=-1).clamp_min(torch.finfo(S.dtype).tiny)
    )
    C = S / scale.unsqueeze(-1) / scale.unsqueeze(-2)
    zeta = z / scale
    return (C, zeta, scale)


def _next_index(score: torch.Tensor, selected: torch.Tensor, floor=None):
    """Choose a positive maximizer with deterministic unused padding after early stopping."""
    dtype = score.dtype
    tiny = torch.finfo(dtype).tiny
    available_score = torch.where(~selected, score, torch.zeros_like(score))
    scale = available_score.abs().amax(-1).clamp_min(tiny)
    tol = 64.0 * torch.finfo(dtype).eps * scale
    admissible = ~selected & (score > tol.unsqueeze(-1))
    if floor is not None:
        admissible = admissible & (score > floor)
    ranked = torch.where(admissible, score, torch.full_like(score, -float("inf")))
    proposed = ranked.argmax(-1)
    has_positive = admissible.any(-1)
    first_unused = (~selected).to(torch.int64).argmax(-1)
    return (torch.where(has_positive, proposed, first_unused), has_positive)


def _gather_principal(S: torch.Tensor, idx: torch.Tensor):
    B, width = idx.shape
    rows = torch.arange(B, device=S.device).view(B, 1, 1)
    return S[rows, idx.unsqueeze(-1), idx.unsqueeze(1)]


def _solve_masked_nnqp(Cff: torch.Tensor, zf: torch.Tensor, allowed: torch.Tensor):
    keep = allowed.to(Cff.dtype)
    outer = keep.unsqueeze(-1) * keep.unsqueeze(-2)
    system = Cff * outer + torch.diag_embed(1.0 - keep)
    rhs = zf * keep
    return rules.nnqp(system, rhs) * keep


@torch.no_grad()
def greedy_face_long_only(
    S: torch.Tensor,
    z: torch.Tensor,
    k: int,
    *,
    appraisal: bool = False,
    return_stats: bool = False,
    measured_stop_T: float | None = None,
):
    """The theorem-faithful maximum-positive-alpha forward selection map.

    Inputs have shapes ``S=(B,N,N)`` and ``z=(B,N)``.  The returned face always has a fixed
    ``(B,k)`` tensor shape.  ``active`` distinguishes genuine greedy augmentations from
    deterministic padding when the unrestricted nonnegative optimum uses fewer than ``k``
    assets.
    """
    B, N = z.shape
    k = min(int(k), N)
    C, zeta, _ = _correlation_problem(S, z)
    face = torch.empty(B, k, dtype=torch.long, device=z.device)
    active = torch.zeros(B, k, dtype=torch.bool, device=z.device)
    selected = torch.zeros(B, N, dtype=torch.bool, device=z.device)
    alpha = zeta.clone()
    cvar = torch.ones_like(zeta)
    v_face = z.new_zeros(B, 0)
    for j in range(k):
        positive = alpha.clamp_min(0)
        score = (
            positive.square() / cvar.clamp_min(torch.finfo(z.dtype).tiny)
            if appraisal
            else positive
        )
        floor = (
            None
            if measured_stop_T is None or appraisal
            else (cvar.clamp_min(0) / measured_stop_T).sqrt()
        )
        idx, accepted = _next_index(score, selected, floor)
        face[:, j] = idx
        active[:, j] = accepted
        selected.scatter_(1, idx.unsqueeze(-1), True)
        current = face[:, : j + 1]
        allowed = active[:, : j + 1]
        Cff = _gather_principal(C, current)
        zf = torch.gather(zeta, 1, current)
        v_face = _solve_masked_nnqp(Cff, zf, allowed)
        rows = torch.arange(B, device=z.device).view(B, 1, 1)
        cols = C[
            rows, torch.arange(N, device=z.device).view(1, N, 1), current.view(B, 1, -1)
        ]
        alpha = zeta - torch.bmm(cols, v_face.unsqueeze(-1)).squeeze(-1)
        rhs = cols.transpose(1, 2) * allowed.to(z.dtype).unsqueeze(-1)
        keep = allowed.to(z.dtype)
        system = Cff * (keep.unsqueeze(-1) * keep.unsqueeze(-2)) + torch.diag_embed(
            1 - keep
        )
        coefficients = torch.linalg.solve(system, rhs) * keep.unsqueeze(-1)
        cvar = (1.0 - (cols * coefficients.transpose(1, 2)).sum(-1)).clamp_min(0)
    if not return_stats:
        return (face, active)
    value = (torch.gather(zeta, 1, face) * v_face).sum(-1)
    return (
        face,
        active,
        {
            "alpha": alpha,
            "conditional_variance": cvar,
            "value": value,
            "size": active.sum(-1),
        },
    )


@torch.no_grad()
def _greedy_face_indexed(C, zeta, batch_rows, k):
    """Reference alpha greedy with one shared correlation matrix per date.

    This is the same sequence of masked prefix NNQPs as greedy_face_long_only.
    It avoids materializing an N x N covariance per draw and does not compute
    unused conditional variances when neither appraisal nor stopping uses them.
    """
    M,N=zeta.shape
    face=torch.empty(M,k,dtype=torch.long,device=zeta.device)
    active=torch.zeros(M,k,dtype=torch.bool,device=zeta.device)
    selected=torch.zeros(M,N,dtype=torch.bool,device=zeta.device)
    alpha=zeta.clone()
    rows=batch_rows[:,None,None]
    assets=torch.arange(N,device=zeta.device)[None,:,None]
    for j in range(k):
        idx,accepted=_next_index(alpha.clamp_min(0),selected)
        face[:,j]=idx;active[:,j]=accepted
        selected.scatter_(1,idx[:,None],True)
        current=face[:,:j+1];allowed=active[:,:j+1]
        Cff=C[rows,current[:,:,None],current[:,None,:]]
        zf=torch.gather(zeta,1,current)
        v=_solve_masked_nnqp(Cff,zf,allowed)
        columns=C[rows,assets,current[:,None,:]]
        alpha=zeta-torch.bmm(columns,v[:,:,None]).squeeze(-1)
    return face,active


@torch.no_grad()
def greedy_face_safe_draws(
    S: torch.Tensor,
    z: torch.Tensor,
    k: int,
    *,
    return_stats: bool = False,
    measured_stop_T: float | None = None,
):
    """Fail-closed fast implementation of theorem-faithful alpha greedy.

    A block-inverse recurrence checks the unrestricted coefficient vector on every selected
    prefix. If all coefficients are strictly positive, the Gram--Schmidt residual is exactly
    the long-only KKT residual. Any row that fails this sufficient condition is recomputed by
    :func:`greedy_face_long_only`; hence the returned map never relies on the condition.
    """
    B, draws, N = z.shape
    k = min(int(k), N)
    tiny = torch.finfo(z.dtype).tiny
    eps = torch.finfo(z.dtype).eps
    scale = torch.sqrt(torch.diagonal(S, dim1=-2, dim2=-1).clamp_min(tiny))
    zeta = z / scale.unsqueeze(1)
    face = torch.empty(B, draws, k, dtype=torch.long, device=z.device)
    active = torch.zeros(B, draws, k, dtype=torch.bool, device=z.device)
    selected = torch.zeros(B, draws, N, dtype=torch.bool, device=z.device)
    basis = torch.zeros(B, draws, N, k, dtype=z.dtype, device=z.device)
    inverse = torch.zeros(B, draws, k, k, dtype=z.dtype, device=z.device)
    coefficients = torch.zeros(B, draws, k, dtype=z.dtype, device=z.device)
    alpha = zeta.clone()
    cvar = torch.ones_like(zeta)
    all_interior = torch.ones(B, draws, dtype=torch.bool, device=z.device)
    expanded = S.unsqueeze(1).expand(B, draws, N, N)
    for j in range(k):
        floor = (
            None
            if measured_stop_T is None
            else (cvar.clamp_min(0) / measured_stop_T).sqrt()
        )
        idx, accepted = _next_index(alpha.clamp_min(0), selected, floor)
        face[:, :, j] = idx
        active[:, :, j] = accepted
        selected.scatter_(2, idx.unsqueeze(-1), True)
        column_index = idx.unsqueeze(-1).unsqueeze(-1).expand(B, draws, N, 1)
        col = torch.gather(expanded, 3, column_index).squeeze(-1)
        selected_scale = torch.gather(
            scale.unsqueeze(1).expand(B, draws, N), 2, idx.unsqueeze(-1)
        )
        col = col / scale.unsqueeze(1) / selected_scale
        alpha_i = torch.gather(alpha, 2, idx.unsqueeze(-1)).squeeze(-1)
        if j:
            previous = face[:, :, :j]
            covariance_to_face = torch.gather(col, 2, previous)
            h = torch.matmul(
                inverse[:, :, :j, :j], covariance_to_face.unsqueeze(-1)
            ).squeeze(-1)
            schur = (1.0 - (covariance_to_face * h).sum(-1)).clamp_min(tiny)
            new_coefficient = alpha_i / schur
            proposed_previous = coefficients[:, :, :j] - h * new_coefficient.unsqueeze(
                -1
            )
            proposed = torch.cat([proposed_previous, new_coefficient.unsqueeze(-1)], -1)
        else:
            h = None
            schur = torch.ones_like(alpha_i)
            proposed = alpha_i.unsqueeze(-1)
        proposed_scale = proposed.abs().amax(-1).clamp_min(tiny)
        prefix_ok = (proposed > (128.0 * eps * proposed_scale).unsqueeze(-1)).all(-1)
        all_interior &= ~accepted | prefix_ok
        use = accepted.to(z.dtype)
        if j:
            coefficients[:, :, :j] = torch.where(
                accepted.unsqueeze(-1), proposed[:, :, :j], coefficients[:, :, :j]
            )
        coefficients[:, :, j] = use * proposed[:, :, j]
        if j:
            inv_old = inverse[:, :, :j, :j].clone()
            inv_new = (
                inv_old + h.unsqueeze(-1) * h.unsqueeze(-2) / schur[..., None, None]
            )
            inverse[:, :, :j, :j] = torch.where(
                accepted[..., None, None], inv_new, inv_old
            )
            cross = -h / schur.unsqueeze(-1)
            inverse[:, :, :j, j] = use.unsqueeze(-1) * cross
            inverse[:, :, j, :j] = use.unsqueeze(-1) * cross
        inverse[:, :, j, j] = torch.where(
            accepted, schur.reciprocal(), torch.ones_like(schur)
        )
        if j:
            projection = torch.gather(
                basis[:, :, :, :j],
                2,
                idx.unsqueeze(-1).unsqueeze(-1).expand(B, draws, 1, j),
            ).squeeze(2)
            col = col - (basis[:, :, :, :j] * projection.unsqueeze(2)).sum(-1)
        denom = torch.sqrt(torch.gather(cvar, 2, idx.unsqueeze(-1)).clamp_min(tiny))
        direction = col / denom
        orthogonal_coefficient = torch.gather(alpha, 2, idx.unsqueeze(-1)) / denom
        direction = direction * use.unsqueeze(-1)
        orthogonal_coefficient = orthogonal_coefficient * use.unsqueeze(-1)
        basis[:, :, :, j] = direction
        alpha = alpha - direction * orthogonal_coefficient
        cvar = (cvar - direction.square()).clamp_min(0)
    fallback = ~all_interior
    fallback_count = int(fallback.sum().item())
    if fallback_count:
        flat_z = z.reshape(B * draws, N)
        flat_fallback = fallback.reshape(-1)
        batch_rows = torch.arange(B, device=z.device).repeat_interleave(draws)
        if measured_stop_T is None:
            C=S/scale.unsqueeze(-1)/scale.unsqueeze(-2)
            exact_face,exact_active=_greedy_face_indexed(
                C,zeta.reshape(B*draws,N)[flat_fallback],batch_rows[flat_fallback],k)
        else:
            exact_face, exact_active = greedy_face_long_only(
                S[batch_rows[flat_fallback]],
                flat_z[flat_fallback],
                k,
                measured_stop_T=measured_stop_T,
            )
        flat_face = face.reshape(B * draws, k)
        flat_active = active.reshape(B * draws, k)
        flat_face[flat_fallback] = exact_face
        flat_active[flat_fallback] = exact_active
    if return_stats:
        return (
            face,
            active,
            {"interior_fast_path": all_interior, "fallback_count": fallback_count},
        )
    return (face, active)


@torch.no_grad()
def face_value(
    S: torch.Tensor,
    z: torch.Tensor,
    face: torch.Tensor,
    active: torch.Tensor | None = None,
):
    """Exact long-only squared-Sharpe value on a possibly padded face."""
    if active is None:
        active = torch.ones_like(face, dtype=torch.bool)
    Sff, zf = rules.gather_face(S, z, face)
    v = _solve_masked_nnqp(Sff, zf, active)
    return ((zf * v).sum(-1), v)


@torch.no_grad()
def protocol_gamma_lower_bound(S: torch.Tensor, delta: float = 0.1):
    """Restricted-eigenvalue lower bound for isotropic covariance shrinkage."""
    diag = torch.diagonal(S, dim1=-2, dim2=-1)
    tau = diag.mean(-1)
    return delta * tau / diag.amax(-1).clamp_min(torch.finfo(S.dtype).tiny)


@torch.no_grad()
def certificate_upper(
    S: torch.Tensor,
    z: torch.Tensor,
    face: torch.Tensor,
    active: torch.Tensor | None = None,
    delta: float = 0.1,
):
    """Residual upper bound for a covariance with the stated isotropic shrinkage."""
    if active is None:
        active = torch.ones_like(face, dtype=torch.bool)
    C, zeta, _ = _correlation_problem(S, z)
    Cff = _gather_principal(C, face)
    zf = torch.gather(zeta, 1, face)
    v = _solve_masked_nnqp(Cff, zf, active)
    rows = torch.arange(len(z), device=z.device).view(-1, 1, 1)
    assets = torch.arange(z.shape[-1], device=z.device).view(1, -1, 1)
    cols = C[rows, assets, face.view(len(z), 1, -1)]
    alpha = zeta - torch.bmm(cols, v.unsqueeze(-1)).squeeze(-1)
    inside = torch.zeros_like(alpha, dtype=torch.bool)
    inside.scatter_(1, face, active)
    outside = ~inside
    residual = torch.where(
        outside, alpha.clamp_min(0).square(), torch.zeros_like(alpha)
    )
    top = residual.topk(min(face.shape[-1], residual.shape[-1]), dim=-1).values.sum(-1)
    value = (zf * v).sum(-1)
    gamma0 = protocol_gamma_lower_bound(S, delta=delta)
    return (value + top / gamma0, value, alpha, gamma0)
