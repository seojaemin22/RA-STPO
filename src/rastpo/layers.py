"""Differentiable soft-coordinate and exact nonnegative asset-face layers."""

from __future__ import annotations
import torch
import math
from . import numerics as rules
from .selection import greedy_face_safe_draws


def zt(a, eps=1e-8):
    return (a - a.mean(-1, keepdim=True)) / (a.std(-1, keepdim=True) + eps)


def cholesky(S):
    """Factor a positive-definite covariance; use CPU only if device factorization fails."""
    try:
        return torch.linalg.cholesky(S)
    except Exception:
        return torch.linalg.cholesky(S.cpu()).to(S.device)


def _unit_risk(d, chol):
    r = torch.linalg.vector_norm(
        torch.bmm(chol.transpose(1, 2), d.unsqueeze(-1)).squeeze(-1),
        dim=-1,
        keepdim=True,
    )
    return d / r.clamp_min(1e-12)


class _ImplicitSoftTopK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, k, sharp, iters):
        if not 0 < k < z.shape[-1] or sharp <= 0:
            raise ValueError("soft top-k requires 0 < k < N and positive sharpness")
        lo = -z.amax(-1, keepdim=True) - 20.0 / sharp
        hi = -z.amin(-1, keepdim=True) + 20.0 / sharp
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            mass = torch.sigmoid(sharp * (z + mid)).sum(-1, keepdim=True)
            lo = torch.where(mass < k, mid, lo)
            hi = torch.where(mass >= k, mid, hi)
        p = torch.sigmoid(sharp * (z + 0.5 * (lo + hi)))
        ctx.save_for_backward(p)
        ctx.sharp = sharp
        return p

    @staticmethod
    def backward(ctx, grad_output):
        (p,) = ctx.saved_tensors
        v = p * (1 - p)
        total = v.sum(-1, keepdim=True).clamp_min(torch.finfo(v.dtype).tiny)
        correction = (grad_output * v).sum(-1, keepdim=True) / total
        return (ctx.sharp * v * (grad_output - correction), None, None, None)


def soft_topk(scores, k, sharp=10.0, iters=32):
    """Published shared-shift mask on raw Cholesky scores."""
    return _ImplicitSoftTopK.apply(scores, k, sharp, iters)


def budgeted_tangency(mu, S=None, *, chol=None):
    """Solve max mu'u subject to u'Su <= 1 and 1'u >= 0.

    This is the published auxiliary-budget program after eliminating tau.
    The equality-active branch projects the inverse-covariance direction onto
    the zero-budget subspace before normalizing its risk. No budget constraint
    is dropped. The returned variable is the risk-constrained program variable,
    not u/tau (which is undefined on the binding boundary).

    Arithmetic follows the input dtype. Callers train this layer in float64.
    Zero-value degenerate objectives return zero; 1e-12 is a numerical norm floor.
    """
    if chol is None:
        if S is None:
            raise ValueError("a covariance or its Cholesky factor is required")
        chol = cholesky(S)
    rhs = torch.stack([mu, torch.ones_like(mu)], dim=-1)
    inverse = torch.cholesky_solve(rhs, chol)
    direction, normal = inverse[..., 0], inverse[..., 1]
    budget = direction.sum(-1, keepdim=True)
    normal_budget = normal.sum(-1, keepdim=True)
    direction = direction - budget.clamp_max(0) / normal_budget * normal
    risk = torch.linalg.vector_norm(
        torch.bmm(chol.transpose(1, 2), direction.unsqueeze(-1)).squeeze(-1),
        dim=-1, keepdim=True,
    )
    return torch.where(
        risk > 1e-12, direction / risk.clamp_min(1e-12),
        torch.zeros_like(direction),
    )


def oscar_soft(mu, S, k, perm=None):
    """DF-STPO Algorithm 1: two budget-constrained solves and one mean mask.

    Raw scores enter the implicit mask. The second solve's output is evaluated
    directly, with neither an additional output mask nor risk renormalization.
    """
    if perm is not None:
        q = perm.to(mu.device)
        inv = torch.empty_like(q).scatter_(
            0, q, torch.arange(q.numel(), device=q.device)
        )
        return oscar_soft(mu[:, q], S[:, q][:, :, q], k)[:, inv]
    Lc = cholesky(S)
    w = budgeted_tangency(mu, chol=Lc)
    sc = torch.abs(torch.bmm(Lc.transpose(1, 2), w.unsqueeze(-1)).squeeze(-1))
    m = soft_topk(sc, k)
    return budgeted_tangency(mu * m, chol=Lc)


def partitioned_nnqp(system, signal, batch_size, allowed):
    """Re-solve individual draw batches near primal or dual numerical boundaries.

    The square-root machine-precision screen concerns numerical classification,
    not a statistical regularization parameter or an optimality certificate.
    A flagged draw is recomputed in full to retain its batch context.
    """
    answer = rules.nnqp(system, signal)
    if len(system) <= batch_size:
        return answer
    fitted = system.bmm(answer.unsqueeze(-1)).squeeze(-1)
    dual = fitted - signal
    relative = math.sqrt(torch.finfo(system.dtype).eps)
    tiny = torch.finfo(system.dtype).tiny
    primal_scale = answer.abs().amax(-1, keepdim=True).clamp_min(tiny)
    dual_scale = torch.maximum(
        fitted.abs().amax(-1, keepdim=True), signal.abs().amax(-1, keepdim=True)
    ).clamp_min(tiny)
    active = answer > 1e-12
    boundary = allowed & torch.where(
        active, answer <= relative * primal_scale, dual <= relative * dual_scale
    )
    boundary &= signal.abs().amax(-1, keepdim=True) > 0
    for start in range(0, len(system), batch_size):
        if bool(boundary[start : start + batch_size].any()):
            answer[start : start + batch_size] = rules.nnqp(
                system[start : start + batch_size], signal[start : start + batch_size]
            )
    return answer


def exact_layer(
    S, mu, k, face=None, face_mask=None, face_factor=None, active_batch_size=None
):
    """Locate a hard face and solve the long-only quadratic exactly on that face.

    Both the discrete face and the restricted QP active set are stopped.  On every cell on
    which those two sets are locally fixed, autograd therefore differentiates the exact
    active-subface linear solve.  This is the conditional Jacobian used in the paper.
    """
    if face is None:
        with torch.no_grad():
            located, allowed = greedy_face_safe_draws(
                S, torch.relu(mu.detach())[:, None], k
            )
            face, face_mask = located[:, 0], allowed[:, 0]
    B, k2 = face.shape
    if active_batch_size is not None and active_batch_size < 1:
        raise ValueError("active-set batch size must be positive")

    def active_solve(system, signal):
        if active_batch_size is None:
            return rules.nnqp(system, signal)
        # Preserve each draw's active-set numerical context. Near a zero
        # coefficient, merging these batches can change the discrete derivative cell.
        return partitioned_nnqp(system, signal, active_batch_size, allowed)

    Sff, muf = rules.gather_face(S, mu, face)
    af = torch.relu(muf)
    with torch.no_grad():
        if face_mask is not None:
            allowed = face_mask.to(device=S.device, dtype=torch.bool)
            allowed_float = allowed.to(S.dtype)
            allowed_outer = allowed_float.unsqueeze(-1) * allowed_float.unsqueeze(-2)
            selection_system = Sff.detach() * allowed_outer + torch.diag_embed(
                1.0 - allowed_float
            )
            rhs0 = (af.detach() * allowed_float).unsqueeze(-1)
            if face_factor is not None:
                candidate = (
                    torch.cholesky_solve(rhs0, face_factor.detach()).squeeze(-1)
                    * allowed_float
                )
                candidate_interior = torch.where(
                    allowed, candidate > 0, torch.ones_like(allowed)
                ).all(-1)
                v0 = candidate.clone()
                if bool((~candidate_interior).any()):
                    v0[~candidate_interior] = rules.nnqp(
                        selection_system[~candidate_interior],
                        (af.detach() * allowed_float)[~candidate_interior],
                    )
            else:
                v0 = active_solve(selection_system, af.detach() * allowed_float)
        else:
            allowed = torch.ones_like(af, dtype=torch.bool)
            v0 = active_solve(Sff.detach(), af.detach())
        active = v0 > 1e-12
        active = active & allowed
        active = torch.where(
            active.any(-1, keepdim=True),
            active,
            (af.detach() >= af.detach().amax(-1, keepdim=True)) & allowed,
        )
    keep = active.to(S.dtype)
    outer = keep.unsqueeze(-1) * keep.unsqueeze(-2)
    system = Sff * outer + torch.diag_embed(1.0 - keep)
    rhs = (af * keep).unsqueeze(-1)
    if face_factor is None:
        v = torch.linalg.solve(system, rhs).squeeze(-1) * keep
    else:
        same_active = (active == allowed).all(-1)
        v = torch.zeros_like(af)
        if bool(same_active.any()):
            cached_v = (
                torch.cholesky_solve(
                    rhs[same_active], face_factor[same_active]
                ).squeeze(-1)
                * keep[same_active]
            )
            v[same_active] = cached_v
        if bool((~same_active).any()):
            solved_v = (
                torch.linalg.solve(system[~same_active], rhs[~same_active]).squeeze(-1)
                * keep[~same_active]
            )
            v[~same_active] = solved_v
    s = v.sum(-1, keepdim=True)
    normalized = v / s.clamp_min(1e-14)
    fallback_keep = allowed.to(v.dtype)
    fallback_count = fallback_keep.sum(-1, keepdim=True)
    fallback = torch.where(
        fallback_count > 0,
        fallback_keep / fallback_count.clamp_min(1),
        torch.full_like(v, 1.0 / k2),
    )
    wf = torch.where(s > 1e-14, normalized, fallback)
    w = torch.zeros_like(mu).scatter_(1, face, wf)
    val = torch.sqrt(torch.relu((af * v).sum(-1)))
    return (w, val)
