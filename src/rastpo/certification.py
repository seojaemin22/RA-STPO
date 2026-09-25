"""Exact sparse Sharpe certificates with explicit unresolved states.

The default optimization target is the empirical Sharpe ratio of a stored
finite excess-return table under a hard cardinality budget

    S0(w) = mu' w / ||C0 w||_2,    w >= 0,  1'w = 1,  ||w||_0 <= m,

with ``mu = R'1/T``, ``X = R - 1 mu'``, ``c = T-1`` and ``C0 = X/sqrt(c)``.  No
ridge or annualization is added by default. Rational covariance and mean maps
can be declared explicitly. Stored binary64 entries are
interpreted as their exact dyadic rationals, so every bound this module emits is
produced by *integer* arithmetic and only converted to binary64 with directed
(outward) rounding at the very end.

Certificate statuses and reasons
--------------------------------
``PRIMARY_STATUSES``  FINITE_CERTIFIED, SPARSE_INFINITE_CERTIFIED, NON_EPER,
                      UNDEFINED, OPEN_NO_FINITE_BOUND, RESOURCE_OPEN, INVALID
``REASONS``           NONE, NO_DEFINED_SHARPE_POINT, ALL_MEANS_NONPOSITIVE,
                      DENSE_RELAXATION_RAY, RESOURCE_LIMIT, INPUT_CONTRACT,
                      SERIALIZATION, NUMERIC_VERIFICATION

``FINITE_CERTIFIED`` confirms a closed finite optimum, and
``SPARSE_INFINITE_CERTIFIED`` carries an infinity witness. Unresolved states may
retain valid lower and upper bounds without certifying global optimality. A dense
relaxation null ray alone never produces ``SPARSE_INFINITE_CERTIFIED``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import hashlib
import json
import math
from itertools import combinations
from time import perf_counter
from typing import Sequence

import numpy as np


SCHEMA = "SAFER_SR_CERTIFICATE_V1"

PRIMARY_STATUSES = (
    "FINITE_CERTIFIED",
    "SPARSE_INFINITE_CERTIFIED",
    "NON_EPER",
    "UNDEFINED",
    "OPEN_NO_FINITE_BOUND",
    "RESOURCE_OPEN",
    "INVALID",
)
REASONS = (
    "NONE",
    "NO_DEFINED_SHARPE_POINT",
    "ALL_MEANS_NONPOSITIVE",
    "DENSE_RELAXATION_RAY",
    "RESOURCE_LIMIT",
    "INPUT_CONTRACT",
    "SERIALIZATION",
    "NUMERIC_VERIFICATION",
)


# ---------------------------------------------------------------------------
# exact integer model of the stored table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExactModel:
    """Integer-exact representation of ``(mu, X)`` for a stored binary64 table.

    Every stored return is a dyadic rational ``R[t,i] = Rint[t,i] / 2**K``.  With
    ``s_i = sum_t Rint[t,i]`` and ``D = T * 2**K`` we have the *exact* identities

        mu = s / D,      X = Bm / D,      Bm[t,i] = T*Rint[t,i] - s_i,

    so ``Sigma = X'X/c = Bm'Bm/(c D^2)``.  ``Bm``, ``s`` and ``D`` are integers,
    which lets the whole certificate run in exact integer arithmetic.

    The solver never needs the ``T x N`` factor itself, only the integer Gram
    ``M = Bm'Bm`` and the scale ``kappa`` with ``Sigma = M / kappa``.  Carrying
    ``(s, D, M, kappa)`` instead of ``(s, D, Bm)`` is what lets a *declared
    target* -- a shrunk covariance, a ridge, a shrunk mean -- be represented
    exactly as well, because those maps are integer maps on ``(s, D, M, kappa)``
    even when they are not representable as any stored binary64 table.
    """

    T: int
    N: int
    c: int
    D: int  # mu = s / D
    s: tuple[int, ...]  # N integer scaled column sums
    M: tuple[tuple[int, ...], ...]  # N x N integer Gram,  Sigma = M / kappa
    kappa: int  # positive integer scale
    target: tuple[tuple[str, str], ...] = ()  # provenance of a declared target
    floor: tuple[int, ...] = ()  # Sigma >= Diag(floor)/kappa, entrywise ints

    def __post_init__(self) -> None:
        if not self.floor:
            object.__setattr__(self, "floor", (0,) * self.N)

    def col_is_zero(self, i: int) -> bool:
        """``Sigma_ii == 0``, i.e. asset ``i`` has exactly zero sample variance."""
        return self.M[i][i] == 0

    # -- declared-target maps.  Each is an exact integer map; each records its
    # -- provenance in ``target`` so the independent verifier can rebuild it.
    def shrink_covariance(
        self, lam: Fraction, toward: str = "DIAGONAL"
    ) -> "ExactModel":
        r"""``Sigma_lam = (1-lam) Sigma + lam * Target``, exactly.

        ``DIAGONAL`` uses ``Target = diag(Sigma)`` (drops correlations, keeps
        variances).  ``IDENTITY`` uses ``Target = (tr Sigma / N) I`` (the
        Ledoit--Wolf scaled-identity target).  Both are integer maps:
        with ``lam = p/q``, ``DIAGONAL`` gives ``M' = (q-p) M`` off-diagonal and
        ``q M_ii`` on the diagonal with ``kappa' = q kappa``; ``IDENTITY`` gives
        ``M' = (q-p) N M + p (tr M) I`` with ``kappa' = q N kappa``.
        """
        lam = Fraction(lam)
        if not 0 <= lam <= 1:
            raise ValueError("lam must lie in [0, 1]")
        p, q = lam.numerator, lam.denominator
        a, N = q - p, self.N
        if toward == "DIAGONAL":
            rows = tuple(
                tuple(
                    q * self.M[i][i] if i == j else a * self.M[i][j] for j in range(N)
                )
                for i in range(N)
            )
            kappa = q * self.kappa
            floor = tuple(a * self.floor[i] + p * self.M[i][i] for i in range(N))
        elif toward == "IDENTITY":
            trace = sum(self.M[i][i] for i in range(N))
            rows = tuple(
                tuple(
                    a * N * self.M[i][j] + (p * trace if i == j else 0)
                    for j in range(N)
                )
                for i in range(N)
            )
            kappa = q * N * self.kappa
            floor = tuple(a * N * self.floor[i] + p * trace for i in range(N))
        else:
            raise ValueError(f"unknown shrinkage target {toward!r}")
        return replace(
            self,
            M=rows,
            kappa=kappa,
            floor=floor,
            target=self.target + (("SHRINK_" + toward, str(lam)),),
        )

    def ridge(self, eps: Fraction) -> "ExactModel":
        r"""``Sigma + eps I`` exactly: ``M' = q M + p kappa I``, ``kappa' = q kappa``."""
        eps = Fraction(eps)
        if eps < 0:
            raise ValueError("eps must be nonnegative")
        p, q = eps.numerator, eps.denominator
        rows = tuple(
            tuple(
                q * self.M[i][j] + (p * self.kappa if i == j else 0)
                for j in range(self.N)
            )
            for i in range(self.N)
        )
        return replace(
            self,
            M=rows,
            kappa=q * self.kappa,
            floor=tuple(q * self.floor[i] + p * self.kappa for i in range(self.N)),
            target=self.target + (("RIDGE", str(eps)),),
        )

    def shrink_mean(self, z: Fraction) -> "ExactModel":
        r"""``mu_z = (1-z) mu + z * (mean mu) 1``, exactly, with ``Sigma`` untouched.

        ``s' = (q-p) N s + p (sum s) 1`` and ``D' = q N D``. Only the mean
        changes; the covariance Gram matrix is preserved.
        """
        z = Fraction(z)
        if not 0 <= z <= 1:
            raise ValueError("z must lie in [0, 1]")
        p, q = z.numerator, z.denominator
        N, total = self.N, sum(self.s)
        s = tuple((q - p) * N * self.s[i] + p * total for i in range(N))
        return replace(
            self,
            s=s,
            D=q * N * self.D,
            target=self.target + (("SHRINK_MEAN", str(z)),),
        )

    def james_stein_intensity(self) -> Fraction:
        r"""Exact James--Stein shrinkage intensity towards the grand mean.

        ``z = min(1, (N-2) * (tr Sigma / N) / (T * ||mu - bar mu 1||^2))``, which
        in the integer model is the exact rational

            z = (N-2) N (tr M) D^2 / (kappa T sum_i (N s_i - sum s)^2).

        The intensity is computed from the supplied return window.
        """
        total = sum(self.s)
        dispersion = sum((self.N * si - total) ** 2 for si in self.s)
        if dispersion == 0:
            return Fraction(1)
        trace = sum(self.M[i][i] for i in range(self.N))
        z = Fraction(
            (self.N - 2) * self.N * trace * self.D * self.D,
            self.kappa * self.T * dispersion,
        )
        return min(Fraction(1), max(Fraction(0), z))

    def exponential_mean(self, halflife: int) -> "ExactModel":
        r"""Replace ``mu`` by a dyadically weighted mean, ``Sigma`` untouched.

        Weight ``W_t = 2**(t // h)`` doubles every ``h`` periods, so the most
        recent block counts most.  Powers of two keep every product exact, and
        the weights are a function of ``(T, h)`` alone, so the verifier rebuilds
        them from the declared half-life.  Requires the per-period rows, so it is
        applied at build time by :func:`build_exact_model`.
        """
        raise NotImplementedError("use build_exact_model(returns, halflife=h)")

    def weight_covariance(self, halflife: int) -> "ExactModel":
        r"""Exponentially weighted covariance, exactly, with ``mu`` untouched.

        ``M'_{ij} = sum_t W_t B_{ti} B_{tj}`` with ``W_t = 2**(t // h)``: dyadic
        weights keep every product integral.  Requires the per-period rows, so it
        is applied at build time by :func:`build_exact_model`.
        """
        raise NotImplementedError("use build_exact_model(returns, cov_halflife=h)")

    def add_mean_outer(self, beta: Fraction) -> "ExactModel":
        r"""``Sigma + beta mu mu'`` exactly, interpolating to the second moment.

        ``beta = 1`` gives the *uncentered* second moment ``R'R/T`` used by some
        comparators.  With ``beta = p/q``:
        ``M' = q D^2 M + p kappa s s'``, ``kappa' = q kappa D^2``.
        """
        beta = Fraction(beta)
        if beta < 0:
            raise ValueError("beta must be nonnegative")
        p, q = beta.numerator, beta.denominator
        square = q * self.D * self.D
        rows = tuple(
            tuple(
                square * self.M[i][j] + p * self.kappa * self.s[i] * self.s[j]
                for j in range(self.N)
            )
            for i in range(self.N)
        )
        return replace(
            self,
            M=rows,
            kappa=square * self.kappa,
            floor=tuple(square * value for value in self.floor),
            target=self.target + (("MEAN_OUTER", str(beta)),),
        )


def cardinality_floor_bound(
    model: ExactModel, allowed: Sequence[int], m: int
) -> Fraction | None:
    r"""Cardinality-*aware* lower bound on ``q`` over ``{v>=0, supp(v) subset A,
    ||v||_0 <= m}``, valid whenever ``Sigma >= Diag(floor)/kappa``.

    With ``Lambda = Diag(floor)/kappa`` we have ``q(v) >= v'Lambda v/2 - mu'v``,
    whose minimum over the cardinality-constrained nonnegative cone is separable:
    coordinate ``i`` contributes ``-mu_i^2 / (2 Lambda_ii)`` when ``mu_i > 0`` and
    ``0`` otherwise, so the minimum keeps the ``m`` most negative contributions

        q_card(A) = -(1/2) * sum of the m largest  kappa s_i^2 / (D^2 floor_i).

    Unlike the relaxation bound this one *knows about the budget*, so it stays
    finite where the relaxation is unbounded.  It needs no witness at all: the
    verifier recomputes it from ``(s, D, floor, kappa)``, which it re-derives
    from the raw table and the declared target.  ``None`` means vacuous -- some
    admissible coordinate has a zero floor, which is always the case for the
    unregularised table itself.
    """
    terms: list[Fraction] = []
    for i in allowed:
        if model.s[i] <= 0:
            continue
        if model.floor[i] <= 0:
            return None
        terms.append(
            Fraction(
                model.kappa * model.s[i] * model.s[i],
                model.D * model.D * model.floor[i],
            )
        )
    if not terms:
        return Fraction(0)
    terms.sort(reverse=True)
    return -sum(terms[:m]) / 2


def build_exact_model(
    returns: np.ndarray,
    halflife: int | None = None,
    *,
    cov_halflife: int | None = None,
    mean_window: int | None = None,
) -> ExactModel:
    """Exact integer model of a stored table.

    ``mean_window`` estimates ``mu`` from the last ``mean_window`` rows only while
    ``Sigma`` still uses the whole table. ``cov_halflife`` replaces the covariance
    by the dyadically weighted one, ``M' = sum_t 2**(t//h) B_t B_t'``.

    ``halflife`` replaces the equal-weight mean by the dyadic weighted mean
    ``W_t = 2**(t // halflife)`` while leaving ``Sigma`` untouched. The weights are
    powers of two, so every product stays exact, and they are a function of
    ``(T, halflife)`` alone, so the verifier rebuilds them from the record.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] < 1:
        raise ValueError("returns must be a T x N table with T >= 2")
    if not np.isfinite(r).all():
        raise ValueError("returns must be finite")
    T, N = r.shape
    # common power-of-two scale making every entry an integer
    exponent = 0
    for value in r.reshape(-1):
        if value == 0.0:
            continue
        _, e = math.frexp(value)
        # value = frac * 2**e with 0.5 <= |frac| < 1 ; 53 significand bits
        exponent = max(exponent, 53 - e)
    exponent = max(exponent, 0)
    scale = 1 << exponent
    # Multiplying a binary64 by a power of two is exact unless it overflows, so an
    # integral product certifies the rescaling without any rational arithmetic.
    rint: list[list[int]] = []
    for i in range(N):
        column = []
        for t in range(T):
            scaled = float(r[t, i]) * scale
            if not math.isfinite(scaled) or scaled != math.floor(scaled):
                raise ValueError("integer rescaling was not exact")
            column.append(int(scaled))
        rint.append(column)
    s = tuple(sum(rint[i]) for i in range(N))
    Bm = tuple(tuple(T * rint[i][t] - s[i] for t in range(T)) for i in range(N))
    # The centred factor uses the equal-weight mean; a declared weighted mean
    # changes only the numerator (s, D) of the Sharpe ratio, never Sigma.
    weighted_s, weighted_D, provenance = None, None, ()
    if halflife is not None:
        h = int(halflife)
        if h < 1:
            raise ValueError("halflife must be a positive integer")
        weights = [1 << (t // h) for t in range(T)]
        weighted_s = tuple(
            sum(w * rint[i][t] for t, w in enumerate(weights)) for i in range(N)
        )
        weighted_D = sum(weights)
        provenance = (("MEAN_EWMA", str(h)),)
    if mean_window is not None:
        k = int(mean_window)
        if not 1 <= k <= T:
            raise ValueError("mean_window must lie in 1..T")
        weighted_s = tuple(sum(rint[i][T - k :]) for i in range(N))
        weighted_D = k
        provenance = (("MEAN_WINDOW", str(k)),)
    # exact integer Gram; Sigma = Bm'Bm / (c D^2) holds entrywise by construction
    gram = [[0] * N for _ in range(N)]
    for i in range(N):
        bi = Bm[i]
        for j in range(i, N):
            value = sum(x * y for x, y in zip(bi, Bm[j]))
            gram[i][j] = value
            gram[j][i] = value
    c = T - 1
    model = ExactModel(
        T=T,
        N=N,
        c=c,
        D=T * scale,
        s=s,
        M=tuple(tuple(row) for row in gram),
        kappa=c * (T * scale) ** 2,
    )
    if weighted_s is not None:
        model = replace(model, s=weighted_s, D=weighted_D * scale, target=provenance)
    if cov_halflife is not None:
        h = int(cov_halflife)
        if h < 1:
            raise ValueError("cov_halflife must be positive")
        coefficients = [1 << (t // h) for t in range(T)]
        rows = []
        for i in range(N):
            bi = Bm[i]
            rows.append(
                tuple(
                    sum(w * bi[t] * Bm[j][t] for t, w in enumerate(coefficients))
                    for j in range(N)
                )
            )
        total = sum(coefficients)
        model = replace(
            model,
            M=tuple(rows),
            kappa=total * (T * scale) ** 2,
            target=model.target + (("COV_EWMA", str(h)),),
        )
    return model


# ---------------------------------------------------------------------------
# exact rational helpers
# ---------------------------------------------------------------------------
def _solve_exact(
    matrix: list[list[Fraction]], rhs: list[Fraction]
) -> list[Fraction] | None:
    """Gauss-Jordan with exact rationals; ``None`` when the system is singular."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
        if pivot is None:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        inverse = 1 / a[col][col]
        a[col] = [value * inverse for value in a[col]]
        for row in range(n):
            if row != col and a[row][col] != 0:
                factor = a[row][col]
                a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [a[i][n] for i in range(n)]


def _clear_denominators(values: Sequence[Fraction]) -> tuple[list[int], int]:
    """Return integers ``u`` and ``g > 0`` with ``values == u / g``."""
    g = 1
    for value in values:
        d = value.denominator
        g = g * d // math.gcd(g, d)
    return [int(value * g) for value in values], g


def _sqrt_down(value: Fraction) -> float:
    """Largest binary64 not exceeding ``sqrt(value)`` for ``value >= 0``."""
    if value < 0:
        raise ValueError("negative radicand")
    if value == 0:
        return 0.0
    x = math.sqrt(float(value))
    if not math.isfinite(x):
        return math.inf
    while x > 0 and Fraction(x) * Fraction(x) > value:
        x = math.nextafter(x, 0.0)
    while True:
        up = math.nextafter(x, math.inf)
        if Fraction(up) * Fraction(up) <= value:
            x = up
        else:
            return x


def _sqrt_up(value: Fraction) -> float:
    """Smallest binary64 not below ``sqrt(value)`` for ``value >= 0``."""
    if value < 0:
        raise ValueError("negative radicand")
    if value == 0:
        return 0.0
    x = math.sqrt(float(value))
    if not math.isfinite(x):
        return math.inf
    while Fraction(x) * Fraction(x) < value:
        x = math.nextafter(x, math.inf)
    while True:
        down = math.nextafter(x, 0.0)
        if down > 0 and Fraction(down) * Fraction(down) >= value:
            x = down
        else:
            return x


# ---------------------------------------------------------------------------
# exact node relaxation:  min_{v >= 0, supp(v) subset A} q(v)
# ---------------------------------------------------------------------------
@dataclass
class NodeSolution:
    """Exact outcome of the cardinality-free node relaxation on ``A``."""

    kind: str  # OPTIMAL | RAY | SINGULAR_OPEN | ITERATION_CAP
    q: Fraction | None = None  # exact optimal value when kind == OPTIMAL
    support: tuple[int, ...] = ()  # indices with strictly positive weight
    u: tuple[int, ...] = ()  # integer numerators aligned with ``support``
    g: int = 1  # common denominator, v[support] = u/g
    ray: tuple[int, ...] = ()  # support of a positive-return null ray
    ray_u: tuple[int, ...] = ()  # exact positive numerators on that support
    iterations: int = 0


def _gram(model: ExactModel, free: Sequence[int]) -> list[list[Fraction]]:
    return [[Fraction(model.M[i][j]) for j in free] for i in free]


def _integer_null_ray(model: ExactModel, free: Sequence[int]):
    """Search a nonnegative zero-variance direction with positive mean.

    Returns (support, integer numerators), or None. The direction is exact: a rational
    ``d >= 0`` supported on ``free`` with ``d' Sigma d = 0`` and ``mu' d > 0``.
    Since every model here has ``Sigma`` positive *semi*definite,
    ``d' Sigma_FF d = 0`` iff ``M_FF d = 0``, so the null space of the ``k x k``
    integer Gram is the right object -- and it is the same set as the null space
    of the ``T x k`` factor whenever a factor exists.
    """
    k = len(free)
    if k == 0:
        return None
    # exact reduced row echelon form of the k x k integer Gram
    rows = [[Fraction(model.M[i][j]) for j in free] for i in free]
    pivots: list[int] = []
    row = 0
    for col in range(k):
        sel = next((r for r in range(row, len(rows)) if rows[r][col] != 0), None)
        if sel is None:
            continue
        rows[row], rows[sel] = rows[sel], rows[row]
        inverse = 1 / rows[row][col]
        rows[row] = [value * inverse for value in rows[row]]
        for r in range(len(rows)):
            if r != row and rows[r][col] != 0:
                factor = rows[r][col]
                rows[r] = [x - factor * y for x, y in zip(rows[r], rows[row])]
        pivots.append(col)
        row += 1
        if row == len(rows):
            break
    freecols = [c for c in range(k) if c not in pivots]
    if not freecols:
        return None
    basis: list[list[Fraction]] = []
    for fc in freecols:
        vec = [Fraction(0)] * k
        vec[fc] = Fraction(1)
        for r, pc in enumerate(pivots):
            vec[pc] = -rows[r][fc]
        basis.append(vec)
    means = [Fraction(model.s[i]) for i in free]
    # nonnegative element of the null space with positive mean: try each basis
    # vector and its negation, then a simple positive combination.
    candidates = []
    for vec in basis:
        candidates.append(vec)
        candidates.append([-x for x in vec])
    total = [sum(col) for col in zip(*basis)] if len(basis) > 1 else None
    if total is not None:
        candidates.append(total)
        candidates.append([-x for x in total])
    for vec in candidates:
        if all(x >= 0 for x in vec) and any(x > 0 for x in vec):
            if sum(m * x for m, x in zip(means, vec)) > 0:
                positive = [j for j in range(k) if vec[j] > 0]
                numerators, _ = _clear_denominators([vec[j] for j in positive])
                return (tuple(free[j] for j in positive), tuple(numerators))
    return None


def solve_node(
    model: ExactModel,
    allowed: Sequence[int],
    *,
    max_iterations: int = 200,
    warm_support: Sequence[int] | None = None,
) -> NodeSolution:
    """Exactly solve ``min q(v)`` over ``v >= 0`` with ``supp(v) subset allowed``.

    Lawson--Hanson style active set entirely in integer/rational arithmetic.  The
    returned ``OPTIMAL`` value is the *exact* node lower bound for every
    cardinality-feasible descendant.
    """
    allowed = list(allowed)
    index_of = {i: p for p, i in enumerate(allowed)}
    D, kappa = model.D, model.kappa

    # pre-screen an exact singleton null ray (zero variance, positive mean)
    for i in allowed:
        if model.col_is_zero(i) and model.s[i] > 0:
            return NodeSolution(kind="RAY", ray=(i,), ray_u=(1,))

    free: list[int] = []
    if warm_support:
        free = [i for i in warm_support if i in index_of]
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        if free:
            # Sigma_FF v_F = mu_F  <=>  M_FF (D v_F) = kappa s_F   (integer RHS)
            gram = _gram(model, free)
            rhs = [Fraction(kappa * model.s[i]) for i in free]
            solution = _solve_exact(gram, rhs)
            if solution is None:
                ray = _integer_null_ray(model, free)
                if ray:
                    return NodeSolution(
                        kind="RAY", ray=ray[0], ray_u=ray[1], iterations=iterations
                    )
                # linearly dependent but no positive-return ray: shrink the set
                free = free[:-1]
                continue
            if any(value <= 0 for value in solution):
                worst = min(range(len(free)), key=lambda a: solution[a])
                free.pop(worst)
                continue
            # solution solves for D*v_F, so v_F = u/(g*D)
            u, g = _clear_denominators(solution)
            g *= D
        else:
            u, g = [], 1
            solution = []
        # exact gradient numerator  z = D (M u)_i - kappa g s_i   (sign of Sigma v - mu)
        if free:
            product = {
                i: sum(model.M[i][j] * uj for j, uj in zip(free, u)) for i in allowed
            }
        else:
            product = {i: 0 for i in allowed}
        violated = None
        worst_value = 0
        for i in allowed:
            if i in free:
                continue
            z = D * product[i] - kappa * g * model.s[i]
            if z < worst_value:
                worst_value = z
                violated = i
        if violated is None:
            if free:
                quad = sum(uj * product[i] for i, uj in zip(free, u))  # u' M u
                lin = sum(model.s[i] * uj for i, uj in zip(free, u))  # s' u
                q = Fraction(D * quad - 2 * kappa * g * lin, 2 * kappa * g * g * D)
            else:
                q = Fraction(0)
            return NodeSolution(
                kind="OPTIMAL",
                q=q,
                support=tuple(free),
                u=tuple(u),
                g=g,
                iterations=iterations,
            )
        free.append(violated)
        if len(free) > len(allowed):  # pragma: no cover - defensive
            break
    return NodeSolution(kind="ITERATION_CAP", iterations=iterations)


# ---------------------------------------------------------------------------
# maximum-weight cap:  w_i <= b  becomes  v_i <= b * 1'v  on the homogenised
# cone, a *homogeneous* linear inequality, so the feasible set stays a finite
# union of closed convex cones, so homogenization still applies.
# ---------------------------------------------------------------------------
def solve_node_capped(
    model: ExactModel,
    allowed: Sequence[int],
    cap: Fraction,
    *,
    max_iterations: int = 200,
) -> NodeSolution:
    r"""Exactly solve ``min q(v)`` over ``v >= 0``, ``supp(v) subset allowed``,
    ``v_i <= cap * 1'v``.

    The Lagrangian ``L = q(v) - s'v + sum_i g_i (v_i - b 1'v)`` gives, with
    ``G = sum_i g_i``,

        (Sigma v - mu)_k  = b G     for free k    (v_k > 0, cap slack)
        (Sigma v - mu)_k <= b G     for capped k  (v_k = b 1'v)
        (Sigma v - mu)_k >= b G     for zero k    (v_k = 0)

    i.e. the uncapped system with the threshold ``0`` replaced by ``bG``, and
    ``G`` pinned by summing the capped rows,
    ``G (1 - |C| b) = - sum_{k in C} (Sigma v - mu)_k``.

    A primal-dual active-set loop adds violated bounds and releases those whose
    multiplier turns negative. The returned node is accepted only after
    :func:`_capped_certificate` verifies the three conditions in exact arithmetic.
    """
    allowed = list(allowed)
    cap = Fraction(cap)
    if cap <= 0:
        raise ValueError("cap must be positive")
    if cap >= 1:
        return solve_node(model, allowed, max_iterations=max_iterations)
    D, kappa = model.D, model.kappa

    for i in allowed:
        if model.col_is_zero(i) and model.s[i] > 0:
            return NodeSolution(kind="RAY", ray=(i,), ray_u=(1,))

    seed = solve_node(model, allowed, max_iterations=max_iterations)
    if seed.kind != "OPTIMAL" or seed.q is None:
        return seed
    free: list[int] = [i for i in seed.support]
    capped: list[int] = []
    if not free:
        return NodeSolution(kind="OPTIMAL", q=Fraction(0), iterations=1)
    # Anti-cycling: once an active set repeats, switch to Bland's rule (smallest
    # index wins every tie), which cannot cycle.
    seen: set[tuple[frozenset, frozenset]] = set()
    bland = False

    for iteration in range(1, max_iterations + 1):
        state = (frozenset(free), frozenset(capped))
        if state in seen:
            bland = True
        seen.add(state)
        k, count = len(free), len(capped)
        denominator = 1 - count * cap
        if denominator <= 0 or k == 0:
            return NodeSolution(kind="ITERATION_CAP", iterations=iteration)
        # --- equality system in (v_free, G), all rows scaled by kappa*den -----
        rows: list[list[Fraction]] = []
        rhs: list[Fraction] = []
        for a in free:
            shared = cap * sum(Fraction(model.M[a][c]) for c in capped)
            row = [Fraction(model.M[a][f]) * denominator + shared for f in free]
            row.append(-cap * kappa * denominator)
            rows.append(row)
            rhs.append(Fraction(kappa * model.s[a], D) * denominator)
        if capped:
            row = [Fraction(0)] * (k + 1)
            total = Fraction(0)
            for c in capped:
                shared = cap * sum(Fraction(model.M[c][d]) for d in capped)
                for j, f in enumerate(free):
                    row[j] += Fraction(model.M[c][f]) * denominator + shared
                total += Fraction(kappa * model.s[c], D) * denominator
            row[k] = kappa * denominator * denominator
            rows.append(row)
            rhs.append(total)
        else:
            row = [Fraction(0)] * (k + 1)
            row[k] = Fraction(1)
            rows.append(row)
            rhs.append(Fraction(0))
        solution = _solve_exact(rows, rhs)
        if solution is None:
            return NodeSolution(kind="ITERATION_CAP", iterations=iteration)
        v_free, multiplier = solution[:k], solution[k]

        # --- primal: drop nonpositive, then bind the worst cap violation -----
        if any(value <= 0 for value in v_free):
            if bland:
                free.pop(
                    min((j for j in range(k) if v_free[j] <= 0), key=lambda j: free[j])
                )
            else:
                free.pop(min(range(k), key=lambda j: v_free[j]))
            continue
        theta = sum(v_free) / denominator
        violator, worst = None, Fraction(0)
        for j, f in enumerate(free):
            gap = v_free[j] - cap * theta
            if gap > worst if not bland else gap > 0:
                worst, violator = gap, f
                if bland:
                    break
        if violator is not None:
            free.remove(violator)
            capped.append(violator)
            continue

        # --- dual: release a cap whose multiplier went negative, then add ----
        values = {f: v_free[j] for j, f in enumerate(free)}
        for c in capped:
            values[c] = cap * theta
        threshold = cap * multiplier

        def gradient(i: int) -> Fraction:
            return sum(
                Fraction(model.M[i][j]) * value for j, value in values.items()
            ) / kappa - Fraction(model.s[i], D)

        release, worst_dual = None, Fraction(0)
        for c in sorted(capped) if bland else capped:
            gamma = threshold - gradient(c)
            if gamma < worst_dual if not bland else gamma < 0:
                worst_dual, release = gamma, c
                if bland:
                    break
        if release is not None:
            capped.remove(release)
            free.append(release)
            continue
        enter, worst_primal = None, Fraction(0)
        for i in sorted(allowed) if bland else allowed:
            if i in values:
                continue
            slack = gradient(i) - threshold
            if slack < worst_primal if not bland else slack < 0:
                worst_primal, enter = slack, i
                if bland:
                    break
        if enter is not None:
            free.append(enter)
            continue

        support = tuple(sorted(values))
        u, g = _clear_denominators([values[i] for i in support])
        return _capped_certificate(model, allowed, cap, support, u, g, iteration)
    return NodeSolution(kind="ITERATION_CAP", iterations=max_iterations)


def _capped_certificate(
    model: ExactModel,
    allowed: Sequence[int],
    cap: Fraction,
    support: Sequence[int],
    u: Sequence[int],
    g: int,
    iterations: int,
) -> NodeSolution:
    """Verify the capped KKT system exactly; emit a bound only if it passes."""
    D, kappa = model.D, model.kappa
    positions = {i: p for p, i in enumerate(support)}
    theta = Fraction(sum(u), g)
    product = {
        i: sum(model.M[i][j] * u[positions[j]] for j in support) for i in allowed
    }

    # gradient numerators, exact:  (Sigma v - mu)_i = (D (Mu)_i - kappa g s_i)/(kappa g D)
    def gradient(i: int) -> Fraction:
        return Fraction(D * product[i] - kappa * g * model.s[i], kappa * g * D)

    capped = [i for i in support if Fraction(u[positions[i]], g) == cap * theta]
    free = [i for i in support if i not in capped]
    if capped:
        denominator = 1 - len(capped) * cap
        if denominator <= 0:
            return NodeSolution(kind="ITERATION_CAP", iterations=iterations)
        multiplier = -sum(gradient(i) for i in capped) / denominator
    else:
        multiplier = Fraction(0)
    threshold = cap * multiplier
    for i in free:  # stationarity on the free block
        if gradient(i) != threshold:
            return NodeSolution(kind="ITERATION_CAP", iterations=iterations)
    for i in capped:  # cap multiplier must be >= 0
        if threshold - gradient(i) < 0:
            return NodeSolution(kind="ITERATION_CAP", iterations=iterations)
    for i in allowed:  # zero block
        if i in positions:
            continue
        if gradient(i) < threshold:
            return NodeSolution(kind="ITERATION_CAP", iterations=iterations)
    if any(x <= 0 for x in u) or g <= 0:
        return NodeSolution(kind="ITERATION_CAP", iterations=iterations)
    for i in support:  # feasibility of the cap itself
        if Fraction(u[positions[i]], g) > cap * theta:
            return NodeSolution(kind="ITERATION_CAP", iterations=iterations)
    quad = sum(u[positions[i]] * product[i] for i in support)
    lin = sum(model.s[i] * u[positions[i]] for i in support)
    q = Fraction(D * quad - 2 * kappa * g * lin, 2 * kappa * g * g * D)
    return NodeSolution(
        kind="OPTIMAL",
        q=q,
        support=tuple(support),
        u=tuple(u),
        g=g,
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# certificate
# ---------------------------------------------------------------------------
@dataclass
class SaferConfig:
    cardinality: int = 4
    max_weight: object = None  # Fraction cap b with w_i <= b, or None
    max_nodes: int = 200
    max_node_iterations: int = 200
    max_ray_support: int = 12
    max_seconds: float = 60.0


@dataclass
class SaferCertificate:
    status: str
    reason: str
    cardinality: int
    n_assets: int
    n_periods: int
    weights: np.ndarray | None
    sharpe_lower: float  # L  (outward, <= S0(what)) or nan
    sharpe_upper: float  # U  (outward, >= S*) or nan/inf
    incumbent_sharpe_lower: float
    q_lower: float
    q_upper: float
    support: tuple[int, ...]
    nodes_processed: int
    root_certified: bool
    exact_zero_gap: bool
    ray_support: tuple[int, ...]
    elapsed: float
    diagnostics: dict = field(default_factory=dict)

    # exact objects retained for the independent verifier
    exact: dict = field(default_factory=dict, repr=False)

    def as_record(self) -> dict:
        record = {
            "schema": SCHEMA,
            "status": self.status,
            "reason": self.reason,
            "cardinality": self.cardinality,
            "n_assets": self.n_assets,
            "n_periods": self.n_periods,
            "support": list(self.support),
            "sharpe_lower": self.sharpe_lower,
            "sharpe_upper": self.sharpe_upper,
            "incumbent_sharpe_lower": self.incumbent_sharpe_lower,
            "q_lower": self.q_lower,
            "q_upper": self.q_upper,
            "nodes_processed": self.nodes_processed,
            "root_certified": self.root_certified,
            "exact_zero_gap": self.exact_zero_gap,
            "ray_support": list(self.ray_support),
            "weights_hex": (
                [float(x).hex() for x in self.weights]
                if self.weights is not None
                else None
            ),
            "exact": self.exact,
        }
        return record

    def digest(self) -> str:
        payload = json.dumps(self.as_record(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _weights_from_support(
    model: ExactModel, support: Sequence[int], u: Sequence[int]
) -> np.ndarray:
    w = np.zeros(model.N, dtype=np.float64)
    total = sum(u)
    if total <= 0:
        return w
    for i, ui in zip(support, u):
        w[i] = float(Fraction(ui, total))
    # renormalise defensively in binary64; the exact rational vector is retained
    if w.sum() > 0:
        w = w / w.sum()
    return w


def _greedy_supports(
    model: ExactModel, m: int, limit: int = 3
) -> list[tuple[int, ...]]:
    """Positive-mean filtered ``max(mu_i,0)^2 / Sigma_ii`` ranking (and prefixes)."""
    scores: list[tuple[Fraction, int]] = []
    for i in range(model.N):
        if model.s[i] <= 0:
            continue
        diag = model.M[i][i]
        if diag == 0:
            continue
        # mu_i^2 / Sigma_ii = kappa s_i^2 / (D^2 M_ii); the D^2 is a common factor
        scores.append((Fraction(model.s[i] * model.s[i] * model.kappa, diag), i))
    scores.sort(key=lambda kv: (-kv[0], kv[1]))
    ordered = [i for _, i in scores]
    out: list[tuple[int, ...]] = []
    for size in {m, max(1, m // 2), 1}:
        if ordered[:size]:
            out.append(tuple(sorted(ordered[:size])))
    return out[:limit]


def certify(
    returns: np.ndarray,
    config: SaferConfig = SaferConfig(),
    *,
    model: ExactModel | None = None,
) -> SaferCertificate:
    """Return the status-complete SAFER-SR certificate for one stored window.

    Passing ``model`` certifies the supplied mean and covariance transformations.
    ``model.target`` records these transformations so the independent verifier
    can reconstruct the objective from the raw return table.
    """
    started = perf_counter()
    m = int(config.cardinality)
    try:
        model = build_exact_model(returns) if model is None else model
    except ValueError:
        return SaferCertificate(
            status="INVALID",
            reason="INPUT_CONTRACT",
            cardinality=m,
            n_assets=0,
            n_periods=0,
            weights=None,
            sharpe_lower=math.nan,
            sharpe_upper=math.nan,
            incumbent_sharpe_lower=math.nan,
            q_lower=math.nan,
            q_upper=math.nan,
            support=(),
            nodes_processed=0,
            root_certified=False,
            exact_zero_gap=False,
            ray_support=(),
            elapsed=perf_counter() - started,
        )
    N = model.N
    m = min(max(m, 1), N)
    cap = None if config.max_weight is None else Fraction(config.max_weight)
    if cap is not None and cap * m <= 1:
        # a cap of b needs more than 1/b positions to leave any slack; with
    # m*b <= 1 the feasible set is empty or restricted to equal-weight portfolios.
    # This implementation requires m*b > 1.
        return SaferCertificate(
            status="INVALID",
            reason="INPUT_CONTRACT",
            cardinality=m,
            n_assets=N,
            n_periods=model.T,
            weights=None,
            sharpe_lower=math.nan,
            sharpe_upper=math.nan,
            incumbent_sharpe_lower=math.nan,
            q_lower=math.nan,
            q_upper=math.nan,
            support=(),
            nodes_processed=0,
            root_certified=False,
            exact_zero_gap=False,
            ray_support=(),
            elapsed=perf_counter() - started,
        )

    def solve(allowed, warm=None):
        """Node solve under the declared cap.

        If the capped active-set search does not resolve, the *uncapped* value is
        still a valid lower bound for the region, because the capped feasible set
        is a subset of the uncapped one.  We then use it as a bound but refuse to
        take an incumbent from that node, so completeness degrades and soundness
        does not."""
        if cap is None:
            return (
                solve_node(
                    model,
                    allowed,
                    max_iterations=config.max_node_iterations,
                    warm_support=warm,
                ),
                True,
            )
        node = solve_node_capped(
            model, allowed, cap, max_iterations=config.max_node_iterations
        )
        if node.kind == "OPTIMAL" or node.kind == "RAY":
            return node, True
        return (
            solve_node(
                model,
                allowed,
                max_iterations=config.max_node_iterations,
                warm_support=warm,
            ),
            False,
        )

    def finish(**kwargs) -> SaferCertificate:
        base = dict(
            cardinality=m,
            n_assets=N,
            n_periods=model.T,
            weights=None,
            sharpe_lower=math.nan,
            sharpe_upper=math.nan,
            incumbent_sharpe_lower=math.nan,
            q_lower=math.nan,
            q_upper=math.nan,
            support=(),
            nodes_processed=0,
            root_certified=False,
            exact_zero_gap=False,
            ray_support=(),
            elapsed=perf_counter() - started,
        )
        base.update(kwargs)
        base["elapsed"] = perf_counter() - started
        return SaferCertificate(**base)

    def finish_ray(node: NodeSolution, nodes_processed=0) -> SaferCertificate:
        # A support is not a direction. In particular, uniform weights on a null-ray
        # support need not have zero variance. Retain and return the exact ray ratios.
        if len(node.ray_u) != len(node.ray) or not all(x > 0 for x in node.ray_u):
            return finish(status="INVALID", reason="NUMERIC_VERIFICATION")
        return finish(
            status="SPARSE_INFINITE_CERTIFIED",
            reason="NONE",
            weights=_weights_from_support(model, node.ray, node.ray_u),
            support=node.ray,
            ray_support=node.ray,
            sharpe_lower=math.inf,
            sharpe_upper=math.inf,
            nodes_processed=nodes_processed,
            exact={
                "ray": list(node.ray),
                "ray_u": list(node.ray_u),
                "ray_kind": "EXACT_NULL_DIRECTION",
                "target": [list(pair) for pair in model.target],
            },
        )

    # ---- domain dispatch, in the order fixed by the target contract ---------
    all_zero_variance = all(model.col_is_zero(i) for i in range(N))
    max_mean_positive = max(model.s) > 0
    if all_zero_variance and not max_mean_positive:
        return finish(status="UNDEFINED", reason="NO_DEFINED_SHARPE_POINT")
    if not max_mean_positive:
        return finish(status="NON_EPER", reason="ALL_MEANS_NONPOSITIVE")

    # ---- exact sparse infinity pre-check -----------------------------------
    for i in range(N):
        if model.col_is_zero(i) and model.s[i] > 0:
            return finish_ray(NodeSolution(kind="RAY", ray=(i,), ray_u=(1,)))

    # ---- incumbent from positive-mean filtered greedy supports --------------
    q_upper: Fraction | None = None
    best_support: tuple[int, ...] = ()
    best_u: tuple[int, ...] = ()
    best_g = 1
    for support in _greedy_supports(model, m):
        node, feasible = solve(support)
        if not feasible:
            continue
        if node.kind == "RAY" and len(node.ray) <= m:
            return finish_ray(node)
        if node.kind != "OPTIMAL" or node.q is None or node.q >= 0:
            continue
        if len(node.support) <= m and (q_upper is None or node.q < q_upper):
            q_upper, best_support, best_u, best_g = node.q, node.support, node.u, node.g

    # ---- branch and bound over a disjoint support cover --------------------
    # node = (forced, excluded); relaxation drops cardinality on allowed = [N]\excluded
    live: list[tuple[Fraction | None, frozenset, frozenset, tuple, tuple[int, ...]]] = [
        (None, frozenset(), frozenset(), (), ())
    ]
    nodes = 0
    has_floor = any(model.floor)
    open_bounds: list[Fraction] = []
    dense_ray_seen = False
    resource_stop = False
    root_certified = False
    ledger: list[dict] = []

    def emit(
        path, kind, disposition, q: Fraction | None, node: NodeSolution | None
    ) -> None:
        entry = {
            "path": [[int(i), bool(b)] for i, b in path],
            "kind": kind,
            "disposition": disposition,
        }
        if q is not None:
            entry["q"] = [(-q).numerator, (-q).denominator]  # store -q >= 0
        if node is not None and node.kind == "OPTIMAL":
            entry["support"] = list(node.support)
            entry["u"] = list(node.u)
            entry["g"] = node.g
        if node is not None and node.kind == "RAY":
            entry["ray"] = list(node.ray)
            entry["ray_u"] = list(node.ray_u)
        ledger.append(entry)

    while live:
        if nodes >= config.max_nodes or perf_counter() - started > config.max_seconds:
            resource_stop = True
            break
        live.sort(
            key=lambda item: (
                item[0] is not None,
                item[0] if item[0] is not None else 0,
            )
        )
        inherited_bound, forced, excluded, path, warm = live.pop(0)
        allowed = [i for i in range(N) if i not in excluded]
        if len(forced) > m or not allowed:
            emit(path, "VOID", "CLOSED_INFEASIBLE", Fraction(0), None)
            continue
        nodes += 1
        node, cap_feasible = solve(allowed, warm or tuple(sorted(forced)))
        if node.kind == "RAY":
            if len(node.ray) <= m:
                return finish_ray(node, nodes_processed=nodes)
            dense_ray_seen = True
            undecided = [i for i in node.ray if i not in forced and i not in excluded]
            if not undecided:
                emit(path, "RAY", "OPEN_DENSE_RAY", None, node)
                return finish(
                    status="OPEN_NO_FINITE_BOUND",
                    reason="DENSE_RELAXATION_RAY",
                    nodes_processed=nodes,
                    ray_support=tuple(node.ray),
                )
            pivot = undecided[0]
            emit(path, "RAY", "BRANCHED", None, node)
            live.append((None, forced | {pivot}, excluded, path + ((pivot, True),), ()))
            live.append(
                (None, forced, excluded | {pivot}, path + ((pivot, False),), ())
            )
            continue
        if node.kind != "OPTIMAL" or node.q is None:
            emit(path, node.kind, "OPEN_NUMERIC", inherited_bound, node)
            # The popped region is unresolved, not closed. Losing it here would
            # incorrectly replace its unknown lower bound by the incumbent.
            live.append((inherited_bound, forced, excluded, path, warm))
            resource_stop = True
            break
        bound = node.q
        if len(node.support) <= m and cap_feasible:
            # The relaxation optimizer is globally feasible, even if it does not use
            # every forced index. Updating the global incumbent to this lower bound
            # safely closes the region without claiming attainment within the region.
            if q_upper is None or bound < q_upper:
                q_upper, best_support, best_u, best_g = (
                    bound,
                    node.support,
                    node.u,
                    node.g,
                )
            if nodes == 1:
                root_certified = True
            emit(path, "OPTIMAL", "CLOSED_EXACT", bound, node)
            continue
        # The relaxation optimum is not cardinality feasible, so its value is only a
        # bound. Use the tighter of the relaxation and cardinality-aware floor
        # bounds. The verifier recomputes the floor bound from the declared model.
        if has_floor:
            region_floor = cardinality_floor_bound(model, allowed, m)
            if region_floor is not None and region_floor > bound:
                bound = region_floor
        if q_upper is not None and bound >= q_upper:
            emit(path, "OPTIMAL", "PRUNED", node.q, node)
            continue  # safe prune: cannot improve
        undecided = [i for i in node.support if i not in forced and i not in excluded]
        if not undecided:
            open_bounds.append(bound)
            emit(path, "OPTIMAL", "CLOSED_UNRESOLVED", node.q, node)
            continue
        pivot = max(undecided, key=lambda i: node.support.index(i))
        emit(path, "OPTIMAL", "BRANCHED", node.q, node)
        live.append(
            (bound, forced | {pivot}, excluded, path + ((pivot, True),), node.support)
        )
        live.append(
            (bound, forced, excluded | {pivot}, path + ((pivot, False),), node.support)
        )

    if q_upper is None:
        if dense_ray_seen:
            return finish(
                status="OPEN_NO_FINITE_BOUND",
                reason="DENSE_RELAXATION_RAY",
                nodes_processed=nodes,
            )
        return finish(
            status="RESOURCE_OPEN", reason="RESOURCE_LIMIT", nodes_processed=nodes
        )

    weights = _weights_from_support(model, best_support, best_u)
    if q_upper >= 0:  # pragma: no cover - defensive
        return finish(
            status="INVALID", reason="NUMERIC_VERIFICATION", nodes_processed=nodes
        )
    lower = _sqrt_down(-2 * q_upper)  # L  <= S0(what)  <= S*

    recorded_paths = {tuple(tuple(pair) for pair in entry["path"]) for entry in ledger}
    for item in live:
        if item[3] not in recorded_paths:
            emit(item[3], "UNPROCESSED", "LIVE", item[0], None)

    # A live region whose stored key is ``None`` carries no valid lower bound, so no
    # finite global lower bound on q -- and therefore no finite upper bound on S* --
    # may be reported.  Reporting one would be an unsound certificate.
    unbounded_live = any(item[0] is None for item in live)
    if unbounded_live:
        status = "OPEN_NO_FINITE_BOUND" if dense_ray_seen else "RESOURCE_OPEN"
        reason = "DENSE_RELAXATION_RAY" if dense_ray_seen else "RESOURCE_LIMIT"
        return finish(
            status=status,
            reason=reason,
            weights=weights,
            support=best_support,
            sharpe_lower=lower,
            sharpe_upper=math.inf,
            incumbent_sharpe_lower=lower,
            q_lower=-math.inf,
            q_upper=float(q_upper),
            nodes_processed=nodes,
            exact={
                "q_upper": [(-q_upper).numerator, (-q_upper).denominator],
                "support": list(best_support),
                "u": list(best_u),
                "g": best_g,
                "scale_D": model.D,
                "c": model.c,
                "target": [list(pair) for pair in model.target],
                "ledger": ledger,
            },
            diagnostics={
                "exhausted": False,
                "live_nodes": len(live),
                "open_regions": len(open_bounds),
                "unbounded_live": True,
            },
        )

    remaining = [item[0] for item in live if item[0] is not None]
    q_lower = min([q_upper] + remaining + open_bounds)
    exhausted = not live and not open_bounds and not resource_stop
    upper = _sqrt_up(-2 * q_lower)  # U  >= S*
    if exhausted:
        status, reason = "FINITE_CERTIFIED", "NONE"
    elif resource_stop:
        status, reason = "RESOURCE_OPEN", "RESOURCE_LIMIT"
    elif dense_ray_seen:
        status, reason = "OPEN_NO_FINITE_BOUND", "DENSE_RELAXATION_RAY"
    else:  # pragma: no cover - defensive
        status, reason = "RESOURCE_OPEN", "RESOURCE_LIMIT"

    exact = {
        "cap": None if cap is None else str(cap),
        "q_upper": [(-q_upper).numerator, (-q_upper).denominator],
        "q_lower": [(-q_lower).numerator, (-q_lower).denominator],
        "support": list(best_support),
        "u": list(best_u),
        "g": best_g,
        "scale_D": model.D,
        "c": model.c,
        "target": [list(pair) for pair in model.target],
        "ledger": ledger,
    }
    return finish(
        status=status,
        reason=reason,
        weights=weights,
        sharpe_lower=lower,
        sharpe_upper=upper,
        incumbent_sharpe_lower=lower,
        q_lower=float(q_lower),
        q_upper=float(q_upper),
        support=best_support,
        nodes_processed=nodes,
        root_certified=root_certified and status == "FINITE_CERTIFIED",
        exact_zero_gap=(q_lower == q_upper and status == "FINITE_CERTIFIED"),
        exact=exact,
        diagnostics={
            "exhausted": exhausted,
            "live_nodes": len(live),
            "open_regions": len(open_bounds),
            "unbounded_live": False,
        },
    )


# ---------------------------------------------------------------------------
# exact evaluation of a supplied portfolio
# ---------------------------------------------------------------------------
def exact_sharpe_interval(
    model: ExactModel, weights: np.ndarray
) -> tuple[float, float]:
    """Outward binary64 interval for ``S0(w/1'w)`` using integer arithmetic only.

    With ``w = a / 2**k`` exactly and ``A = sum(a)`` the scale cancels:

        S0(w/1'w)^2 = kappa * (s'a)^2 / (D^2 * a'M a) ,

    so the whole evaluation is a dot product, one integer quadratic form and one
    exact rational square root with directed rounding.  ``(nan, nan)`` marks a
    point outside the defined domain.  Because the identity is stated in
    ``(s, D, M, kappa)``, it also applies to explicitly declared rational mean
    and covariance transformations.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != model.N:
        raise ValueError("weight dimension mismatch")
    if not np.isfinite(w).all() or (w < 0).any():
        return math.nan, math.nan
    exponent = 0
    for value in w:
        if value != 0.0:
            exponent = max(exponent, 53 - math.frexp(float(value))[1])
    scale = 1 << max(exponent, 0)
    a = [int(round(float(x) * scale)) for x in w]
    if any(Fraction(ai, scale) != Fraction(float(x)) for ai, x in zip(a, w)):
        return math.nan, math.nan
    total = sum(a)
    if total <= 0:
        return math.nan, math.nan
    numerator = sum(si * ai for si, ai in zip(model.s, a))
    active = [i for i in range(model.N) if a[i]]
    risk_sq_num = 0  # a' M a, by symmetry
    for position, i in enumerate(active):
        ai, row = a[i], model.M[i]
        cross = sum(a[j] * row[j] for j in active[position + 1 :])
        risk_sq_num += ai * (ai * row[i] + 2 * cross)
    if risk_sq_num == 0:
        return (math.inf, math.inf) if numerator > 0 else (math.nan, math.nan)
    ratio_sq = Fraction(
        model.kappa * numerator * numerator, model.D * model.D * risk_sq_num
    )
    if numerator >= 0:
        return _sqrt_down(ratio_sq), _sqrt_up(ratio_sq)
    return -_sqrt_up(ratio_sq), -_sqrt_down(ratio_sq)


__all__ = [
    "PRIMARY_STATUSES",
    "REASONS",
    "SCHEMA",
    "ExactModel",
    "NodeSolution",
    "SaferCertificate",
    "SaferConfig",
    "build_exact_model",
    "exact_sharpe_interval",
    "certify",
    "solve_node",
]
