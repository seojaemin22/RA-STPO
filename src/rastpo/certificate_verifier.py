"""Optimizer-free independent verifier for a SAFER-SR certificate record.

This module shares no solver code with ``certification``. It rebuilds the
exact integer model, re-derives every bound from the stored proof ledger, and
re-checks the directed rounding without running an optimization solver.

Verdicts
--------
``VERIFIED``            every claim in the record was re-established.
``REJECTED``            at least one claim failed; ``findings`` says which.
``UNSUPPORTED_STATUS``  the record's status carries no re-provable numeric claim
                        (non-certified states); enum and shape checks still run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Sequence

import numpy as np


PRIMARY_STATUSES = frozenset(
    {
        "FINITE_CERTIFIED",
        "SPARSE_INFINITE_CERTIFIED",
        "NON_EPER",
        "UNDEFINED",
        "OPEN_NO_FINITE_BOUND",
        "RESOURCE_OPEN",
        "INVALID",
    }
)
REASONS = frozenset(
    {
        "NONE",
        "NO_DEFINED_SHARPE_POINT",
        "ALL_MEANS_NONPOSITIVE",
        "DENSE_RELAXATION_RAY",
        "RESOURCE_LIMIT",
        "INPUT_CONTRACT",
        "SERIALIZATION",
        "NUMERIC_VERIFICATION",
    }
)
_CERTIFIED = frozenset({"FINITE_CERTIFIED", "SPARSE_INFINITE_CERTIFIED"})
_CLOSED_DISPOSITIONS = frozenset(
    {"CLOSED_EXACT", "PRUNED", "CLOSED_INFEASIBLE", "CLOSED_UNRESOLVED"}
)
_OPEN_DISPOSITIONS = frozenset({"LIVE", "OPEN_DENSE_RAY", "OPEN_NUMERIC"})


@dataclass
class Verdict:
    verdict: str
    findings: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def ok(self) -> bool:
        return self.verdict == "VERIFIED"


# --- independent exact model ------------------------------------------------
def _model(returns: np.ndarray, target=()):
    r = np.asarray(returns, dtype=np.float64)
    T, N = r.shape
    k = 0
    for value in r.reshape(-1):
        if value != 0.0:
            k = max(k, 53 - math.frexp(value)[1])
    k = max(k, 0)
    two = 1 << k
    cols = []
    sums = []
    for i in range(N):
        column = []
        for t in range(T):
            exact = Fraction(float(r[t, i]))
            scaled = exact * two
            if scaled.denominator != 1:
                raise ValueError("non-dyadic entry")
            column.append(int(scaled))
        cols.append(column)
        sums.append(sum(column))
    bm = [tuple(T * cols[i][t] - sums[i] for t in range(T)) for i in range(N)]
    gram = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i, N):
            value = sum(x * y for x, y in zip(bm[i], bm[j]))
            gram[i][j] = value
            gram[j][i] = value
    c, D = T - 1, T * two
    D0 = T * two  # the unshifted scale, needed by COV_EWMA
    kappa = c * D * D
    floor = [0] * N  # Sigma >= Diag(floor)/kappa, entrywise
    # Re-derive the declared target independently from the raw return table.
    if any(kind == "MEAN_EWMA" for kind, _ in target) and target[0][0] != "MEAN_EWMA":
        raise ValueError("MEAN_EWMA must be the first declared map")
    for kind, value in target:
        lam = Fraction(value)
        p_, q_ = lam.numerator, lam.denominator
        if kind == "SHRINK_DIAGONAL":
            if not 0 <= lam <= 1:
                raise ValueError("shrinkage outside [0,1]")
            floor = [(q_ - p_) * floor[i] + p_ * gram[i][i] for i in range(N)]
            gram = [
                [
                    q_ * gram[i][i] if i == j else (q_ - p_) * gram[i][j]
                    for j in range(N)
                ]
                for i in range(N)
            ]
            kappa *= q_
        elif kind == "SHRINK_IDENTITY":
            if not 0 <= lam <= 1:
                raise ValueError("shrinkage outside [0,1]")
            trace = sum(gram[i][i] for i in range(N))
            floor = [(q_ - p_) * N * floor[i] + p_ * trace for i in range(N)]
            gram = [
                [
                    (q_ - p_) * N * gram[i][j] + (p_ * trace if i == j else 0)
                    for j in range(N)
                ]
                for i in range(N)
            ]
            kappa *= q_ * N
        elif kind == "RIDGE":
            if lam < 0:
                raise ValueError("negative ridge")
            floor = [q_ * floor[i] + p_ * kappa for i in range(N)]
            gram = [
                [q_ * gram[i][j] + (p_ * kappa if i == j else 0) for j in range(N)]
                for i in range(N)
            ]
            kappa *= q_
        elif kind == "MEAN_EWMA":
            # weights are a function of (T, halflife) alone; rebuild them here
            h = int(value)
            if h < 1:
                raise ValueError("halflife must be positive")
            weights = [1 << (t // h) for t in range(T)]
            sums = [
                sum(w * cols[i][t] for t, w in enumerate(weights)) for i in range(N)
            ]
            D = sum(weights) * two
        elif kind == "MEAN_WINDOW":
            k = int(value)
            if not 1 <= k <= T:
                raise ValueError("mean window out of range")
            sums = [sum(cols[i][T - k :]) for i in range(N)]
            D = k * two
        elif kind == "COV_EWMA":
            h = int(value)
            if h < 1:
                raise ValueError("covariance half-life must be positive")
            coefficients = [1 << (t // h) for t in range(T)]
            gram = [
                [
                    sum(w * bm[i][t] * bm[j][t] for t, w in enumerate(coefficients))
                    for j in range(N)
                ]
                for i in range(N)
            ]
            floor = [0] * N
            kappa = sum(coefficients) * D0 * D0
        elif kind == "MEAN_OUTER":
            if lam < 0:
                raise ValueError("negative beta")
            square = q_ * D * D
            gram = [
                [square * gram[i][j] + p_ * kappa * sums[i] * sums[j] for j in range(N)]
                for i in range(N)
            ]
            floor = [square * value_ for value_ in floor]
            kappa *= square
        elif kind == "SHRINK_MEAN":
            if not 0 <= lam <= 1:
                raise ValueError("shrinkage outside [0,1]")
            total = sum(sums)
            sums = [(q_ - p_) * N * sums[i] + p_ * total for i in range(N)]
            D *= q_ * N
        else:
            raise ValueError(f"undeclarable target {kind!r}")
    return T, N, D, sums, [tuple(row) for row in gram], kappa, floor


def _sqrt_down(value: Fraction) -> float:
    if value <= 0:
        return 0.0
    x = math.sqrt(float(value))
    while x > 0 and Fraction(x) ** 2 > value:
        x = math.nextafter(x, 0.0)
    while Fraction(math.nextafter(x, math.inf)) ** 2 <= value:
        x = math.nextafter(x, math.inf)
    return x


def _sqrt_up(value: Fraction) -> float:
    if value <= 0:
        return 0.0
    x = math.sqrt(float(value))
    while Fraction(x) ** 2 < value:
        x = math.nextafter(x, math.inf)
    while True:
        d = math.nextafter(x, 0.0)
        if d > 0 and Fraction(d) ** 2 >= value:
            x = d
        else:
            return x


def _q_exact(
    sums, D, gram, kappa, support: Sequence[int], u: Sequence[int], g: int
) -> Fraction:
    """``q(v) = v'Sigma v/2 - mu'v`` at ``v = u/g``, exactly."""
    product = [sum(gram[i][j] * uj for j, uj in zip(support, u)) for i in support]
    quad = sum(uj * pi for uj, pi in zip(u, product))
    lin = sum(sums[i] * ui for i, ui in zip(support, u))
    return Fraction(D * quad - 2 * kappa * g * lin, 2 * kappa * g * g * D)


def _gradient_numerators(
    sums, D, gram, kappa, support, u, g, allowed
) -> dict[int, int]:
    """Sign-preserving numerators of ``(Sigma v - mu)_i`` for ``i`` in ``allowed``."""
    out = {}
    for i in allowed:
        row = gram[i]
        out[i] = D * sum(row[j] * uj for j, uj in zip(support, u)) - kappa * g * sums[i]
    return out


def _floor_bound(sums, D, floor, kappa, allowed, m) -> Fraction | None:
    """Cardinality-aware bound from ``Sigma >= Diag(floor)/kappa``; ``None`` if vacuous.

    Recomputed from scratch: this bound carries no witness because it needs none.
    """
    terms = []
    for i in allowed:
        if sums[i] <= 0:
            continue
        if floor[i] <= 0:
            return None
        terms.append(Fraction(kappa * sums[i] * sums[i], D * D * floor[i]))
    if not terms:
        return Fraction(0)
    terms.sort(reverse=True)
    return -sum(terms[:m]) / 2


def verify(
    returns: np.ndarray, record: dict, cardinality: int | None = None
) -> Verdict:
    findings: list[str] = []
    checks: dict = {}

    status = record.get("status")
    reason = record.get("reason")
    if status not in PRIMARY_STATUSES:
        return Verdict("REJECTED", [f"status not in closed enum: {status!r}"])
    if reason not in REASONS:
        return Verdict("REJECTED", [f"reason not in closed enum: {reason!r}"])
    if status in _CERTIFIED and reason != "NONE":
        findings.append("certified status must carry reason NONE")

    declared = tuple(
        tuple(pair) for pair in (record.get("exact") or {}).get("target", ()) or ()
    )
    try:
        T, N, D, sums, gram, kappa, floor = _model(returns, declared)
    except ValueError as exc:
        return Verdict("REJECTED", [f"declared target is not re-derivable: {exc}"])
    m = int(cardinality if cardinality is not None else record.get("cardinality", 0))
    if not 1 <= m <= N:
        return Verdict("REJECTED", [f"cardinality {m} outside 1..{N}"])
    if record.get("n_assets") not in (None, N) or record.get("n_periods") not in (
        None,
        T,
    ):
        findings.append("declared shape disagrees with the table")

    # --- domain dispatch, re-derived independently --------------------------
    zero_variance = [gram[i][i] == 0 for i in range(N)]
    max_mean_positive = max(sums) > 0
    if all(zero_variance) and not max_mean_positive:
        expected = "UNDEFINED"
    elif not max_mean_positive:
        expected = "NON_EPER"
    else:
        expected = None
    if expected is not None and status != expected:
        findings.append(f"domain dispatch requires {expected}, record says {status}")
    if expected is None and status in {"UNDEFINED", "NON_EPER"}:
        findings.append(f"{status} claimed although the EPER branch applies")
    checks["domain"] = expected or "EPER"

    if status in {"UNDEFINED", "NON_EPER"}:
        return Verdict("VERIFIED" if not findings else "REJECTED", findings, checks)

    # --- returned vector ----------------------------------------------------
    weights_hex = record.get("weights_hex")
    w_exact: list[Fraction] | None = None
    if weights_hex is not None:
        if len(weights_hex) != N:
            findings.append("weight vector length mismatch")
        else:
            w = [float.fromhex(h) for h in weights_hex]
            if any(x < 0 for x in w):
                findings.append("returned vector has a negative coordinate")
            nz = [i for i, x in enumerate(w) if x != 0.0]
            if len(nz) > m:
                findings.append(f"returned cardinality {len(nz)} exceeds budget {m}")
            w_exact = [Fraction(x) for x in w]
            total = sum(w_exact)
            if total <= 0:
                findings.append("returned vector does not sum to a positive value")
            checks["returned_cardinality"] = len(nz)

    # --- infinity branch ----------------------------------------------------
    if status == "SPARSE_INFINITE_CERTIFIED":
        ray = list(record.get("ray_support", []))
        exact = record.get("exact") or {}
        ray_u = exact.get("ray_u", [1] if len(ray) == 1 else [])
        if not ray:
            findings.append("infinity status without a ray witness")
        elif (
            len(ray) > m
            or len(set(ray)) != len(ray)
            or not all(isinstance(i, int) and 0 <= i < N for i in ray)
        ):
            findings.append("invalid infinity-witness support")
        elif len(ray_u) != len(ray) or not all(
            isinstance(u, int) and u > 0 for u in ray_u
        ):
            findings.append("infinity witness needs exact positive ray numerators")
        else:
            # Check the actual direction, not existence of an unspecified vector on
            # its support. No nullspace search (and no T-versus-support-size indexing)
            # is needed by this independent verifier.
            if any(
                sum(gram[i][j] * u for j, u in zip(ray, ray_u)) != 0 for i in range(N)
            ):
                findings.append("stored ray direction does not have zero variance")
            if sum(sums[i] * u for i, u in zip(ray, ray_u)) <= 0:
                findings.append("stored ray direction does not have positive mean")
            if w_exact is not None:
                expected = [Fraction(0)] * N
                for i, u in zip(ray, ray_u):
                    expected[i] = Fraction(u, sum(ray_u))
                tolerance = Fraction(8 * np.finfo(float).eps)
                if any(abs(wi - ei) > tolerance for wi, ei in zip(w_exact, expected)):
                    findings.append(
                        "returned weights do not represent the certified ray"
                    )
        checks["ray_support"] = ray
        return Verdict("VERIFIED" if not findings else "REJECTED", findings, checks)

    exact = record.get("exact") or {}
    ledger = exact.get("ledger")
    if status not in {"FINITE_CERTIFIED", "OPEN_NO_FINITE_BOUND", "RESOURCE_OPEN"}:
        return Verdict("UNSUPPORTED_STATUS", findings, checks)
    if not ledger:
        findings.append("no proof ledger present")
        return Verdict("REJECTED", findings, checks)

    # --- ledger: every leaf carries a re-provable bound ---------------------
    leaves = [e for e in ledger if e.get("disposition") != "BRANCHED"]
    branched = [e for e in ledger if e.get("disposition") == "BRANCHED"]
    if len(leaves) != len(branched) + 1:
        findings.append(
            f"tree is not a full binary decomposition: {len(branched)} internal, "
            f"{len(leaves)} leaves"
        )
    # trie completeness: every internal path must have both children present
    paths = {tuple((int(i), bool(b)) for i, b in e["path"]): e for e in ledger}
    for e in branched:
        base = tuple((int(i), bool(b)) for i, b in e["path"])
        kids = [p for p in paths if len(p) == len(base) + 1 and p[: len(base)] == base]
        if len(kids) != 2 or {p[-1][1] for p in kids} != {True, False}:
            findings.append("a branched region does not have exactly two children")
            break
    if any(tuple() == tuple((int(i), bool(b)) for i, b in e["path"]) for e in ledger):
        pass
    else:
        findings.append("ledger has no root region")

    open_leaves = [e for e in leaves if e.get("disposition") in _OPEN_DISPOSITIONS]
    bad = [
        e
        for e in leaves
        if e.get("disposition") not in (_CLOSED_DISPOSITIONS | _OPEN_DISPOSITIONS)
    ]
    if bad:
        findings.append("unknown leaf disposition present")

    # Every entry that carries a witness -- leaf *or* internal -- is re-proved, because
    # an open leaf inherits its bound from the nearest ancestor that has one.
    verified: dict[tuple, Fraction] = {}
    for e in ledger:
        if "support" not in e or "u" not in e or "g" not in e or e.get("q") is None:
            continue
        path = [(int(i), bool(b)) for i, b in e["path"]]
        excluded = {i for i, b in path if not b}
        allowed = [i for i in range(N) if i not in excluded]
        support = [int(i) for i in e["support"]]
        u = [int(x) for x in e["u"]]
        g = int(e["g"])
        if g <= 0 or any(x <= 0 for x in u):
            findings.append("witness is not strictly positive")
            continue
        if any(i in excluded for i in support):
            findings.append("witness uses an excluded asset")
            continue
        q_stored = -Fraction(e["q"][0], e["q"][1])
        if _q_exact(sums, D, gram, kappa, support, u, g) != q_stored:
            findings.append("stored region value disagrees with the table")
            continue
        grad = _gradient_numerators(sums, D, gram, kappa, support, u, g, allowed)
        if any(grad[i] < 0 for i in allowed):
            findings.append("region witness violates dual feasibility (grad q >= 0)")
            continue
        if any(grad[i] != 0 for i in support):
            findings.append(
                "region witness violates complementarity (grad q = 0 on support)"
            )
            continue
        verified[tuple(path)] = q_stored

    def inherited_bound(path: tuple) -> Fraction | None:
        """Tightest re-proved relaxation value on this region or an ancestor."""
        for cut in range(len(path), -1, -1):
            value = verified.get(path[:cut])
            if value is not None:
                return value
        return None

    q_bounds: list[Fraction] = []
    q_feasible: list[Fraction] = []
    unbounded_leaf = False
    for e in leaves:
        path = tuple((int(i), bool(b)) for i, b in e["path"])
        forced = {i for i, b in path if b}
        excluded = {i for i, b in path if not b}
        allowed = [i for i in range(N) if i not in excluded]
        disposition = e["disposition"]
        if disposition == "CLOSED_INFEASIBLE":
            if len(forced) <= m and allowed:
                findings.append("region declared infeasible is in fact feasible")
            continue
        bound = inherited_bound(path)
        region_floor = _floor_bound(sums, D, floor, kappa, allowed, m)
        if bound is None:
            bound = region_floor  # a finite floor bounds an unbounded relaxation
        elif region_floor is not None and region_floor > bound:
            bound = region_floor
        if bound is None:
            unbounded_leaf = True
            continue
        q_bounds.append(bound)
        if disposition in _OPEN_DISPOSITIONS:
            continue
        if path not in verified:
            findings.append("closed region without a re-proved witness of its own")
            continue
        if len(e["support"]) <= m:
            q_feasible.append(verified[path])

    # The *upper* bound q_upper needs only a feasible point, never an optimal
    # one: any v >= 0 with ||v||_0 <= m has q(v) >= q*, so sqrt(-2 q(v)) <= S*.
    # The declared incumbent is therefore admissible on feasibility alone, which
    # is what makes an open search (RESOURCE_OPEN) still carry a re-proved L.
    declared = record.get("exact") or {}
    if all(k in declared for k in ("support", "u", "g")) and declared["support"]:
        support = [int(i) for i in declared["support"]]
        u = [int(x) for x in declared["u"]]
        g = int(declared["g"])
        if len(support) != len(u) or len(set(support)) != len(support):
            findings.append("declared incumbent is malformed")
        elif not all(0 <= i < N for i in support):
            findings.append("declared incumbent indexes outside the table")
        elif g <= 0 or any(x <= 0 for x in u):
            findings.append(
                "declared incumbent is not strictly positive on its support"
            )
        elif len(support) > m:
            findings.append("declared incumbent violates the cardinality budget")
        else:
            value = _q_exact(sums, D, gram, kappa, support, u, g)
            stored = declared.get("q_upper")
            if stored is not None and value != -Fraction(
                int(stored[0]), int(stored[1])
            ):
                findings.append(
                    "declared incumbent value does not match its own witness"
                )
            q_feasible.append(value)
            checks["incumbent_reproved"] = True

    if not q_feasible:
        findings.append("no cardinality-feasible witness in the ledger")
        return Verdict("REJECTED", findings, checks)

    q_upper = min(q_feasible)
    q_lower = None if unbounded_leaf else min(q_bounds + [q_upper])
    exhausted = not open_leaves
    checks["exhausted"] = exhausted
    checks["leaves"] = len(leaves)
    checks["open_leaves"] = len(open_leaves)

    if status == "FINITE_CERTIFIED" and not exhausted:
        findings.append("FINITE_CERTIFIED claimed with open regions remaining")
    if q_upper >= 0:
        findings.append("incumbent value is not negative; no positive Sharpe certified")
        return Verdict("REJECTED", findings, checks)

    lower = _sqrt_down(-2 * q_upper)
    if not math.isclose(
        lower, float(record.get("sharpe_lower", math.nan)), rel_tol=0, abs_tol=0
    ):
        findings.append(
            "recorded sharpe_lower does not match outward sqrt of the incumbent"
        )
    checks["sharpe_lower"] = lower
    if q_lower is None:
        if float(record.get("sharpe_upper", 0.0)) != math.inf:
            findings.append(
                "no finite global bound available but a finite upper bound is claimed"
            )
        checks["sharpe_upper"] = math.inf
    else:
        upper = _sqrt_up(-2 * q_lower)
        if upper != float(record.get("sharpe_upper", math.nan)):
            findings.append(
                "recorded sharpe_upper does not match outward sqrt of the global bound"
            )
        if upper < lower:
            findings.append("bound direction is inverted")
        checks["sharpe_upper"] = upper
        checks["exact_zero_gap"] = q_lower == q_upper

    # --- the returned vector really attains at least L ----------------------
    if w_exact is not None and sum(w_exact) > 0:
        norm = [x / sum(w_exact) for x in w_exact]
        numerator = sum(Fraction(sums[i]) * norm[i] for i in range(N)) / D
        risk_sq = sum(
            norm[i] * norm[j] * gram[i][j] for i in range(N) for j in range(N)
        ) / Fraction(kappa)
        if risk_sq == 0:
            if numerator <= 0:
                findings.append("returned vector is outside the defined domain")
        elif numerator <= 0:
            findings.append("returned vector has a nonpositive mean")
        else:
            if numerator * numerator < Fraction(lower) ** 2 * risk_sq:
                findings.append(
                    "returned vector does not attain the claimed lower bound"
                )
            checks["returned_sharpe_lower_ok"] = True

    return Verdict("VERIFIED" if not findings else "REJECTED", findings, checks)


__all__ = ["Verdict", "verify", "PRIMARY_STATUSES", "REASONS"]
