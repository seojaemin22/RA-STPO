"""Lifted semidefinite selection problem solved through CVXPY/SCS."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import multiprocessing

import numpy as np

from .data import file_hash, save_json


def sdp_reference(mu_np, S_np, k, eps=1e-06, max_iters=50000, **solver_options):
    """Solve the normalized sparse tangency SDP and return its diagonal, primal matrix, value, and status."""
    import cvxpy as cp
    import numpy as np

    N = S_np.shape[0]
    Sn = S_np / float(np.trace(S_np) / N)
    mn = mu_np / max(float(np.linalg.norm(mu_np)), 1e-30)
    Y = cp.Variable((N, N), symmetric=True)
    z = cp.Variable(nonneg=True)
    cons = [Y >> 0, cp.trace(Y) == z, cp.sum(cp.abs(Y)) <= k * z, cp.trace(Sn @ Y) == 1]
    prob = cp.Problem(cp.Maximize(mn @ Y @ mn), cons)
    prob.solve(
        solver=cp.SCS, eps=eps, max_iters=max_iters, verbose=False, **solver_options
    )
    value = None if Y.value is None else np.asarray(Y.value)
    if (
        value is None
        or value.shape != (N, N)
        or (not np.isfinite(value).all())
        or (prob.value is None)
        or (not np.isfinite(prob.value))
    ):
        raise RuntimeError(
            f"SCS returned status={prob.status!r} without a finite solution"
        )
    return (np.diag(value), value, float(prob.value), prob.status)


def problem_key(signal, covariance, cardinality):
    h = hashlib.sha256()
    for value in [signal, covariance]:
        array = np.ascontiguousarray(value, dtype=np.float64)
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    h.update(str(int(cardinality)).encode())
    return h.hexdigest()


def _solve_worker(connection, args, kwargs):
    try:
        connection.send((True, sdp_reference(*args, **kwargs)))
    except Exception as error:
        connection.send((False, f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def timed_sdp(signal, covariance, cardinality, *, max_seconds=0, **kwargs):
    """Bound a cold solve without accepting an early-terminated solver result."""
    if not max_seconds:
        return sdp_reference(signal, covariance, cardinality, **kwargs)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(
        target=_solve_worker,
        args=(sender, (signal, covariance, cardinality), kwargs),
    )
    worker.start()
    sender.close()
    try:
        if not receiver.poll(max_seconds):
            raise TimeoutError(f"SDP solve exceeded {max_seconds:g} seconds")
        success, value = receiver.recv()
        if not success:
            raise RuntimeError(value)
        return value
    finally:
        receiver.close()
        if worker.is_alive():
            worker.terminate()
        worker.join()


def solve_cached(
    signal,
    covariance,
    cardinality,
    cache,
    *,
    tolerance=1e-4,
    max_iterations=100000,
    solver_options=None,
    max_seconds=0,
):
    """Content-addressed numerical SDP solve with explicit recovery provenance."""
    root = Path(cache)
    root.mkdir(parents=True, exist_ok=True)
    key = problem_key(signal, covariance, cardinality)
    settings = dict(
        tolerance=tolerance,
        max_iterations=max_iterations,
        solver_options={} if solver_options is None else solver_options,
    )
    marker = root / f"{key}.json"
    path = root / f"{key}.npy"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["problem_key"] != key or record["settings"] != settings:
            raise ValueError("incompatible cached SDP settings")
        if file_hash(path) != record["diagonal_sha256"]:
            raise ValueError("corrupt SDP diagonal")
        return np.load(path), record
    attempts = [("default", settings["solver_options"])]
    if not solver_options:
        attempts.extend(
            [
                ("no-acceleration", dict(acceleration_lookback=0)),
                (
                    "indirect-no-acceleration",
                    dict(use_indirect=True, acceleration_lookback=0),
                ),
                (
                    "direct-adaptive-scale",
                    dict(
                        use_indirect=False, acceleration_lookback=0, adaptive_scale=True
                    ),
                ),
            ]
        )
    record = dict(problem_key=key, settings=settings, attempts=[], origin="computed",
                  max_seconds_per_attempt=max_seconds)
    for label, options in attempts:
        started = time.perf_counter()
        try:
            diagonal, _, objective, status = timed_sdp(
                signal,
                covariance,
                cardinality,
                eps=tolerance,
                max_iters=max_iterations,
                max_seconds=max_seconds,
                **options,
            )
            record["attempts"].append(
                dict(
                    setting=label,
                    seconds=time.perf_counter() - started,
                    status=status,
                    objective=objective,
                )
            )
            diagonal = diagonal.astype(np.float32)
            np.save(path, diagonal)
            record["diagonal_sha256"] = file_hash(path)
            save_json(marker, record)
            return diagonal, record
        except Exception as error:
            record["attempts"].append(
                dict(
                    setting=label,
                    seconds=time.perf_counter() - started,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    save_json(root / f"{key}.failed.json", record)
    raise RuntimeError(f"SDP failed without a finite primal solution: {key}")


def ensure_ranking(cache, market, signal, cardinality, **settings):
    """Solve each distinct mean/covariance problem and cache its selection scores."""
    covariance = market.array("covariances")[market.test]
    ranking, seconds, computed = [], 0.0, 0
    for index, (mean, risk) in enumerate(zip(signal, covariance)):
        key = problem_key(mean, risk, cardinality)
        existed = (Path(cache) / f"{key}.json").exists()
        diagonal, record = solve_cached(mean, risk, cardinality, cache, **settings)
        ranking.append(diagonal)
        seconds += sum(item["seconds"] for item in record["attempts"])
        computed += not existed
        if not existed:
            print(f"SD-relaxation {market.name}: {index + 1}/{len(signal)}", flush=True)
    return np.stack(ranking), dict(
        solve_seconds=seconds, newly_solved=computed, problems=len(signal),
        scope="semidefinite solves, including recovery attempts; separate from allocation",
    )


class RepeatedSDP:
    """Reuse one date's conic system while re-optimizing every changed objective."""

    def __init__(self, covariance, signal, cardinality, tolerance=1e-4,
                 max_iterations=100000):
        import cvxpy as cp
        import scs
        from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict

        n = len(signal)
        normalized_covariance = covariance / (np.trace(covariance) / n)
        mean = signal / max(float(np.linalg.norm(signal)), 1e-30)
        matrix = cp.Variable((n, n), symmetric=True)
        scale = cp.Variable(nonneg=True)
        problem = cp.Problem(cp.Maximize(mean @ matrix @ mean), [
            matrix >> 0, cp.trace(matrix) == scale,
            cp.sum(cp.abs(matrix)) <= cardinality * scale,
            cp.trace(normalized_covariance @ matrix) == 1,
        ])
        data, _, _ = problem.get_problem_data(cp.SCS)
        self.indices = np.triu_indices(n)
        self.size = len(self.indices[0])
        self.diagonal = self.indices[0] == self.indices[1]
        self.template = data["c"].copy()
        self.multipliers = np.where(self.diagonal, 1.0, 2.0)
        self.offset = data["param_prob"].var_id_to_col[matrix.id]
        objective = self.objective(signal)
        np.testing.assert_allclose(objective, data["c"], atol=1e-15, rtol=1e-13)
        self.solver = scs.SCS(
            dict(A=data["A"], b=data["b"], c=objective),
            dims_to_solver_dict(data["dims"]),
            eps_abs=tolerance, eps_rel=tolerance, max_iters=max_iterations, verbose=False,
        )

    def objective(self, signal):
        mean = signal / max(float(np.linalg.norm(signal)), 1e-30)
        objective = self.template.copy()
        i, j = self.indices
        objective[self.offset:self.offset + self.size] = (
            -mean[i] * mean[j] * self.multipliers
        )
        return objective

    def solve(self, signal, warm=True):
        self.solver.update(c=self.objective(signal))
        started = time.perf_counter()
        result = self.solver.solve(warm_start=warm)
        info, primal = result["info"], result["x"]
        if primal is None or not np.isfinite(primal).all() or info["status_val"] not in (1, 2):
            raise RuntimeError(f"SCS failed: {info['status']}")
        diagonal = primal[self.offset:self.offset + self.size][self.diagonal]
        return diagonal.astype(np.float32), dict(
            seconds=time.perf_counter() - started, status=info["status"],
            iterations=int(info["iter"]), objective=float(-info["pobj"]), warm_start=warm,
        )


def load_ranking(cache, market, signal, cardinality):
    if cache is None:
        raise ValueError("SDP cache directory is required")
    covariance = market.array("covariances")[market.test]
    values = []
    for mean, S in zip(signal, covariance):
        key = problem_key(mean, S, cardinality)
        path = Path(cache) / f"{key}.npy"
        record = json.loads((Path(cache) / f"{key}.json").read_text())
        if record["problem_key"] != key or file_hash(path) != record["diagonal_sha256"]:
            raise ValueError("ranking input/hash mismatch")
        values.append(np.load(path))
    return np.stack(values)
