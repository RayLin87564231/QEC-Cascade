r"""Surface-code error suppression factor :math:`\Lambda` fitter.

The Cascade paper reports :math:`\Lambda` from log-linear regression of
per-cycle :math:`P_L` against the half-distance :math:`\lfloor (d+1)/2 \rfloor`:

    log P_L(d) = a - floor((d+1)/2) * log Lambda

Fit is on at least three points (e.g. d=5,7,9). Confidence intervals come
from a parametric bootstrap that resamples each (n_failures, n_total) point
from a Beta posterior on the block error rate before mapping to per-cycle
P_L and refitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cascade.eval.pblock_to_pl import per_cycle_pl


@dataclass(frozen=True)
class LambdaFit:
    Lambda: float
    Lambda_lo: float
    Lambda_hi: float
    intercept: float
    distances: tuple[int, ...]
    p_l_per_cycle: tuple[float, ...]


def _half_distance(d: int) -> int:
    return (d + 1) // 2


def _fit_one(
    distances: np.ndarray, p_l: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float]:
    """Return (log_Lambda, intercept) from log-linear fit. Filters non-positive p_l.

    If ``weights`` is given (per-distance, e.g. n_failures — the delta-method
    inverse-variance of log P_L), performs a weighted least-squares fit so that
    statistics-starved high-d points do not dominate the slope.
    """
    mask = p_l > 0
    if mask.sum() < 2:
        return float("nan"), float("nan")
    x = np.array([_half_distance(int(d)) for d in distances[mask]], dtype=np.float64)
    y = np.log(p_l[mask])
    A = np.column_stack([np.ones_like(x), x])
    if weights is not None:
        w = np.sqrt(np.maximum(np.asarray(weights, dtype=np.float64)[mask], 1e-9))
        A = A * w[:, None]
        y = y * w
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = coef
    log_lambda = -slope
    return log_lambda, intercept


def fit_lambda(
    distances: list[int],
    n_failures: list[int],
    n_total: list[int],
    rounds_per_d: list[int],
    *,
    k: int = 1,
    n_bootstrap: int = 2000,
    seed: int = 0,
    weighted: bool = False,
) -> LambdaFit:
    """Fit Lambda over multiple distances at a fixed physical error rate.

    Args:
        distances: ordered list, e.g. [5, 7, 9].
        n_failures, n_total: per-distance block-level statistics at the
            same physical error rate p.
        rounds_per_d: rounds R used for each distance (typically R=d).
        k: number of logical qubits (1 for single-logical surface).
        n_bootstrap: bootstrap samples for CI.
        seed: PRNG seed.
        weighted: if True, weight the log-linear fit by n_failures (the
            delta-method inverse-variance of log P_L), so under-sampled
            high-d points do not dominate the slope.
    """
    distances_arr = np.asarray(distances, dtype=np.int64)
    n_fail_arr = np.asarray(n_failures, dtype=np.int64)
    n_tot_arr = np.asarray(n_total, dtype=np.int64)
    rounds_arr = np.asarray(rounds_per_d, dtype=np.int64)

    if not (len(distances) == len(n_fail_arr) == len(n_tot_arr) == len(rounds_arr)):
        raise ValueError("length mismatch in fit_lambda inputs")

    p_block_obs = n_fail_arr / np.maximum(n_tot_arr, 1)
    p_l_obs = np.array([
        per_cycle_pl(p, k=k, rounds=int(R))
        for p, R in zip(p_block_obs, rounds_arr)
    ])
    weights = n_fail_arr.astype(np.float64) if weighted else None
    log_lambda_obs, intercept_obs = _fit_one(distances_arr, p_l_obs, weights)

    rng = np.random.default_rng(seed)
    log_lambdas = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        # Beta(1+f, 1+n-f) posterior with Jeffreys-ish prior
        p_block_b = rng.beta(1.0 + n_fail_arr, 1.0 + (n_tot_arr - n_fail_arr))
        p_l_b = np.array([
            per_cycle_pl(p, k=k, rounds=int(R))
            for p, R in zip(p_block_b, rounds_arr)
        ])
        log_lambda_b, _ = _fit_one(distances_arr, p_l_b, weights)
        log_lambdas[b] = log_lambda_b

    log_lambdas = log_lambdas[np.isfinite(log_lambdas)]
    if log_lambdas.size == 0:
        return LambdaFit(
            Lambda=float("nan"),
            Lambda_lo=float("nan"),
            Lambda_hi=float("nan"),
            intercept=float(intercept_obs),
            distances=tuple(int(d) for d in distances_arr),
            p_l_per_cycle=tuple(float(x) for x in p_l_obs),
        )
    lo, hi = np.quantile(log_lambdas, [0.025, 0.975])
    return LambdaFit(
        Lambda=float(np.exp(log_lambda_obs)),
        Lambda_lo=float(np.exp(lo)),
        Lambda_hi=float(np.exp(hi)),
        intercept=float(intercept_obs),
        distances=tuple(int(d) for d in distances_arr),
        p_l_per_cycle=tuple(float(x) for x in p_l_obs),
    )
