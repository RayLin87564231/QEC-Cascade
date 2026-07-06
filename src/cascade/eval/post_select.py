"""Post-selection by per-shot decoder confidence.

The Cascade paper measures how the per-cycle :math:`P_L` improves when
shots with low confidence are rejected. Confidence per shot is

    c = min_i |sigmoid(logit_i) - 0.5|

across logical observables ``i``; shots with ``c < threshold`` are dropped.
The reported error rate is the per-shot block error among the surviving
shots, mapped to per-cycle via ``per_cycle_pl``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cascade.eval.pblock_to_pl import per_cycle_pl_with_ci


@dataclass(frozen=True)
class PostSelectPoint:
    threshold: float
    acceptance: float
    n_accepted: int
    n_total: int
    n_failures: int
    p_block: float
    p_l_per_cycle: float
    p_l_lo: float
    p_l_hi: float


def confidence(probs: np.ndarray) -> np.ndarray:
    """Per-shot decoder confidence: min over observables of |p - 0.5|.

    Args:
        probs: ``(N, num_obs)`` post-sigmoid probabilities.
    Returns:
        ``(N,)`` confidence in ``[0, 0.5]``.
    """
    return np.min(np.abs(probs - 0.5), axis=1)


def sweep_thresholds(
    probs: np.ndarray,
    obs: np.ndarray,
    *,
    rounds: int,
    k: int,
    thresholds: np.ndarray | None = None,
) -> list[PostSelectPoint]:
    """Compute (acceptance, P_L) at each threshold.

    Args:
        probs: ``(N, num_obs)`` sigmoid outputs.
        obs: ``(N, num_obs)`` ground-truth logical flips, 0/1.
        rounds: R used to map block to per-cycle P_L.
        k: number of logical qubits.
        thresholds: confidences to sweep. Defaults to 21 points in
            ``[0, 0.5)``.
    """
    obs = obs.astype(np.uint8)
    pred = (probs > 0.5).astype(np.uint8)
    err = (pred != obs).any(axis=1)
    conf = confidence(probs)
    n_total = probs.shape[0]
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.5, 21, endpoint=True)[:-1]

    out: list[PostSelectPoint] = []
    for t in thresholds:
        accept = conf >= t
        n_acc = int(accept.sum())
        n_fail = int(err[accept].sum()) if n_acc > 0 else 0
        p_block = n_fail / max(n_acc, 1)
        p_l, lo, hi = per_cycle_pl_with_ci(n_fail, n_acc, k=k, rounds=rounds)
        out.append(PostSelectPoint(
            threshold=float(t),
            acceptance=n_acc / max(n_total, 1),
            n_accepted=n_acc,
            n_total=n_total,
            n_failures=n_fail,
            p_block=p_block,
            p_l_per_cycle=p_l,
            p_l_lo=lo,
            p_l_hi=hi,
        ))
    return out
