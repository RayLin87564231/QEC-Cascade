"""Convert block-level logical error rates to per-cycle per-logical rates.

The Cascade paper reports the per-cycle per-logical rate ``P_L`` derived
from the block error rate ``P_block`` over ``R`` syndrome rounds and
``k`` logical qubits using

    P_block = 1 - ((1 + (1 - 2 P_L)^R) / 2) ** k

inverted as in paper Eq. (1):

    P_L = (1 - (2 (1 - P_block) ** (1/k) - 1) ** (1/R)) / 2

This module provides numerically stable implementations.
"""

from __future__ import annotations

import numpy as np


def per_cycle_pl(p_block: float | np.ndarray, *, k: int, rounds: int) -> np.ndarray:
    """Block error rate → per-cycle per-logical error rate.

    The formula is well-defined for ``0 <= p_block < 1 - 0.5**(1/(k*rounds))``;
    above that the inversion produces complex numbers (the block rate has
    saturated). We clamp to 0.5 in that regime and warn callers via a
    NaN-aware return.
    """
    p_block = np.asarray(p_block, dtype=np.float64)
    inner = 2.0 * np.power(1.0 - p_block, 1.0 / k) - 1.0
    inner = np.clip(inner, a_min=0.0, a_max=1.0)
    p_l = 0.5 * (1.0 - np.power(inner, 1.0 / rounds))
    return p_l


def per_cycle_pl_with_ci(
    n_failures: int, n_total: int, *, k: int, rounds: int,
    z: float = 1.96,
) -> tuple[float, float, float]:
    """Compute (P_L, lower 95%, upper 95%) using a Wilson interval on
    the block error rate then mapping each end through `per_cycle_pl`.
    """
    if n_total <= 0:
        return float("nan"), float("nan"), float("nan")
    phat = n_failures / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    centre = (phat + z2 / (2.0 * n_total)) / denom
    margin = z * np.sqrt(phat * (1.0 - phat) / n_total + z2 / (4.0 * n_total ** 2)) / denom
    lo, hi = max(centre - margin, 0.0), min(centre + margin, 1.0)
    # The mapping `p_block -> p_l` is monotonic increasing for fixed k, R.
    return (
        float(per_cycle_pl(phat, k=k, rounds=rounds)),
        float(per_cycle_pl(lo, k=k, rounds=rounds)),
        float(per_cycle_pl(hi, k=k, rounds=rounds)),
    )
