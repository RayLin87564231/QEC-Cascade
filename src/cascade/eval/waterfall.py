r"""Waterfall-scaling analysis for surface-code logical error rates.

The Cascade paper's central empirical claim is that below threshold the
per-cycle logical error rate falls *faster* than the distance-limited
scaling predicts. For a fixed distance the curve is modelled as two power
laws in the physical error rate ``p``:

    P_L(p) ~= A * p**a   +   B * p**b        with a > b

The steep term (exponent ``a``) dominates at moderate sub-threshold p — the
"waterfall" — while the shallow floor (``b ~= floor((d+1)/2)``, the number of
faults needed to cause an undetectable logical error) takes over only at very
low p. Direct Monte-Carlo typically only reaches the waterfall regime, so the
honest deliverable is:

  * the fitted single power-law slope ``m`` of P_L(p) over the measured
    sub-threshold range (the effective suppression exponent), and
  * a comparison of ``m`` against the distance floor exponent
    ``floor((d+1)/2)`` — ``m`` markedly steeper than the floor is the
    waterfall signature — with the floor drawn as extrapolation below the
    measured range.

A two-component fit is attempted when there are enough points and it is
better-conditioned; otherwise it degrades gracefully to the single slope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def floor_exponent(distance: int) -> int:
    """Distance-limited floor exponent floor((d+1)/2)."""
    return (distance + 1) // 2


@dataclass(frozen=True)
class PowerLawFit:
    slope: float          # effective suppression exponent m in P_L ~ p^m
    slope_lo: float
    slope_hi: float
    log_intercept: float  # log C in log P_L = log C + m log p
    n_points: int


def fit_powerlaw(
    p: np.ndarray,
    p_l: np.ndarray,
    weights: np.ndarray | None = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> PowerLawFit:
    """Log-log least-squares fit P_L ~ C p^m with a bootstrap CI on the slope.

    ``weights`` (per point, e.g. n_failures) down-weights statistics-starved
    points. Non-positive P_L points are dropped.
    """
    p = np.asarray(p, dtype=np.float64)
    p_l = np.asarray(p_l, dtype=np.float64)
    mask = (p_l > 0) & (p > 0)
    x = np.log(p[mask])
    y = np.log(p_l[mask])
    n = int(mask.sum())
    if n < 2:
        return PowerLawFit(float("nan"), float("nan"), float("nan"), float("nan"), n)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)[mask]

    def _slope_intercept(xx, yy, ww):
        A = np.column_stack([np.ones_like(xx), xx])
        if ww is not None:
            sw = np.sqrt(np.maximum(ww, 1e-9))
            A = A * sw[:, None]
            yy = yy * sw
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        return coef[1], coef[0]  # slope, intercept

    slope, intercept = _slope_intercept(x, y, w)

    # Residual bootstrap on the slope for a CI.
    rng = np.random.default_rng(seed)
    fitted = intercept + slope * x
    resid = y - fitted
    slopes = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        yb = fitted + resid[rng.integers(0, n, n)]
        slopes[b], _ = _slope_intercept(x, yb, w)
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return PowerLawFit(float(slope), float(lo), float(hi), float(intercept), n)


def two_component_curve(p: np.ndarray, A: float, a: float, B: float, b: float) -> np.ndarray:
    return A * np.power(p, a) + B * np.power(p, b)


def fit_two_component(
    p: np.ndarray,
    p_l: np.ndarray,
    floor_exp: float,
    sigma: np.ndarray | None = None,
) -> dict | None:
    """Non-linear fit of P_L ~ A p^a + B p^b (b fixed near the floor exponent).

    Returns a params dict, or None if scipy is unavailable, there are too few
    points, or the fit fails to converge — callers fall back to the single
    power law. ``b`` is fixed to ``floor_exp`` (the distance-limited slope) so
    the free steep exponent ``a`` is identifiable from the waterfall region.
    """
    try:
        from scipy.optimize import curve_fit
    except Exception:
        return None
    p = np.asarray(p, dtype=np.float64)
    p_l = np.asarray(p_l, dtype=np.float64)
    m = (p_l > 0) & (p > 0)
    if int(m.sum()) < 4:
        return None
    x, y = p[m], p_l[m]
    ln_y = np.log(y)

    def model_log(pp, A, a, B):
        return np.log(np.maximum(two_component_curve(pp, A, a, B, floor_exp), 1e-300))

    try:
        p0 = [y.max() / x.max() ** floor_exp, floor_exp + 4.0, y.min() / x.min() ** floor_exp]
        popt, pcov = curve_fit(
            model_log, x, ln_y, p0=p0, maxfev=20000,
            bounds=([0, floor_exp, 0], [np.inf, floor_exp + 20, np.inf]),
        )
    except Exception:
        return None
    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [float("nan")] * 3
    return {
        "A": float(popt[0]), "a": float(popt[1]), "B": float(popt[2]),
        "b": float(floor_exp), "a_err": float(perr[1]),
    }
