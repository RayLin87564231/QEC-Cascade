"""Unit tests for eval/* and the new BPOSDDecoder wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.codes.surface import SurfaceCode
from cascade.decoders.bposd import BPOSDDecoder
from cascade.eval.lambda_fit import fit_lambda
from cascade.eval.pblock_to_pl import per_cycle_pl
from cascade.eval.post_select import confidence, sweep_thresholds


# ---------------------------------------------------------------------------
# Lambda fit
# ---------------------------------------------------------------------------

def test_lambda_recovers_synthetic_truth():
    """Generate fake counts consistent with Lambda=8 and check the fitter."""
    true_lambda = 8.0
    distances = [5, 7, 9]
    n = 4_000_000
    fails = []
    for d in distances:
        h = (d + 1) // 2
        p_l = 1e-3 * true_lambda ** (-h)
        p_block = 2 * d * p_l  # k=1, R=d, small p_l ≈ R*P_L (factor 2 from formula)
        fails.append(int(round(p_block * n)))
    fit = fit_lambda(distances, fails, [n] * 3, distances, n_bootstrap=400)
    # 95% CI should contain truth
    assert fit.Lambda_lo <= true_lambda <= fit.Lambda_hi
    # Point estimate within 30%
    assert 0.7 * true_lambda <= fit.Lambda <= 1.3 * true_lambda


def test_lambda_handles_zero_failures():
    """A distance with zero observed failures must not crash the bootstrap."""
    fit = fit_lambda(
        distances=[5, 7, 9],
        n_failures=[100, 10, 0],
        n_total=[10_000, 10_000, 10_000],
        rounds_per_d=[5, 7, 9],
        n_bootstrap=200,
    )
    # Either we get a finite estimate or NaN (if too many bootstraps zero out).
    # We don't fail outright.
    assert np.isfinite(fit.intercept) or np.isnan(fit.intercept)


# ---------------------------------------------------------------------------
# Post-selection
# ---------------------------------------------------------------------------

def test_confidence_in_range():
    probs = np.array([[0.1, 0.9], [0.5, 0.5], [0.99, 0.01]])
    c = confidence(probs)
    assert c.shape == (3,)
    assert np.all((c >= 0.0) & (c <= 0.5))


def test_sweep_thresholds_monotone_acceptance():
    """Acceptance must be non-increasing as the threshold rises."""
    rng = np.random.default_rng(0)
    n = 5000
    probs = rng.uniform(0, 1, size=(n, 1))
    obs = (probs > 0.5).astype(np.uint8)  # perfect predictor
    pts = sweep_thresholds(probs, obs, rounds=5, k=1,
                           thresholds=np.linspace(0, 0.45, 10))
    accs = np.array([pt.acceptance for pt in pts])
    assert np.all(np.diff(accs) <= 0)
    # With a perfect predictor, P_block must be 0 wherever any shot is accepted.
    for pt in pts:
        if pt.n_accepted > 0:
            assert pt.p_block == 0.0


def test_sweep_thresholds_pl_consistent_with_pblock():
    """Spot-check: P_L matches per_cycle_pl(p_block)."""
    rng = np.random.default_rng(1)
    n = 3000
    probs = rng.uniform(0, 1, size=(n, 1))
    obs = rng.integers(0, 2, size=(n, 1))
    pts = sweep_thresholds(probs, obs, rounds=5, k=1,
                           thresholds=np.array([0.0]))
    pt = pts[0]
    expected = float(per_cycle_pl(pt.p_block, k=1, rounds=5))
    assert abs(pt.p_l_per_cycle - expected) < 1e-12


# ---------------------------------------------------------------------------
# BPOSDDecoder wiring (exercises stimbposd on a tiny surface circuit)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_bposd_runs_on_surface_d3():
    """BP+OSD must produce predictions of the right shape on a small circuit."""
    code = SurfaceCode(distance=3)
    circuit = code.make_circuit(p=0.005, rounds=3)
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(shots=64, separate_observables=True, bit_packed=False)
    decoder = BPOSDDecoder(circuit, max_bp_iters=10, osd_order=0)
    preds = decoder.decode_batch(det)
    assert preds.shape == (64, code.num_logical_observables)
    assert preds.dtype == np.uint8
    # On d=3 at p=0.5%, BP+OSD should beat a random predictor (block error < 50%)
    err = (preds != obs.astype(np.uint8)).any(axis=1)
    assert err.mean() < 0.5
