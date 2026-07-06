"""Evaluate Cascade vs MWPM at fixed (code, p, rounds) and compare."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cascade.codes.base import Code
from cascade.data.tensorize import detection_events_to_grid
from cascade.decoders.bposd import BPOSDDecoder
from cascade.decoders.mwpm import MWPMDecoder
from cascade.eval.pblock_to_pl import per_cycle_pl_with_ci
from cascade.models.cascade import CascadeModel


@dataclass
class DecoderResult:
    name: str
    n_failures: int
    n_total: int
    p_block: float
    p_l_per_cycle: float
    p_l_lo: float
    p_l_hi: float


def _wrap_result(name: str, n_fail: int, n_total: int, k: int, R: int) -> DecoderResult:
    p_block = n_fail / max(n_total, 1)
    p, lo, hi = per_cycle_pl_with_ci(n_fail, n_total, k=k, rounds=R)
    return DecoderResult(name, n_fail, n_total, p_block, p, lo, hi)


def evaluate_decoders(
    code: Code,
    rounds: int,
    p: float,
    n_shots: int,
    model: CascadeModel,
    flat_idx: torch.Tensor,
    grid_shape: torch.Tensor,
    device: torch.device,
    batch: int = 4096,
    include_mwpm: bool = True,
    include_bposd: bool = False,
    bposd_max_shots: int | None = None,
    target_failures: int | None = None,
) -> dict[str, DecoderResult]:
    """Sample once, decode the same shots with each requested decoder.

    BP+OSD is optional and slow (CPU, per-shot loop). When ``include_bposd``
    is True and ``bposd_max_shots`` is set, BP+OSD only consumes the first
    ``bposd_max_shots`` shots and the result is reported with that smaller
    sample size; Cascade and MWPM still see all ``n_shots``.

    ``target_failures`` enables a self-sizing (target-failures) mode for clean
    deep-sub-threshold Λ statistics: sampling stops as soon as **Cascade**
    reaches ``target_failures`` failures, or when the ``n_shots`` cap is hit,
    whichever comes first. Cascade is the gate because it is the rarer-failing
    decoder — if it clears the target, MWPM (which fails more often on the same
    shots) has already cleared it too. When ``n_shots`` is exhausted first, the
    point is honestly under-sampled (a measurement floor), which the caller
    flags via ``sufficient``.
    """
    circuit = code.make_circuit(p=p, rounds=rounds)
    sampler = circuit.compile_detector_sampler()
    mwpm = MWPMDecoder(circuit) if include_mwpm else None
    bposd = BPOSDDecoder(circuit) if include_bposd else None

    casc_fail = 0
    mwpm_fail = 0
    bposd_fail = 0
    n_total = 0
    n_bposd_total = 0
    obs_count = code.num_logical_observables

    model.eval()
    with torch.inference_mode():
        while n_total < n_shots and (
            target_failures is None or casc_fail < target_failures
        ):
            this_batch = min(batch, n_shots - n_total)
            det, obs = sampler.sample(
                shots=this_batch, separate_observables=True, bit_packed=False
            )
            det_t = torch.from_numpy(det.astype(np.float32)).to(device)
            grid = detection_events_to_grid(det_t, flat_idx, grid_shape)
            logits = model(grid)
            preds_casc = (torch.sigmoid(logits) > 0.5).to(torch.uint8).cpu().numpy()
            cascade_err = (preds_casc != obs.astype(np.uint8)).any(axis=1)
            casc_fail += int(cascade_err.sum())

            if mwpm is not None:
                preds_m = mwpm.decode_batch(det)
                mwpm_err = (preds_m != obs.astype(np.uint8)).any(axis=1)
                mwpm_fail += int(mwpm_err.sum())

            if bposd is not None:
                if bposd_max_shots is None:
                    take = this_batch
                else:
                    take = max(0, min(this_batch, bposd_max_shots - n_bposd_total))
                if take > 0:
                    preds_b = bposd.decode_batch(det[:take])
                    bposd_err = (preds_b != obs[:take].astype(np.uint8)).any(axis=1)
                    bposd_fail += int(bposd_err.sum())
                    n_bposd_total += take

            n_total += this_batch

    out = {
        "Cascade": _wrap_result("Cascade", casc_fail, n_total, obs_count, rounds),
    }
    if mwpm is not None:
        out["MWPM"] = _wrap_result("MWPM", mwpm_fail, n_total, obs_count, rounds)
    if bposd is not None:
        out["BP+OSD"] = _wrap_result("BP+OSD", bposd_fail, n_bposd_total, obs_count, rounds)
    return out
