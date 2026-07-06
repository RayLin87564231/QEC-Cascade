"""End-to-end smoke test: sample → tensorise → forward → loss → backward.

Runs in seconds on a single GPU. The test passes if:
* Stim sampling produces detection events of the expected shape
* The grid scatter places every detector at a unique cell
* The Cascade model produces finite logits
* Loss is finite and gradients flow through the model
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from cascade.codes.surface import SurfaceCode
from cascade.data.stim_dataset import StimMemoryDataset
from cascade.data.tensorize import (
    detection_events_to_grid,
    make_grid_indexer,
)
from cascade.models.cascade import CascadeModel


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    # 1. Code + circuit + sampler
    code = SurfaceCode(distance=3)
    layout = code.detector_layout(rounds=3)
    print(f"[smoke] surface d=3: grid_shape={layout.grid_shape}, "
          f"detectors={layout.num_detectors}, mask_true={int(layout.grid_mask.sum())}")

    ds = StimMemoryDataset(code=code, rounds=3, p=0.005, batch_size=64, seed=0)
    flat_idx, grid_shape = make_grid_indexer(layout)
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)

    # 2. Pull a single batch
    t0 = time.time()
    it = iter(ds)
    det, obs = next(it)
    sample_dt = time.time() - t0
    print(f"[smoke] sampled batch shapes: det={tuple(det.shape)}, obs={tuple(obs.shape)} "
          f"(took {sample_dt*1000:.1f} ms)")
    assert det.shape == (64, layout.num_detectors), det.shape
    assert obs.shape == (64, code.num_logical_observables), obs.shape

    det = det.to(device)
    obs = obs.to(device)
    grid = detection_events_to_grid(det, flat_idx, grid_shape)
    print(f"[smoke] grid shape: {tuple(grid.shape)}, sum={grid.sum().item():.0f}")
    assert grid.sum().item() == det.sum().item(), "scatter must conserve event count"

    # 3. Model
    model = CascadeModel(
        layout=layout,
        num_logicals=code.num_logical_observables,
        hidden=64,
        num_blocks=4,
        kernel=3,
    ).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[smoke] model params: {nparams:,}")

    # 4. Forward
    t0 = time.time()
    logits = model(grid)
    fwd_dt = time.time() - t0
    print(f"[smoke] logits shape: {tuple(logits.shape)} (forward {fwd_dt*1000:.1f} ms)")
    assert logits.shape == (64, code.num_logical_observables)
    assert torch.isfinite(logits).all(), "logits must be finite"

    # 5. Loss + backward
    loss = F.binary_cross_entropy_with_logits(logits, obs)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"[smoke] loss={loss.item():.4f}, grad norm={grad_norm:.3f}")
    assert torch.isfinite(loss), "loss must be finite"
    assert grad_norm > 0, "gradients must flow"

    print("[smoke] OK")


if __name__ == "__main__":
    main()
