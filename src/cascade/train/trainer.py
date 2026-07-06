"""Minimal MVP trainer: AdamW + online Stim sampling + periodic eval.

The Track 2 paper-faithful trainer (Muon + Lion + MuP + EMA + curriculum)
will be added in a sibling module. For now this exists to validate the
pipeline and produce baseline numbers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim
import torch
import torch.nn.functional as F

from cascade.codes.base import Code
from cascade.data.stim_dataset import StimMemoryDataset
from cascade.data.tensorize import detection_events_to_grid, make_grid_indexer
from cascade.models.cascade import CascadeModel


@dataclass
class TrainConfig:
    distance: int
    rounds: int
    p_train: float
    batch_size: int = 512
    hidden: int = 64
    num_blocks: int = 4
    kernel: int = 3
    num_steps: int = 4000
    eval_every: int = 500
    eval_batch: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 100


def build_model(code: Code, rounds: int, hidden: int, num_blocks: int,
                kernel: int) -> tuple[CascadeModel, "torch.Tensor", "torch.Tensor"]:
    layout = code.detector_layout(rounds=rounds)
    flat_idx, grid_shape = make_grid_indexer(layout)
    model = CascadeModel(
        layout=layout,
        num_logicals=code.num_logical_observables,
        hidden=hidden,
        num_blocks=num_blocks,
        kernel=kernel,
    )
    return model, flat_idx, grid_shape


def evaluate(
    model: CascadeModel,
    code: Code,
    rounds: int,
    p: float,
    n_shots: int,
    flat_idx: torch.Tensor,
    grid_shape: torch.Tensor,
    device: torch.device,
    batch: int = 4096,
) -> dict:
    """Sample `n_shots` independent Stim trials, decode with the current model,
    and report block error rate + BCE loss.
    """
    model.eval()
    sampler = code.make_circuit(p=p, rounds=rounds).compile_detector_sampler()
    n_failures = 0
    n_total = 0
    bce_sum = 0.0
    bce_n = 0
    obs_count = code.num_logical_observables
    with torch.inference_mode():
        while n_total < n_shots:
            this_batch = min(batch, n_shots - n_total)
            det, obs = sampler.sample(
                shots=this_batch, separate_observables=True, bit_packed=False
            )
            det_t = torch.from_numpy(det.astype(np.float32)).to(device)
            obs_t = torch.from_numpy(obs.astype(np.float32)).to(device)
            grid = detection_events_to_grid(det_t, flat_idx, grid_shape)
            logits = model(grid)
            preds = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
            obs_uint = obs_t.to(torch.uint8)
            # Block error: any logical observable mispredicted
            err = (preds != obs_uint).any(dim=1)
            n_failures += int(err.sum().item())
            n_total += this_batch
            bce = F.binary_cross_entropy_with_logits(logits, obs_t, reduction="sum")
            bce_sum += float(bce.item())
            bce_n += this_batch * obs_count
    p_block = n_failures / max(n_total, 1)
    avg_bce = bce_sum / max(bce_n, 1)
    return dict(p_block=p_block, n_failures=n_failures, n_total=n_total, bce=avg_bce)


def train(
    code: Code,
    cfg: TrainConfig,
    device: torch.device | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Run a minimal training loop and return final eval metrics."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    model, flat_idx, grid_shape = build_model(
        code, cfg.rounds, cfg.hidden, cfg.num_blocks, cfg.kernel
    )
    model.to(device)
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    ds = StimMemoryDataset(
        code=code, rounds=cfg.rounds, p=cfg.p_train,
        batch_size=cfg.batch_size, seed=cfg.seed,
    )
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    print(f"[train] code={code.name}, params={n_params:,}, "
          f"steps={cfg.num_steps}, batch={cfg.batch_size}, p_train={cfg.p_train}")

    best_p_block = float("inf")
    best_step = -1
    best_state = None
    history: list[dict] = []

    t_start = time.time()
    iter_ds = iter(ds)
    for step in range(1, cfg.num_steps + 1):
        model.train()
        det, obs = next(iter_ds)
        det = det.to(device, non_blocking=True)
        obs = obs.to(device, non_blocking=True)
        grid = detection_events_to_grid(det, flat_idx, grid_shape)
        logits = model(grid)
        loss = F.binary_cross_entropy_with_logits(logits, obs)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        if step % cfg.log_every == 0 or step == 1:
            print(f"  step {step:5d} | loss {loss.item():.4f} "
                  f"| {step / (time.time() - t_start):.1f} steps/s")

        if step % cfg.eval_every == 0 or step == cfg.num_steps:
            metrics = evaluate(
                model, code, cfg.rounds, cfg.p_train,
                n_shots=cfg.eval_batch, flat_idx=flat_idx,
                grid_shape=grid_shape, device=device,
            )
            metrics.update(step=step, lr=cfg.lr)
            history.append(metrics)
            print(f"  [eval] step={step} p_block={metrics['p_block']:.5f} "
                  f"({metrics['n_failures']}/{metrics['n_total']}) bce={metrics['bce']:.4f}")
            if metrics["p_block"] < best_p_block:
                best_p_block = metrics["p_block"]
                best_step = step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t_start
    print(f"[train] done in {elapsed/60:.1f} min | best p_block={best_p_block:.5f} at step {best_step}")

    if out_dir is not None and best_state is not None:
        ckpt_path = out_dir / "best.pt"
        torch.save({
            "model": best_state,
            "config": cfg.__dict__,
            "history": history,
            "best_step": best_step,
            "best_p_block": best_p_block,
        }, ckpt_path)
        print(f"[train] saved best checkpoint → {ckpt_path}")

    return dict(
        best_p_block=best_p_block,
        best_step=best_step,
        history=history,
        elapsed_sec=elapsed,
    )
