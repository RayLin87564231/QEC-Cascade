"""Track 2 trainer: Muon + Lion + EMA + 3-stage curriculum + bf16 mixed precision.

The forward/backward are run in bf16 autocast; the optimiser step is
explicitly in fp32 to keep Newton-Schulz stable. EMA tracks the live
weights with decay 0.9998. A 3-stage curriculum drives the dataset's
noise level from a low p1 through linear annealing to the target p2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cascade.codes.base import Code
from cascade.data.stim_dataset import StimMemoryDataset
from cascade.data.tensorize import detection_events_to_grid, make_grid_indexer
from cascade.models.cascade import CascadeModel
from cascade.models.ema import ParamEMA
from cascade.train.curriculum import CurriculumConfig
from cascade.train.lr_schedule import cosine_with_warmup
from cascade.train.optimizers import (
    build_muon_lion, step_muon_lion, zero_grad_muon_lion,
)
from cascade.train.trainer import build_model


@dataclass
class TrainConfigV2:
    distance: int
    rounds: int
    p_train: float                # target p2 of curriculum
    p_warmup: float = 0.001       # p1 of curriculum (low-noise warm-up)
    batch_size: int = 512         # micro-batch; effective batch = batch_size * accum_steps
    accum_steps: int = 1          # gradient accumulation; effective batch = batch_size * accum_steps
    hidden: int = 128
    num_blocks: int = 8
    kernel: int = 3
    num_steps: int = 40000
    eval_every: int = 1000
    eval_batch: int = 8192
    muon_lr: float = 3e-3
    lion_lr: float = 2e-4
    weight_decay: float = 3e-3
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    lr_floor: float = 0.1
    ema_decay: float = 0.9998
    use_bf16: bool = True
    seed: int = 0
    log_every: int = 200
    curriculum: CurriculumConfig | None = None


@torch.inference_mode()
def evaluate_with(
    model: CascadeModel,
    code: Code,
    rounds: int,
    p: float,
    n_shots: int,
    flat_idx: torch.Tensor,
    grid_shape: torch.Tensor,
    device: torch.device,
    batch: int = 8192,
    use_bf16: bool = True,
) -> dict:
    """Evaluate `model` (already configured in eval mode by caller)."""
    sampler = code.make_circuit(p=p, rounds=rounds).compile_detector_sampler()
    n_failures = 0
    n_total = 0
    bce_sum = 0.0
    obs_count = code.num_logical_observables
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    while n_total < n_shots:
        this = min(batch, n_shots - n_total)
        det, obs = sampler.sample(
            shots=this, separate_observables=True, bit_packed=False
        )
        det_t = torch.from_numpy(det.astype(np.float32)).to(device)
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(device)
        grid = detection_events_to_grid(det_t, flat_idx, grid_shape)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
            logits = model(grid)
        logits = logits.float()
        preds = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
        err = (preds != obs_t.to(torch.uint8)).any(dim=1)
        n_failures += int(err.sum().item())
        n_total += this
        bce_sum += float(F.binary_cross_entropy_with_logits(
            logits, obs_t, reduction="sum"
        ).item())
    return dict(
        p_block=n_failures / max(n_total, 1),
        n_failures=n_failures,
        n_total=n_total,
        bce=bce_sum / max(n_total * obs_count, 1),
    )


def train_v2(
    code: Code,
    cfg: TrainConfigV2,
    device: torch.device | None = None,
    out_dir: Path | None = None,
) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    curriculum = cfg.curriculum or CurriculumConfig(p1=cfg.p_warmup, p2=cfg.p_train)
    accum_steps = max(1, int(cfg.accum_steps))
    effective_batch = cfg.batch_size * accum_steps

    model, flat_idx, grid_shape = build_model(
        code, cfg.rounds, cfg.hidden, cfg.num_blocks, cfg.kernel
    )
    model.to(device)
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    ema = ParamEMA(model, decay=cfg.ema_decay)

    ds = StimMemoryDataset(
        code=code, rounds=cfg.rounds, p=curriculum.p_at(0, cfg.num_steps),
        batch_size=cfg.batch_size, seed=cfg.seed,
    )
    opts = build_muon_lion(
        model,
        muon_lr=cfg.muon_lr,
        muon_weight_decay=cfg.weight_decay,
        lion_lr=cfg.lion_lr,
        lion_weight_decay=0.0,
    )

    print(f"[train v2] code={code.name}, params={n_params:,}, "
          f"steps={cfg.num_steps}, micro_batch={cfg.batch_size}, "
          f"accum={accum_steps}, effective_batch={effective_batch}, "
          f"p1={curriculum.p1:.4f}→p2={curriculum.p2:.4f}, "
          f"H={cfg.hidden}, L={cfg.num_blocks}, ema={cfg.ema_decay}")

    history: list[dict] = []
    best_p_block = float("inf")
    best_step = -1

    autocast_dtype = torch.bfloat16 if cfg.use_bf16 else torch.float32
    t_start = time.time()
    iter_ds = iter(ds)
    for step in range(1, cfg.num_steps + 1):
        # Curriculum: update sampler noise.
        p_now = curriculum.p_at(step, cfg.num_steps)
        if abs(p_now - ds.p) > 1e-6:
            ds.set_noise(p_now)
            iter_ds = iter(ds)

        # LR schedule.
        sched_mult = cosine_with_warmup(
            step, total_steps=cfg.num_steps,
            warmup_steps=cfg.warmup_steps, decay_floor=cfg.lr_floor,
        )
        for g in opts["muon"].param_groups:
            g["lr"] = cfg.muon_lr * sched_mult
        for g in opts["lion"].param_groups:
            g["lr"] = cfg.lion_lr * sched_mult

        model.train()
        # Gradient accumulation to reach the paper's effective batch (~3328)
        # without OOM. Mirror scripts/14_train_bb_v2.py:174-190 exactly:
        # curriculum + LR already updated once per *opt* step above; a single
        # zero_grad, accum_steps micro-batches each contributing loss/accum,
        # then one clip -> step -> ema per opt step.
        zero_grad_muon_lion(opts)
        loss_accum = 0.0
        for _ in range(accum_steps):
            det, obs = next(iter_ds)
            det = det.to(device, non_blocking=True)
            obs = obs.to(device, non_blocking=True)
            grid = detection_events_to_grid(det, flat_idx, grid_shape)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=cfg.use_bf16):
                logits = model(grid)
            # Cast to fp32 before scaling so backward stays fp32 — avoids
            # bf16 underflow when accum_steps is large.
            loss = F.binary_cross_entropy_with_logits(logits.float(), obs) / accum_steps
            loss.backward()
            loss_accum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # Optimizer step happens in fp32 by default (parameters are fp32).
        step_muon_lion(opts)
        ema.update(model)

        if step % cfg.log_every == 0 or step == 1:
            print(f"  step {step:6d} | p={p_now:.4f} lr_mult={sched_mult:.3f} "
                  f"loss {loss_accum:.4f} | {step / (time.time() - t_start):.1f} steps/s")

        if step % cfg.eval_every == 0 or step == cfg.num_steps:
            # Always evaluate live model (no EMA delay)
            model.eval()
            live_metrics = evaluate_with(
                model, code, cfg.rounds, cfg.p_train,
                n_shots=cfg.eval_batch, flat_idx=flat_idx,
                grid_shape=grid_shape, device=device,
                use_bf16=cfg.use_bf16,
            )
            # EMA only meaningful after ~5/(1-decay) steps; until then it's
            # mostly random init weights. Skip EMA eval before that.
            ema_warm_steps = int(5.0 / max(1e-8, 1.0 - cfg.ema_decay))
            ema_metrics = None
            if step >= ema_warm_steps:
                with ema.applied(model):
                    model.eval()
                    ema_metrics = evaluate_with(
                        model, code, cfg.rounds, cfg.p_train,
                        n_shots=cfg.eval_batch, flat_idx=flat_idx,
                        grid_shape=grid_shape, device=device,
                        use_bf16=cfg.use_bf16,
                    )

            metrics = {"step": step, "live": live_metrics, "ema": ema_metrics}
            history.append(metrics)
            ema_str = (f"ema p_block={ema_metrics['p_block']:.5f}" if ema_metrics
                       else "ema=cold")
            print(f"  [eval] step={step} live p_block={live_metrics['p_block']:.5f} "
                  f"bce={live_metrics['bce']:.4f}  {ema_str}")
            # Track the best of either model for checkpointing
            best_metric = ema_metrics or live_metrics
            if best_metric["p_block"] < best_p_block:
                best_p_block = best_metric["p_block"]
                best_step = step
                if out_dir is not None:
                    # Convert config (may contain dataclass curriculum) to plain dict
                    cfg_dict = dict(cfg.__dict__)
                    cfg_dict["effective_batch"] = effective_batch
                    if cfg.curriculum is not None:
                        cfg_dict["curriculum"] = cfg.curriculum.__dict__
                    # Save EMA-applied weights as a directly-loadable state_dict
                    if ema_metrics is not None:
                        with ema.applied(model):
                            ema_state_dict = {
                                k: v.detach().clone()
                                for k, v in model.state_dict().items()
                            }
                    else:
                        ema_state_dict = None
                    torch.save({
                        "model": model.state_dict(),
                        "model_ema": ema_state_dict,
                        "ema": ema.state_dict(),
                        "config": cfg_dict,
                        "history": history,
                        "best_step": best_step,
                        "best_p_block": best_p_block,
                    }, out_dir / "best.pt")

    elapsed = time.time() - t_start
    print(f"[train v2] done in {elapsed/60:.1f} min "
          f"| best p_block={best_p_block:.5f} at step {best_step}")
    return dict(
        best_p_block=best_p_block,
        best_step=best_step,
        history=history,
        elapsed_sec=elapsed,
    )
