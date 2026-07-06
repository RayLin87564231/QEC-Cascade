"""Phase 0 diagnostic: train BB-72 with the simplest possible setup.

Goal: rule out optimizer / curriculum / EMA / bf16 as the cause of BB-72
not learning. We use:
  * AdamW (no Muon, no Newton-Schulz)
  * fixed p=0.005 (no curriculum warm-up)
  * no EMA
  * fp32 (no bf16 mixed precision)
  * BBCascadeModel(H=128, L=6) — same as the failed slurm run

If BCE descends below 0.6 in 3000 steps, the failed slurm run was a
hyperparameter / optimizer issue (not architecture). If it stays at
ln(2) ≈ 0.69, architecture or basis is the bottleneck.

Runtime on H100 login GPU: ~5-10 minutes.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from cascade.codes.bb import BBCode
from cascade.data.tensorize import detection_events_to_grid, make_grid_indexer
from cascade.models.cascade_bb import BBCascadeModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p-train", type=float, default=0.005)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    code = BBCode.code_72_12_6()
    layout = code.detector_layout(rounds=args.rounds)
    flat_idx, grid_shape = make_grid_indexer(layout)
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)

    model = BBCascadeModel(
        code=code, layout=layout,
        hidden=args.hidden, num_blocks=args.blocks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] BBCascadeModel  H={args.hidden}  L={args.blocks}  "
          f"params={n_params:,}")
    print(f"[smoke] AdamW lr={args.lr}  wd={args.weight_decay}  "
          f"batch={args.batch}  fp32, no curriculum, no EMA")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    sampler = code.make_circuit(p=args.p_train, rounds=args.rounds).compile_detector_sampler()

    t_start = time.time()
    bce_curve: list[tuple[int, float]] = []
    for step in range(1, args.steps + 1):
        det, obs = sampler.sample(
            shots=args.batch, separate_observables=True, bit_packed=False,
        )
        det_t = torch.from_numpy(det.astype(np.float32)).to(device)
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(device)
        grid = detection_events_to_grid(det_t, flat_idx, grid_shape)

        model.train()
        logits = model(grid)
        loss = F.binary_cross_entropy_with_logits(logits, obs_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            sps = step / (time.time() - t_start)
            bce_curve.append((step, float(loss.item())))
            print(f"  step {step:5d}  loss {loss.item():.4f}  "
                  f"({sps:.1f} steps/s)")

    # Final eval: per-logical logit stats on a fresh batch
    model.eval()
    with torch.inference_mode():
        det, obs = sampler.sample(
            shots=2000, separate_observables=True, bit_packed=False,
        )
        det_t = torch.from_numpy(det.astype(np.float32)).to(device)
        obs_t = torch.from_numpy(obs.astype(np.uint8)).to(device)
        grid = detection_events_to_grid(det_t, flat_idx, grid_shape)
        logits = model(grid)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).to(torch.uint8)
        per_logical_err = (preds != obs_t).float().mean(dim=0)
        block_err = (preds != obs_t).any(dim=1).float().mean()
        eval_bce = F.binary_cross_entropy_with_logits(
            logits.float(), obs_t.float()
        ).item()

    print()
    print(f"[final] eval over 2000 shots:")
    print(f"  block error rate (any logical wrong): {block_err.item():.4f}")
    print(f"  BCE: {eval_bce:.4f}")
    print(f"  per-logical error rate:")
    for i in range(12):
        s = float(logits[:, i].std().item())
        print(f"    L{i:2d}: err={per_logical_err[i].item():.4f}  "
              f"logit std={s:.4f}")
    print()
    print(f"[summary] BCE curve: {bce_curve}")
    if bce_curve:
        first = bce_curve[0][1]
        last = bce_curve[-1][1]
        print(f"  start={first:.4f}  end={last:.4f}  delta={last - first:.4f}")
        if last < 0.60:
            print("  ⇒ OPTIMIZER/CURRICULUM was the cause. Architecture is OK.")
        elif last < 0.66:
            print("  ⇒ partial signal — basis or architecture also limiting.")
        else:
            print(
                "  ⇒ stuck at ln(2). Architecture or basis is bottleneck. "
                "Proceed to Phase 1."
            )


if __name__ == "__main__":
    main()
