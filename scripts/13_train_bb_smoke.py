"""Quick BB code MVP training (smoke test on login node).

Uses the simple AdamW trainer to verify the BB pipeline learns something
above random. For real training on slurm, use 14_train_bb_v2.py with
Track 2 (Muon + Lion + curriculum).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from cascade.codes.bb import BBCode
from cascade.data.stim_dataset import StimMemoryDataset
from cascade.data.tensorize import detection_events_to_grid, make_grid_indexer
from cascade.models.cascade_bb import BBCascadeModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--code", choices=["72", "144"], default="72")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--p-train", type=float, default=0.005)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-shots", type=int, default=4096)
    return p.parse_args()


def build(code: BBCode, rounds: int, hidden: int, blocks: int, device):
    layout = code.detector_layout(rounds=rounds)
    flat_idx, grid_shape = make_grid_indexer(layout)
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)
    model = BBCascadeModel(code=code, layout=layout, hidden=hidden, num_blocks=blocks).to(device)
    return model, layout, flat_idx, grid_shape


def evaluate(model, code, rounds, p, n_shots, flat_idx, grid_shape, device, batch=4096):
    sampler = code.make_circuit(p=p, rounds=rounds).compile_detector_sampler()
    n_fail = n_total = 0
    bce_sum = 0.0
    bce_n = 0
    obs_count = code.num_logical_observables
    model.eval()
    with torch.inference_mode():
        while n_total < n_shots:
            this = min(batch, n_shots - n_total)
            det, obs = sampler.sample(shots=this, separate_observables=True, bit_packed=False)
            det_t = torch.from_numpy(det.astype(np.float32)).to(device)
            obs_t = torch.from_numpy(obs.astype(np.float32)).to(device)
            grid = detection_events_to_grid(det_t, flat_idx, grid_shape)
            logits = model(grid)
            preds = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
            obs_uint = obs_t.to(torch.uint8)
            err = (preds != obs_uint).any(dim=1)
            n_fail += int(err.sum().item())
            n_total += this
            bce_sum += float(F.binary_cross_entropy_with_logits(logits, obs_t, reduction="sum").item())
            bce_n += this * obs_count
    return dict(p_block=n_fail / max(n_total, 1), n_failures=n_fail, n_total=n_total,
                bce=bce_sum / max(bce_n, 1))


def main() -> None:
    args = parse_args()
    code = BBCode.code_72_12_6() if args.code == "72" else BBCode.code_144_12_12()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}, code={code.name}")

    model, layout, flat_idx, grid_shape = build(code, args.rounds, args.hidden, args.blocks, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[main] params={n_params:,}, grid_shape={layout.grid_shape}")

    ds = StimMemoryDataset(code=code, rounds=args.rounds, p=args.p_train,
                            batch_size=args.batch, seed=0)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    t_start = time.time()
    iter_ds = iter(ds)
    for step in range(1, args.steps + 1):
        model.train()
        det, obs = next(iter_ds)
        det = det.to(device, non_blocking=True)
        obs = obs.to(device, non_blocking=True)
        grid = detection_events_to_grid(det, flat_idx, grid_shape)
        logits = model(grid)
        loss = F.binary_cross_entropy_with_logits(logits, obs)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 100 == 0 or step == 1:
            print(f"  step {step:5d} | loss {loss.item():.4f} | "
                  f"{step / (time.time() - t_start):.1f} steps/s")

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, code, args.rounds, args.p_train,
                               args.eval_shots, flat_idx, grid_shape, device)
            print(f"  [eval] step={step} p_block={metrics['p_block']:.4f} "
                  f"({metrics['n_failures']}/{metrics['n_total']}) bce={metrics['bce']:.4f}")


if __name__ == "__main__":
    main()
