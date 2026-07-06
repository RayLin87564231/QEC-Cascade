"""Per-logical diagnostic for a trained BB checkpoint.

Loads ckpt, runs a sweep of shots at a chosen p, prints for each of the K
logical observables: logit std, mean predicted P(1), per-logical block error,
and a "alive" verdict (logit std > 0.1).

Used to resolve the iter-2 decision: which of the 12 BB-72 logicals are
actually being decoded vs still dormant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cascade.codes.bb import BBCode
from cascade.data.tensorize import detection_events_to_grid, make_grid_indexer
from cascade.models.cascade_bb import BBCascadeModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--bb-variant", choices=["72", "144"], default="72")
    p.add_argument("--p", type=float, default=0.005)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--shots", type=int, default=8192)
    p.add_argument("--prefer", choices=["ema", "live", "auto"], default="ema")
    p.add_argument("--alive-thresh", type=float, default=0.1,
                   help="logit std threshold for marking a logical 'alive'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    hidden = cfg.get("hidden")
    blocks = cfg.get("num_blocks", cfg.get("blocks"))
    if hidden is None or blocks is None:
        raise SystemExit("ckpt missing config['hidden'] or config['blocks']")

    code = BBCode.code_72_12_6() if args.bb_variant == "72" else BBCode.code_144_12_12()
    rounds = args.rounds or (6 if args.bb_variant == "72" else 12)
    layout = code.detector_layout(rounds=rounds)
    flat_idx, grid_shape = make_grid_indexer(layout)
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)

    model = BBCascadeModel(code=code, layout=layout, hidden=int(hidden),
                           num_blocks=int(blocks)).to(device).eval()
    has_ema = ckpt.get("model_ema") is not None
    if args.prefer == "ema" or (args.prefer == "auto" and has_ema):
        model.load_state_dict(ckpt["model_ema"])
        src = "ema"
    else:
        model.load_state_dict(ckpt["model"])
        src = "live"

    # lz row weights (which logicals are heavy)
    lz = code.lz_matrix() if hasattr(code, "lz_matrix") else None
    if lz is None:
        lz_rows = code.logical_z if hasattr(code, "logical_z") else None
    else:
        lz_rows = lz
    if lz_rows is not None:
        lz_arr = np.asarray(lz_rows)
        weights = lz_arr.sum(axis=1).astype(int).tolist()
    else:
        weights = None

    sampler = code.make_circuit(p=args.p, rounds=rounds).compile_detector_sampler()
    K = code.num_logical_observables
    all_logits = []
    all_obs = []
    n = 0
    batch = 512
    with torch.inference_mode():
        while n < args.shots:
            this = min(batch, args.shots - n)
            det, obs = sampler.sample(shots=this, separate_observables=True, bit_packed=False)
            det_t = torch.from_numpy(det.astype(np.float32)).to(device)
            grid = detection_events_to_grid(det_t, flat_idx, grid_shape)
            logits = model(grid).float().cpu().numpy()
            all_logits.append(logits)
            all_obs.append(obs.astype(np.uint8))
            n += this

    logits = np.concatenate(all_logits, axis=0)  # (N, K)
    obs = np.concatenate(all_obs, axis=0)        # (N, K)
    preds = (logits > 0).astype(np.uint8)
    err = (preds != obs).astype(np.float32)

    block_err = err.any(axis=1).mean()
    print(f"ckpt={args.ckpt}  weights={src}  p={args.p}  shots={n}  "
          f"hidden={hidden} blocks={blocks}")
    print(f"block error rate = {block_err:.4f}  ({int(err.any(axis=1).sum())}/{n})")
    print(f"BCE (mean over K logicals, per shot): "
          f"{F.binary_cross_entropy_with_logits(torch.from_numpy(logits), torch.from_numpy(obs.astype(np.float32))).item():.4f}")
    print()
    print(f"{'k':>3} {'wt':>4} {'logit_std':>10} {'mean_p1':>9} "
          f"{'obs_rate':>9} {'pred_rate':>10} {'per_l_err':>10} alive?")
    print("-" * 78)
    alive_count = 0
    for k in range(K):
        std = float(logits[:, k].std())
        p1 = float((1 / (1 + np.exp(-logits[:, k]))).mean())
        obs_rate = float(obs[:, k].mean())
        pred_rate = float(preds[:, k].mean())
        per_l_err = float(err[:, k].mean())
        wt = weights[k] if weights else None
        alive = std > args.alive_thresh
        if alive:
            alive_count += 1
        print(f"{k:>3} {wt if wt is not None else '?':>4} {std:>10.4f} {p1:>9.4f} "
              f"{obs_rate:>9.4f} {pred_rate:>10.4f} {per_l_err:>10.4f} "
              f"{'YES' if alive else 'no'}")
    print("-" * 78)
    print(f"alive (std > {args.alive_thresh}): {alive_count}/{K} logicals")


if __name__ == "__main__":
    main()
