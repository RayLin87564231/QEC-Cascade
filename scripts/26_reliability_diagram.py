"""Reliability diagram for trained Cascade decoder (paper Fig 5c).

Bins predicted P(observable_i = 1) into N bins and plots actual flip rate
within each bin. Diagonal y = x = perfect calibration.

Example
-------

    python scripts/26_reliability_diagram.py \\
        --code surface --distance 7 \\
        --ckpt checkpoints/surface_d7_v2_d7/best.pt \\
        --hidden 128 --blocks 8 \\
        --p 0.003 0.005 0.007 \\
        --shots 200000 \\
        --out results/reliability_surf_d7.json \\
        --plot figures/demo_reliability_surf_d7.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cascade.codes.bb import BBCode
from cascade.codes.surface import SurfaceCode
from cascade.data.tensorize import detection_events_to_grid, make_grid_indexer
from cascade.models.cascade_bb import BBCascadeModel
from cascade.train.trainer import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--code", choices=["surface", "bb"], required=True)
    p.add_argument("--distance", type=int, default=None)
    p.add_argument("--bb-variant", choices=["72", "144"], default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--prefer", choices=["ema", "live", "auto"], default="auto")
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--blocks", type=int, default=None)
    p.add_argument("--kernel", type=int, default=None)
    p.add_argument("--p", type=float, nargs="+", required=True)
    p.add_argument("--shots", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plot", type=Path, default=None)
    return p.parse_args()


def _resolve(args, ckpt):
    cfg = ckpt.get("config") or {}
    h = args.hidden if args.hidden is not None else cfg.get("hidden")
    b = args.blocks if args.blocks is not None else cfg.get("num_blocks", cfg.get("blocks"))
    k = args.kernel if args.kernel is not None else cfg.get("kernel", 3)
    if h is None or b is None:
        raise SystemExit("Need --hidden/--blocks (not in ckpt config)")
    return int(h), int(b), int(k)


def _load(args, ckpt, model):
    has_ema = ckpt.get("model_ema") is not None
    if args.prefer == "ema" or (args.prefer == "auto" and has_ema):
        model.load_state_dict(ckpt["model_ema"])
        return "ema"
    model.load_state_dict(ckpt["model"])
    return "live"


@torch.inference_mode()
def collect_predictions(model, code, rounds, p, n_shots, flat_idx, grid_shape,
                        device, batch=4096):
    """Sample shots and return concatenated (probs, obs_truth) arrays."""
    sampler = code.make_circuit(p=p, rounds=rounds).compile_detector_sampler()
    probs = []
    truth = []
    n = 0
    while n < n_shots:
        this = min(batch, n_shots - n)
        det, obs = sampler.sample(shots=this, separate_observables=True, bit_packed=False)
        det_t = torch.from_numpy(det.astype(np.float32)).to(device)
        grid = detection_events_to_grid(det_t, flat_idx, grid_shape)
        logits = model(grid).float().cpu()
        probs.append(torch.sigmoid(logits).numpy())
        truth.append(obs.astype(np.uint8))
        n += this
    return np.concatenate(probs, axis=0), np.concatenate(truth, axis=0)


def reliability_bins(probs_flat, truth_flat, n_bins):
    """Bin by predicted P(1). For each bin, return (mean predicted P, actual flip rate, count)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    actuals = []
    counts = []
    for i in range(n_bins):
        mask = (probs_flat >= edges[i]) & (probs_flat < edges[i + 1])
        if i == n_bins - 1:
            mask |= (probs_flat == 1.0)
        c = int(mask.sum())
        if c == 0:
            continue
        centers.append(float(probs_flat[mask].mean()))
        actuals.append(float(truth_flat[mask].mean()))
        counts.append(c)
    return np.array(centers), np.array(actuals), np.array(counts)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    hidden, blocks, kernel = _resolve(args, ckpt)

    if args.code == "surface":
        if args.distance is None:
            raise SystemExit("--distance required")
        code = SurfaceCode(distance=args.distance)
        rounds = args.rounds or args.distance
        model, flat_idx, grid_shape = build_model(code, rounds, hidden, blocks, kernel)
    else:
        if args.bb_variant is None:
            raise SystemExit("--bb-variant required")
        code = (BBCode.code_72_12_6() if args.bb_variant == "72"
                else BBCode.code_144_12_12())
        rounds = args.rounds or (6 if args.bb_variant == "72" else 12)
        layout = code.detector_layout(rounds=rounds)
        flat_idx, grid_shape = make_grid_indexer(layout)
        model = BBCascadeModel(code=code, layout=layout, hidden=hidden, num_blocks=blocks)

    src = _load(args, ckpt, model)
    model.to(device).eval()
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)

    print(f"[reliability] code={code.name} R={rounds} weights={src} "
          f"H={hidden} L={blocks} bins={args.n_bins}")

    series = []
    for p_phys in args.p:
        probs, truth = collect_predictions(model, code, rounds, p_phys, args.shots,
                                            flat_idx, grid_shape, device,
                                            batch=args.batch)
        # Flatten across logical axis (treat each logical-shot as one event)
        probs_flat = probs.reshape(-1)
        truth_flat = truth.reshape(-1).astype(np.float32)
        centers, actuals, counts = reliability_bins(probs_flat, truth_flat, args.n_bins)
        # Expected calibration error (ECE)
        weights = counts / counts.sum()
        ece = float(np.sum(weights * np.abs(centers - actuals)))
        print(f"  p={p_phys:.4f}  shots={args.shots}  bins_used={len(centers)}  "
              f"ECE={ece:.4f}")
        series.append(dict(
            p=float(p_phys),
            shots=int(args.shots),
            bin_centers=centers.tolist(),
            actual_rates=actuals.tolist(),
            counts=counts.tolist(),
            ece=ece,
        ))

    payload = dict(code=code.name, rounds=rounds, hidden=hidden, blocks=blocks,
                   ckpt=str(args.ckpt), weights=src, n_bins=args.n_bins,
                   series=series)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[reliability] wrote {args.out}")

    if args.plot is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        cmap = plt.get_cmap("viridis")
        for i, s in enumerate(series):
            color = cmap(i / max(1, len(series) - 1))
            ax.plot(s["bin_centers"], s["actual_rates"], "o-",
                    color=color, label=f"p={s['p']:.4f} (ECE={s['ece']:.3f})")
        ax.set_xlabel("predicted flip probability")
        ax.set_ylabel("actual flip rate")
        ax.set_title(f"Reliability — {code.name}, R={rounds}, H={hidden}, L={blocks}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"[reliability] wrote {args.plot}")


if __name__ == "__main__":
    main()
