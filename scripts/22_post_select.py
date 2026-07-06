"""Acceptance vs per-cycle P_L curve via post-selection on confidence.

Loads a checkpoint, samples shots at the chosen physical error rate, runs
the Cascade model to get per-shot probabilities, then sweeps confidence
thresholds via ``cascade.eval.post_select.sweep_thresholds``.

Output: a JSON with the curve and a PDF plot.

Example
-------

    python scripts/22_post_select.py \
        --code surface --distance 5 \
        --ckpt checkpoints/surface_d5_v2_d5/best.pt \
        --p 0.005 --shots 200000 --hidden 128 --blocks 6 \
        --out results/surface_d5_postselect.json \
        --plot figures/surface_d5_postselect.pdf
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cascade.codes.bb import BBCode
from cascade.codes.surface import SurfaceCode
from cascade.data.tensorize import detection_events_to_grid, make_grid_indexer
from cascade.eval.post_select import sweep_thresholds
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
    p.add_argument("--p", type=float, required=True)
    p.add_argument("--shots", type=int, default=200_000)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--blocks", type=int, required=True)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--n-thresholds", type=int, default=41)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plot", type=Path, default=None)
    return p.parse_args()


@torch.inference_mode()
def collect_probs(model, code, rounds, p_phys, n_shots, flat_idx, grid_shape,
                  device, batch):
    sampler = code.make_circuit(p=p_phys, rounds=rounds).compile_detector_sampler()
    probs_chunks: list[np.ndarray] = []
    obs_chunks: list[np.ndarray] = []
    n_total = 0
    while n_total < n_shots:
        this = min(batch, n_shots - n_total)
        det, obs = sampler.sample(shots=this, separate_observables=True, bit_packed=False)
        det_t = torch.from_numpy(det.astype(np.float32)).to(device)
        grid = detection_events_to_grid(det_t, flat_idx, grid_shape)
        logits = model(grid).float()
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_chunks.append(probs)
        obs_chunks.append(obs.astype(np.uint8))
        n_total += this
    return np.concatenate(probs_chunks, axis=0), np.concatenate(obs_chunks, axis=0)


def _load_ckpt(model: torch.nn.Module, ckpt: dict, prefer: str) -> str:
    has_ema = ckpt.get("model_ema") is not None
    if prefer == "ema" or (prefer == "auto" and has_ema):
        if not has_ema:
            raise ValueError("--prefer ema requested but checkpoint has no model_ema")
        model.load_state_dict(ckpt["model_ema"])
        return "ema"
    model.load_state_dict(ckpt["model"])
    return "live"


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.code == "surface":
        if args.distance is None:
            raise ValueError("--distance required")
        code = SurfaceCode(distance=args.distance)
        rounds = args.rounds or args.distance
        model, flat_idx, grid_shape = build_model(
            code, rounds, args.hidden, args.blocks, args.kernel
        )
        k_logical = 1
    else:
        if args.bb_variant is None:
            raise ValueError("--bb-variant required")
        code = (BBCode.code_72_12_6() if args.bb_variant == "72"
                else BBCode.code_144_12_12())
        rounds = args.rounds or (6 if args.bb_variant == "72" else 12)
        layout = code.detector_layout(rounds=rounds)
        flat_idx, grid_shape = make_grid_indexer(layout)
        model = BBCascadeModel(code=code, layout=layout,
                               hidden=args.hidden, num_blocks=args.blocks)
        k_logical = code.num_logical_observables

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    src = _load_ckpt(model, ckpt, args.prefer)
    model.to(device).eval()
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)

    print(f"[postselect] {code.name} rounds={rounds} p={args.p} shots={args.shots} weights={src}")
    probs, obs = collect_probs(model, code, rounds, args.p, args.shots,
                               flat_idx, grid_shape, device, args.batch)

    thresholds = np.linspace(0.0, 0.5, args.n_thresholds, endpoint=False)
    points = sweep_thresholds(probs, obs, rounds=rounds, k=k_logical,
                              thresholds=thresholds)
    payload = {
        "code": code.name,
        "rounds": rounds,
        "p": args.p,
        "shots": args.shots,
        "weights": src,
        "k": k_logical,
        "points": [asdict(pt) for pt in points],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[postselect] wrote {args.out}")

    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        accept = np.array([pt.acceptance for pt in points])
        p_l = np.array([pt.p_l_per_cycle for pt in points])
        lo = np.array([pt.p_l_lo for pt in points])
        hi = np.array([pt.p_l_hi for pt in points])
        valid = accept > 0
        fig, ax = plt.subplots(figsize=(6, 4.5))
        yerr = np.vstack([np.maximum(p_l[valid] - lo[valid], 0.0),
                          np.maximum(hi[valid] - p_l[valid], 0.0)])
        ax.errorbar(accept[valid], p_l[valid], yerr=yerr,
                    marker="o", linestyle="-", capsize=2.5, markersize=4,
                    label=f"{code.name} @ p={args.p}")
        ax.set_xlabel("acceptance rate")
        ax.set_ylabel(r"per-cycle $P_L$ on accepted shots")
        ax.set_yscale("log")
        ax.invert_xaxis()  # high acceptance on left → low on right
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot, bbox_inches="tight")
        print(f"[postselect] wrote {args.plot}")


if __name__ == "__main__":
    main()
