"""Forward-pass latency benchmark for trained Cascade decoders.

Reports single-shot (unbatched) and amortized (batched) latency per syndrome
round, matching paper Fig 1c / Fig 3b reporting convention. Uses CUDA events
for accurate GPU timing.

Example
-------

    python scripts/25_inference_latency.py \\
        --code surface --distance 7 \\
        --ckpt checkpoints/surface_d7_v2_d7/best.pt

    python scripts/25_inference_latency.py \\
        --code bb --bb-variant 72 \\
        --ckpt checkpoints/bb_72_12_6_v3_bb72/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    p.add_argument("--batches", type=int, nargs="+",
                   default=[1, 8, 64, 512, 4096])
    p.add_argument("--n-iters", type=int, default=200,
                   help="forward passes per batch size (after warmup)")
    p.add_argument("--n-warmup", type=int, default=20)
    p.add_argument("--use-bf16", action="store_true",
                   help="run inference in bf16 (matches training cast)")
    p.add_argument("--out", type=Path, default=None,
                   help="optional JSON dump of timing results")
    return p.parse_args()


def _resolve(args, ckpt):
    cfg = ckpt.get("config") or {}
    h = cfg.get("hidden")
    b = cfg.get("num_blocks", cfg.get("blocks"))
    k = cfg.get("kernel", 3)
    if h is None or b is None:
        raise SystemExit("ckpt missing hidden/blocks")
    return int(h), int(b), int(k)


def _load(args, ckpt, model):
    has_ema = ckpt.get("model_ema") is not None
    if args.prefer == "ema" or (args.prefer == "auto" and has_ema):
        model.load_state_dict(ckpt["model_ema"])
        return "ema"
    model.load_state_dict(ckpt["model"])
    return "live"


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required for latency benchmark")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    hidden, blocks, kernel = _resolve(args, ckpt)

    if args.code == "surface":
        if args.distance is None:
            raise SystemExit("--distance required for surface")
        code = SurfaceCode(distance=args.distance)
        rounds = args.rounds or args.distance
        model, flat_idx, grid_shape = build_model(code, rounds, hidden, blocks, kernel)
    else:
        if args.bb_variant is None:
            raise SystemExit("--bb-variant required for bb")
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

    # Sample one batch of detection events to feed model with realistic input.
    # Use the largest batch size we'll need.
    max_b = max(args.batches)
    sampler = code.make_circuit(p=0.005, rounds=rounds).compile_detector_sampler()
    det, _ = sampler.sample(shots=max_b, separate_observables=True, bit_packed=False)
    det_t = torch.from_numpy(det.astype(np.float32)).to(device)
    grid = detection_events_to_grid(det_t, flat_idx, grid_shape)

    print(f"[latency] code={code.name} R={rounds} ckpt={args.ckpt} weights={src} "
          f"H={hidden} L={blocks} bf16={args.use_bf16}")
    print(f"{'batch':>6}  {'µs/forward':>12}  {'µs/shot':>10}  "
          f"{'µs/cycle (=µs/shot/R)':>22}  {'shots/s':>10}")
    print("-" * 76)

    autocast_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    out = []
    with torch.inference_mode():
        for bs in args.batches:
            x = grid[:bs].contiguous()
            # Warm-up
            for _ in range(args.n_warmup):
                with torch.autocast(device_type="cuda", dtype=autocast_dtype,
                                    enabled=args.use_bf16):
                    _ = model(x)
            torch.cuda.synchronize()

            # Timed loop
            evt_start = torch.cuda.Event(enable_timing=True)
            evt_end = torch.cuda.Event(enable_timing=True)
            evt_start.record()
            for _ in range(args.n_iters):
                with torch.autocast(device_type="cuda", dtype=autocast_dtype,
                                    enabled=args.use_bf16):
                    _ = model(x)
            evt_end.record()
            torch.cuda.synchronize()
            total_ms = evt_start.elapsed_time(evt_end)
            us_per_fwd = total_ms * 1000 / args.n_iters
            us_per_shot = us_per_fwd / bs
            us_per_cycle = us_per_shot / rounds
            shots_per_s = 1e6 / us_per_shot
            print(f"{bs:>6}  {us_per_fwd:>12.1f}  {us_per_shot:>10.2f}  "
                  f"{us_per_cycle:>22.3f}  {shots_per_s:>10.0f}")
            out.append(dict(batch=bs, us_per_forward=us_per_fwd,
                            us_per_shot=us_per_shot, us_per_cycle=us_per_cycle,
                            shots_per_s=shots_per_s))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "code": code.name, "rounds": rounds, "hidden": hidden,
                "blocks": blocks, "ckpt": str(args.ckpt), "weights": src,
                "use_bf16": args.use_bf16,
                "results": out,
            }, f, indent=2)
        print(f"[latency] wrote {args.out}")


if __name__ == "__main__":
    main()
