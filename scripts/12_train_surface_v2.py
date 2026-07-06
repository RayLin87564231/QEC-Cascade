"""Train Cascade decoder on surface code with Track 2 recipe.

Track 2 = Muon + Lion + 3-stage curriculum + EMA + bf16 mixed precision.

Usage:
    python scripts/12_train_surface_v2.py --distance 5 --steps 40000
    python scripts/12_train_surface_v2.py --distance 7 --steps 60000 --hidden 128 --blocks 8

Designed to be invoked from a slurm batch script for serious runs.
Login-node sanity OK for d=3 with --steps 2000.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cascade.codes.surface import SurfaceCode
from cascade.eval.decoder_compare import evaluate_decoders
from cascade.train.curriculum import CurriculumConfig
from cascade.train.trainer import build_model
from cascade.train.trainer_v2 import TrainConfigV2, train_v2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--distance", type=int, default=5)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--p-train", type=float, default=0.005)
    p.add_argument("--p-warmup", type=float, default=0.001)
    p.add_argument("--p-eval", type=float, nargs="+", default=None)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch", type=int, default=512,
                   help="micro-batch size; effective batch = batch * accum_steps")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="gradient accumulation factor; effective batch = batch * "
                        "accum_steps. Set to reach paper batch ~3328 without OOM "
                        "(e.g. 256 micro x 13 = 3328).")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--muon-lr", type=float, default=3e-3)
    p.add_argument("--lion-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=3e-3)
    p.add_argument("--ema-decay", type=float, default=0.9998)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--final-shots", type=int, default=200000)
    p.add_argument("--out", type=Path,
                   default=Path("/home/leo07010/Ray/QEC/cascade/checkpoints"))
    p.add_argument("--tag", type=str, default="v2")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rounds = args.rounds or args.distance
    p_eval = args.p_eval or [args.p_train]
    code = SurfaceCode(distance=args.distance)
    out_dir = args.out / f"{code.name}_{args.tag}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}")

    cfg = TrainConfigV2(
        distance=args.distance,
        rounds=rounds,
        p_train=args.p_train,
        p_warmup=args.p_warmup,
        batch_size=args.batch,
        accum_steps=args.accum_steps,
        hidden=args.hidden,
        num_blocks=args.blocks,
        muon_lr=args.muon_lr,
        lion_lr=args.lion_lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        use_bf16=not args.no_bf16,
        num_steps=args.steps,
        eval_every=args.eval_every,
        eval_batch=8192,
        seed=args.seed,
        curriculum=CurriculumConfig(p1=args.p_warmup, p2=args.p_train),
    )
    train_v2(code, cfg, device=device, out_dir=out_dir)

    # Final comparison
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model, flat_idx, grid_shape = build_model(
        code, rounds, cfg.hidden, cfg.num_blocks, cfg.kernel
    )
    # Use EMA-applied weights if available; otherwise fall back to live.
    if ckpt.get("model_ema") is not None:
        model.load_state_dict(ckpt["model_ema"])
        print("[main] using EMA weights for final comparison")
    else:
        model.load_state_dict(ckpt["model"])
        print("[main] using LIVE weights (EMA was cold during training)")
    model.to(device).eval()
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)

    print(f"\n[compare] Cascade vs MWPM @ d={args.distance}, rounds={rounds}")
    print(f"{'p':>8} {'decoder':>10} {'fail/total':>16} {'p_block':>10} {'P_L/cycle':>14}  CI95")
    for p in p_eval:
        results = evaluate_decoders(
            code=code, rounds=rounds, p=p, n_shots=args.final_shots,
            model=model, flat_idx=flat_idx, grid_shape=grid_shape,
            device=device, batch=8192, include_mwpm=True,
        )
        for name, r in results.items():
            print(f"{p:>8.4f} {name:>10} {r.n_failures:>7}/{r.n_total:<7} "
                  f"{r.p_block:>10.5f} {r.p_l_per_cycle:>10.4e}  "
                  f"[{r.p_l_lo:.2e}, {r.p_l_hi:.2e}]")


if __name__ == "__main__":
    main()
