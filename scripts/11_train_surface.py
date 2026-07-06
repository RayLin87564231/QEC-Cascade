"""Train Cascade decoder on a surface code.

Usage:
    python scripts/11_train_surface.py --distance 3 --steps 2000

Designed for login-node sanity runs (d=3) and slurm-submitted larger
distances (d=5, 7, 9).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cascade.codes.surface import SurfaceCode
from cascade.eval.decoder_compare import evaluate_decoders
from cascade.eval.pblock_to_pl import per_cycle_pl_with_ci
from cascade.train.trainer import TrainConfig, train, build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--distance", type=int, default=3)
    p.add_argument("--rounds", type=int, default=None,
                   help="default = distance")
    p.add_argument("--p-train", type=float, default=0.005)
    p.add_argument("--p-eval", type=float, nargs="+", default=None,
                   help="defaults to [p_train]")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--final-shots", type=int, default=20000,
                   help="shots per p in final Cascade-vs-MWPM eval")
    p.add_argument("--out", type=Path,
                   default=Path("/home/leo07010/Ray/QEC/cascade/checkpoints"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rounds = args.rounds or args.distance
    p_eval = args.p_eval or [args.p_train]
    code = SurfaceCode(distance=args.distance)
    out_dir = args.out / code.name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}")

    cfg = TrainConfig(
        distance=args.distance,
        rounds=rounds,
        p_train=args.p_train,
        batch_size=args.batch,
        hidden=args.hidden,
        num_blocks=args.blocks,
        lr=args.lr,
        num_steps=args.steps,
        eval_every=args.eval_every,
        eval_batch=4096,
        seed=args.seed,
    )
    result = train(code, cfg, device=device, out_dir=out_dir)

    # Reload best checkpoint into a fresh model for comparison
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model, flat_idx, grid_shape = build_model(
        code, rounds, cfg.hidden, cfg.num_blocks, cfg.kernel
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)

    print(f"\n[compare] Cascade vs MWPM @ d={args.distance}, rounds={rounds}")
    print(f"{'p':>10}  {'decoder':>10}  {'failures/total':>20}  {'p_block':>10}  "
          f"{'P_L (per cycle)':>20}")
    for p in p_eval:
        results = evaluate_decoders(
            code=code, rounds=rounds, p=p, n_shots=args.final_shots,
            model=model, flat_idx=flat_idx, grid_shape=grid_shape,
            device=device, batch=4096, include_mwpm=True,
        )
        for name, r in results.items():
            print(f"  {p:>10.4f}  {name:>10}  {r.n_failures:>10}/{r.n_total:<8}  "
                  f"{r.p_block:>10.5f}  {r.p_l_per_cycle:>10.4e} "
                  f"[{r.p_l_lo:.2e}, {r.p_l_hi:.2e}]")


if __name__ == "__main__":
    main()
