"""Evaluate a trained Cascade checkpoint over a noise sweep.

Reused by 21_compare_baselines and 22_post_select. Output is JSON keyed by
(decoder, p), preserving Wilson 95% CI on the per-cycle P_L.

Examples
--------

    # Surface code d=5 v2 checkpoint, with MWPM baseline
    python scripts/20_eval_decoder.py \
        --code surface --distance 5 \
        --ckpt checkpoints/surface_d5_v2_d5/best.pt \
        --p 0.001 0.002 0.003 0.004 0.005 0.006 0.007 \
        --shots 200000 --hidden 128 --blocks 6 \
        --out results/surface_d5_v2.json

    # BB-72 with BP+OSD baseline (slow; bposd-shots caps the slow decoder)
    python scripts/20_eval_decoder.py \
        --code bb --bb-variant 72 \
        --ckpt checkpoints/bb_72_12_6_v2_bb72/best.pt \
        --p 0.002 0.003 0.004 0.005 \
        --shots 200000 --bposd-shots 5000 --hidden 128 --blocks 6 \
        --out results/bb72_v2.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from cascade.codes.bb import BBCode
from cascade.codes.surface import SurfaceCode
from cascade.data.tensorize import make_grid_indexer
from cascade.eval.decoder_compare import evaluate_decoders
from cascade.models.cascade_bb import BBCascadeModel
from cascade.train.trainer import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--code", choices=["surface", "bb"], required=True)
    p.add_argument("--distance", type=int, default=None,
                   help="surface code distance (required when --code surface)")
    p.add_argument("--bb-variant", choices=["72", "144"], default=None,
                   help="BB code variant (required when --code bb)")
    p.add_argument("--rounds", type=int, default=None,
                   help="defaults to distance for surface, 6 for bb-72, 12 for bb-144")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--prefer", choices=["ema", "live", "auto"], default="auto",
                   help="which weights to load; auto = ema if present else live")
    p.add_argument("--p", type=float, nargs="+", required=True,
                   help="physical error rates to evaluate")
    p.add_argument("--shots", type=int, default=200_000,
                   help="shots per p for Cascade and MWPM (a hard cap in "
                        "target-failures mode)")
    p.add_argument("--target-failures", type=int, default=None,
                   help="self-sizing MC: sample until Cascade reaches this many "
                        "failures or --shots is hit. Use for deep-sub-threshold "
                        "Lambda stats (e.g. 100). Sets --min-errors if unset.")
    p.add_argument("--hidden", type=int, default=None,
                   help="model hidden width; defaults to ckpt['config']['hidden']")
    p.add_argument("--blocks", type=int, default=None,
                   help="number of bottleneck blocks; defaults to ckpt['config']['num_blocks']")
    p.add_argument("--kernel", type=int, default=None,
                   help="surface code only; defaults to ckpt['config']['kernel'] or 3")
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--no-mwpm", action="store_true")
    p.add_argument("--bposd", action="store_true",
                   help="add BP+OSD baseline (CPU, slow)")
    p.add_argument("--bposd-shots", type=int, default=None,
                   help="cap BP+OSD shots per p (defaults to --shots)")
    p.add_argument("--min-errors", type=int, default=0,
                   help="mark a (decoder, p) point as sufficient only if "
                        "n_failures >= this. 0 disables the flag (point always "
                        "sufficient). Recommended: 100 for downstream Lambda fits.")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def _resolve_model_hparams(args: argparse.Namespace, ckpt: dict) -> tuple[int, int, int]:
    """Resolve hidden / blocks / kernel from CLI args, falling back to ckpt config.

    The trainer stores num_blocks under either key depending on era: surface
    runs use ``num_blocks``, BB runs use ``blocks``. Accept both.
    """
    cfg = ckpt.get("config") or {}
    hidden = args.hidden if args.hidden is not None else cfg.get("hidden")
    blocks = args.blocks if args.blocks is not None else cfg.get("num_blocks", cfg.get("blocks"))
    kernel = args.kernel if args.kernel is not None else cfg.get("kernel", 3)
    if hidden is None or blocks is None:
        raise ValueError(
            "Could not determine --hidden / --blocks: not provided on the CLI "
            "and not present in ckpt['config']. Pass them explicitly."
        )
    return int(hidden), int(blocks), int(kernel)


def _load_ckpt_into(model: torch.nn.Module, ckpt: dict, prefer: str) -> str:
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

    # Load ckpt first so we can resolve hidden / blocks from its stored config.
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    hidden, blocks, kernel = _resolve_model_hparams(args, ckpt)

    if args.code == "surface":
        if args.distance is None:
            raise ValueError("--distance is required for surface")
        code = SurfaceCode(distance=args.distance)
        rounds = args.rounds or args.distance
        model, flat_idx, grid_shape = build_model(
            code, rounds, hidden, blocks, kernel
        )
    else:
        if args.bb_variant is None:
            raise ValueError("--bb-variant is required for bb")
        code = (BBCode.code_72_12_6() if args.bb_variant == "72"
                else BBCode.code_144_12_12())
        rounds = args.rounds or (6 if args.bb_variant == "72" else 12)
        layout = code.detector_layout(rounds=rounds)
        flat_idx, grid_shape = make_grid_indexer(layout)
        model = BBCascadeModel(code=code, layout=layout,
                               hidden=hidden, num_blocks=blocks)

    src = _load_ckpt_into(model, ckpt, args.prefer)
    model.to(device).eval()
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)

    bposd_max = args.bposd_shots if args.bposd_shots is not None else args.shots
    # In target-failures mode, default the sufficiency threshold to the target
    # so under-sampled points (cap hit before target) are flagged automatically.
    if args.target_failures is not None and args.min_errors == 0:
        args.min_errors = args.target_failures
    sweep: list[dict] = []
    mode = (f"target_failures={args.target_failures} (cap {args.shots})"
            if args.target_failures is not None else f"shots={args.shots}")
    print(f"[eval] code={code.name} rounds={rounds} weights={src} "
          f"{mode} bposd={'on' if args.bposd else 'off'}")
    print(f"{'p':>8}  {'decoder':>10}  {'fail/total':>16}  {'p_block':>10}  "
          f"{'P_L/cycle':>14}  CI95")
    for p_phys in args.p:
        results = evaluate_decoders(
            code=code, rounds=rounds, p=p_phys, n_shots=args.shots,
            model=model, flat_idx=flat_idx, grid_shape=grid_shape,
            device=device, batch=args.batch,
            include_mwpm=not args.no_mwpm,
            include_bposd=args.bposd,
            bposd_max_shots=bposd_max,
            target_failures=args.target_failures,
        )
        result_dicts: dict[str, dict] = {}
        for name, r in results.items():
            sufficient = bool(r.n_failures >= args.min_errors)
            flag = "" if (args.min_errors == 0 or sufficient) else "  [insufficient]"
            print(f"{p_phys:>8.4f}  {name:>10}  {r.n_failures:>7}/{r.n_total:<7}  "
                  f"{r.p_block:>10.5f}  {r.p_l_per_cycle:>10.4e}  "
                  f"[{r.p_l_lo:.2e}, {r.p_l_hi:.2e}]{flag}")
            d = asdict(r)
            d["sufficient"] = sufficient
            result_dicts[name] = d
        sweep.append({
            "p": float(p_phys),
            "results": result_dicts,
        })

    payload = {
        "code": code.name,
        "rounds": rounds,
        "hidden": hidden,
        "blocks": blocks,
        "kernel": kernel,
        "ckpt": str(args.ckpt),
        "weights": src,
        "shots": args.shots,
        "target_failures": args.target_failures,
        "min_errors": args.min_errors,
        "sweep": sweep,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
