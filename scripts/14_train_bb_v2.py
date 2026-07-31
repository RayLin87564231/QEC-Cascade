"""Train Cascade decoder on BB code with Track 2 recipe.

Track 2 = Muon + Lion + 3-stage curriculum + EMA + bf16 mixed precision.

Usage:
    python scripts/14_train_bb_v2.py --code 72 --steps 40000
    python scripts/14_train_bb_v2.py --code 144 --steps 60000 --hidden 256
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cascade.codes.bb import BBCode
from cascade.data.stim_dataset import StimMemoryDataset
from cascade.data.tensorize import detection_events_to_grid, make_grid_indexer
from cascade.models.cascade_bb import BBCascadeModel
from cascade.models.ema import ParamEMA
from cascade.train.curriculum import CurriculumConfig
from cascade.train.lr_schedule import cosine_with_warmup
from cascade.train.optimizers import (
    build_muon_lion, step_muon_lion, zero_grad_muon_lion,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--code", choices=["72", "144"], default="72")
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--p-train", type=float, default=0.0055)  # paper p_2
    p.add_argument("--p-warmup", type=float, default=0.001)
    p.add_argument("--p-eval", type=float, nargs="+",
                   default=[0.001, 0.002, 0.003, 0.004, 0.005])
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch", type=int, default=256,
                   help="micro-batch size; effective batch = batch * accum_steps")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="gradient accumulation factor; effective batch = batch * "
                        "accum_steps. Used to reach paper batch=3328 without OOM "
                        "(set 13 for batch=256 micro).")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--muon-lr", type=float, default=3e-3)
    p.add_argument("--lion-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=3e-3)
    p.add_argument("--ema-decay", type=float, default=0.9998)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-batch", type=int, default=4096,
                   help="per-forward eval chunk (not the eval sample count); "
                        "lower for BB-144 to avoid the scatter-einsum OOM "
                        "(e.g. 256). Also used for the final-eval sweep.")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore any last.pt and start fresh (default: auto-resume)")
    p.add_argument("--final-shots", type=int, default=200000)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "checkpoints")
    p.add_argument("--tag", type=str, default="v2")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_bb(code: BBCode, rounds: int, hidden: int, blocks: int, device):
    layout = code.detector_layout(rounds=rounds)
    flat_idx, grid_shape = make_grid_indexer(layout)
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)
    model = BBCascadeModel(code=code, layout=layout, hidden=hidden, num_blocks=blocks).to(device)
    return model, layout, flat_idx, grid_shape


@torch.inference_mode()
def evaluate_bb(model, code, rounds, p, n_shots, flat_idx, grid_shape, device,
                batch=4096, use_bf16=True):
    sampler = code.make_circuit(p=p, rounds=rounds).compile_detector_sampler()
    n_fail = n_total = 0
    bce_sum = 0.0
    bce_n = 0
    obs_count = code.num_logical_observables
    per_logical_bce_sum = torch.zeros(obs_count, dtype=torch.float64)
    per_logical_std_sumsq = torch.zeros(obs_count, dtype=torch.float64)
    per_logical_sum = torch.zeros(obs_count, dtype=torch.float64)
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    while n_total < n_shots:
        this = min(batch, n_shots - n_total)
        det, obs = sampler.sample(shots=this, separate_observables=True, bit_packed=False)
        det_t = torch.from_numpy(det.astype(np.float32)).to(device)
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(device)
        grid = detection_events_to_grid(det_t, flat_idx, grid_shape)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
            logits = model(grid)
        logits = logits.float()
        preds = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
        err = (preds != obs_t.to(torch.uint8)).any(dim=1)
        n_fail += int(err.sum().item())
        n_total += this
        bce_sum += float(F.binary_cross_entropy_with_logits(logits, obs_t, reduction="sum").item())
        bce_n += this * obs_count
        # Per-logical accumulators (Risk R1/R2 monitoring)
        per_logical_bce_sum += F.binary_cross_entropy_with_logits(
            logits, obs_t, reduction="none").sum(dim=0).double().cpu()
        logits_cpu = logits.double().cpu()
        per_logical_sum += logits_cpu.sum(dim=0)
        per_logical_std_sumsq += (logits_cpu * logits_cpu).sum(dim=0)
    per_logical_bce = (per_logical_bce_sum / max(n_total, 1)).tolist()
    mean = per_logical_sum / max(n_total, 1)
    var = per_logical_std_sumsq / max(n_total, 1) - mean * mean
    per_logical_std = var.clamp(min=0).sqrt().tolist()
    return dict(p_block=n_fail / max(n_total, 1), n_failures=n_fail,
                n_total=n_total, bce=bce_sum / max(bce_n, 1),
                per_logical_bce=per_logical_bce,
                per_logical_std=per_logical_std)


def main() -> None:
    args = parse_args()
    code = BBCode.code_72_12_6() if args.code == "72" else BBCode.code_144_12_12()
    rounds = args.rounds or (6 if args.code == "72" else 12)
    out_dir = args.out / f"{code.name}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}, code={code.name}, rounds={rounds}")

    torch.manual_seed(args.seed)
    model, layout, flat_idx, grid_shape = build_bb(code, rounds, args.hidden, args.blocks, device)
    n_params = sum(p.numel() for p in model.parameters())

    curriculum = CurriculumConfig(p1=args.p_warmup, p2=args.p_train)
    ds = StimMemoryDataset(code=code, rounds=rounds, p=curriculum.p_at(0, args.steps),
                            batch_size=args.batch, seed=args.seed)
    opts = build_muon_lion(
        model, muon_lr=args.muon_lr, muon_weight_decay=args.weight_decay,
        lion_lr=args.lion_lr, lion_weight_decay=0.0,
    )
    ema = ParamEMA(model, decay=args.ema_decay)

    accum_steps = max(1, int(args.accum_steps))
    effective_batch = args.batch * accum_steps
    print(f"[main] params={n_params:,}, micro_batch={args.batch}, "
          f"accum={accum_steps}, effective_batch={effective_batch}, "
          f"p1={curriculum.p1:.4f}→p2={curriculum.p2:.4f}, "
          f"H={args.hidden}, L={args.blocks}, ema={args.ema_decay}")

    use_bf16 = not args.no_bf16
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    history = []
    best_p_block = float("inf")
    best_step = -1
    start_step = 0
    last_path = out_dir / "last.pt"

    def save_last(step: int) -> None:
        """Full resume snapshot — survives preemption on the >48h BB-144 run.

        Written atomically (tmp + rename) so a preemption mid-write never
        leaves a truncated last.pt.
        """
        tmp = out_dir / "last.pt.tmp"
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "muon": opts["muon"].state_dict(),
            "lion": opts["lion"].state_dict(),
            "rng": torch.get_rng_state(),
            "history": history,
            "best_p_block": best_p_block,
            "best_step": best_step,
        }, tmp)
        tmp.replace(last_path)

    if last_path.exists() and not args.no_resume:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"], model)
        opts["muon"].load_state_dict(ck["muon"])
        opts["lion"].load_state_dict(ck["lion"])
        if ck.get("rng") is not None:
            torch.set_rng_state(ck["rng"].cpu() if hasattr(ck["rng"], "cpu") else ck["rng"])
        history = ck.get("history", [])
        best_p_block = ck.get("best_p_block", float("inf"))
        best_step = ck.get("best_step", -1)
        start_step = int(ck["step"])
        print(f"[resume] loaded {last_path} at step {start_step} "
              f"(best p_block={best_p_block:.5f} @ {best_step}); continuing.")

    t_start = time.time()
    iter_ds = iter(ds)
    ema_warm_steps = int(5.0 / max(1e-8, 1.0 - args.ema_decay))

    # Iter-2 dead-head set (reports/iteration_2_status.md §3.6); used only for
    # the early-warning print at step 5000 (Risk R1).
    iter2_dead_heads = (6, 8, 9, 10, 11)
    early_warning_step = 5000
    early_warning_emitted = False

    for step in range(start_step + 1, args.steps + 1):
        # Curriculum + LR update — once per *opt* step, before accum loop (Risk R3).
        p_now = curriculum.p_at(step, args.steps)
        if abs(p_now - ds.p) > 1e-6:
            ds.set_noise(p_now)
            iter_ds = iter(ds)

        sched_mult = cosine_with_warmup(step, total_steps=args.steps, warmup_steps=1000)
        for g in opts["muon"].param_groups:
            g["lr"] = args.muon_lr * sched_mult
        for g in opts["lion"].param_groups:
            g["lr"] = args.lion_lr * sched_mult

        model.train()
        zero_grad_muon_lion(opts)
        loss_accum = 0.0
        for _ in range(accum_steps):
            det, obs = next(iter_ds)
            det = det.to(device, non_blocking=True)
            obs = obs.to(device, non_blocking=True)
            grid = detection_events_to_grid(det, flat_idx, grid_shape)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
                logits = model(grid)
            # Cast to fp32 before scaling so backward stays fp32 — avoids
            # bf16 underflow when accum_steps is large.
            loss = F.binary_cross_entropy_with_logits(logits.float(), obs) / accum_steps
            loss.backward()
            loss_accum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step_muon_lion(opts)
        ema.update(model)
        loss_logged = loss_accum  # already scaled by 1/accum so it's the mean BCE

        if step % 200 == 0 or step == 1:
            print(f"  step {step:6d} | p={p_now:.4f} lr_mult={sched_mult:.3f} "
                  f"loss {loss_logged:.4f} | {step / (time.time() - t_start):.2f} steps/s")

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            live_metrics = evaluate_bb(model, code, rounds, args.p_train,
                                        8192, flat_idx, grid_shape, device,
                                        batch=args.eval_batch, use_bf16=use_bf16)
            ema_metrics = None
            if step >= ema_warm_steps:
                with ema.applied(model):
                    model.eval()
                    ema_metrics = evaluate_bb(model, code, rounds, args.p_train,
                                                8192, flat_idx, grid_shape, device,
                                                batch=args.eval_batch, use_bf16=use_bf16)
            history.append(dict(step=step, live=live_metrics, ema=ema_metrics))
            ema_str = f"ema p_block={ema_metrics['p_block']:.5f}" if ema_metrics else "ema=cold"
            print(f"  [eval] step={step} live p_block={live_metrics['p_block']:.5f} "
                  f"bce={live_metrics['bce']:.4f}  {ema_str}")
            # Per-logical BCE / std (Risk R1/R2 monitoring)
            pl_bce = live_metrics["per_logical_bce"]
            pl_std = live_metrics["per_logical_std"]
            print("    per-logical BCE: [" +
                  ", ".join(f"{b:.3f}" for b in pl_bce) + "]")
            print("    per-logical std: [" +
                  ", ".join(f"{s:.3f}" for s in pl_std) + "]")
            best = ema_metrics or live_metrics
            if best["p_block"] < best_p_block:
                best_p_block = best["p_block"]
                best_step = step
                # save checkpoint
                ema_state_dict = None
                if ema_metrics is not None:
                    with ema.applied(model):
                        ema_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
                cfg_dict = vars(args).copy()
                cfg_dict["out"] = str(cfg_dict["out"])
                cfg_dict["effective_batch"] = effective_batch
                torch.save({
                    "model": model.state_dict(),
                    "model_ema": ema_state_dict,
                    "ema": ema.state_dict(),
                    "config": cfg_dict,
                    "history": history,
                    "best_step": best_step,
                    "best_p_block": best_p_block,
                }, out_dir / "best.pt")
            # Resume snapshot every eval — bounds preemption loss to eval_every.
            save_last(step)

        # Early-warning probe (Risk R1): heavy-head inertness at step 5000.
        if (not early_warning_emitted and step >= early_warning_step
                and code.num_logical_observables > max(iter2_dead_heads)):
            early_warning_emitted = True
            std_at_warning = live_metrics["per_logical_std"] if step % args.eval_every == 0 else None
            if std_at_warning is None:
                # Eval did not coincide with this step; do a cheap probe.
                model.eval()
                std_at_warning = evaluate_bb(
                    model, code, rounds, args.p_train, 2048,
                    flat_idx, grid_shape, device,
                    batch=args.eval_batch, use_bf16=use_bf16,
                )["per_logical_std"]
            heavy_inert = all(std_at_warning[k] < 0.05 for k in iter2_dead_heads)
            n_alive = sum(1 for s in std_at_warning if s > 0.1)
            tag = "INERT" if heavy_inert else "MOVING"
            print(f"  [early-warning step={step}] heavy_heads={tag} "
                  f"n_alive_total={n_alive}/{code.num_logical_observables}")
            if heavy_inert:
                print(f"  [early-warning] heavy heads inert at step {step} "
                      f"— iter-2 dead set {iter2_dead_heads} all std < 0.05; "
                      f"recipe likely insufficient (see plan Risk R1).")

    elapsed = time.time() - t_start
    print(f"[train v2] done in {elapsed/60:.1f} min "
          f"| best p_block={best_p_block:.5f} at step {best_step}")

    # Final multi-p evaluation using best (EMA or live) weights.
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    if ckpt.get("model_ema") is not None:
        model.load_state_dict(ckpt["model_ema"])
        print("[main] using EMA weights for final evaluation")
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"\n[final] BB-{args.code} sweep over p:")
    for p in args.p_eval:
        m = evaluate_bb(model, code, rounds, p, args.final_shots,
                        flat_idx, grid_shape, device,
                        batch=args.eval_batch, use_bf16=use_bf16)
        # Per-cycle P_L using paper Eq. (1)
        from cascade.eval.pblock_to_pl import per_cycle_pl_with_ci
        p_l, p_l_lo, p_l_hi = per_cycle_pl_with_ci(
            m["n_failures"], m["n_total"],
            k=code.num_logical_observables, rounds=rounds,
        )
        print(f"  p={p:.4f}  p_block={m['p_block']:.5f}  ({m['n_failures']}/{m['n_total']})"
              f"  P_L/cycle={p_l:.4e}  CI=[{p_l_lo:.2e}, {p_l_hi:.2e}]")


if __name__ == "__main__":
    main()
