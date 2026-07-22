"""Train Cascade decoder on BB code — Track 2 recipe + per-head loss reweighting.

v3 = v2 (Muon + Lion + 3-stage curriculum + EMA + bf16) + adaptive per-head
BCE reweighting, added after iter-6 run v6_bb144 (jobs 163114/115/175) hit the
iter-2 dead-head failure again: heads 8-11 stuck at BCE=ln2, std=0.000 through
step 11000 while heads 0-7 trained fine (reports/iteration_5_6_handoff.md §iter-6).

Mechanism: keep an EMA of each head's training BCE; weight head k's loss by
    w_k = clamp((bce_ema_k / mean(bce_ema))^alpha, 1/clamp, clamp),  mean-normalized.
Heads still at chance (BCE ~0.693) get up-weighted relative to converged heads,
and the weights anneal back toward 1 automatically as heads wake — no
hand-picked head set. alpha=0 disables (exact v2 behaviour).

Usage:
    python scripts/14_train_bb_v3.py --code 144 --steps 40000 --head-reweight-alpha 1.0
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
from cascade.data.stim_dataset import PDist, StimMemoryDataset
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
    # Mixed-p training (iter-7 plan §3b). Mixed mode is ON iff (--p-min AND
    # --p-max) or --p-list is given; otherwise --p-train drives the original
    # single-p path verbatim (backward compat).
    p.add_argument("--p-min", type=float, default=None,
                   help="low end of the mixed-p sampling range; must be given "
                        "together with --p-max (mixed-p range mode)")
    p.add_argument("--p-max", type=float, default=None,
                   help="high end of the mixed-p sampling range; also becomes "
                        "the curriculum's stage-2 anneal target (p2) in mixed "
                        "mode. Must be given together with --p-min")
    p.add_argument("--p-sampling", choices=["log-uniform", "uniform", "grid"],
                   default="log-uniform",
                   help="mixed-p draw distribution over the log-spaced grid; "
                        "'grid' draws uniformly from the grid points "
                        "themselves (PDist mode='list')")
    p.add_argument("--p-grid-points", type=int, default=16,
                   help="number of log-spaced cache/snap points spanning "
                        "[p-min, p-max]")
    p.add_argument("--p-list", type=float, nargs="+", default=None,
                   help="explicit list of p values for mixed-p sampling "
                        "(PDist mode='list'); also turns mixed mode on and "
                        "overrides --p-min/--p-max range construction")
    p.add_argument("--p-weights", type=float, nargs="+", default=None,
                   help="optional per-point weights for --p-list (must match "
                        "its length); default uniform. Requires --p-list")
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
    p.add_argument("--head-reweight-alpha", type=float, default=1.0,
                   help="exponent on (head BCE / mean BCE) for per-head loss "
                        "weights; 0 = uniform weighting (v2 behaviour)")
    p.add_argument("--head-reweight-clamp", type=float, default=4.0,
                   help="per-head weights clamped to [1/clamp, clamp] before "
                        "mean-normalization")
    p.add_argument("--head-reweight-ema", type=float, default=0.98,
                   help="EMA decay (per optimizer step) of per-head training BCE "
                        "used to set the weights")
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
    p.add_argument("--tag", type=str, default="v3")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Fail fast on ambiguous mixed-p flag combos (plan §3b decision: silent
    # fallback to single-p is forbidden).
    if (args.p_min is None) != (args.p_max is None):
        p.error("--p-min and --p-max must be given together (mixed-p range "
                "mode); got only one — refusing to silently fall back to "
                "single-p")
    if args.p_weights is not None and args.p_list is None:
        p.error("--p-weights requires --p-list")
    if args.p_list is not None and args.p_weights is not None \
            and len(args.p_weights) != len(args.p_list):
        p.error(f"--p-weights length ({len(args.p_weights)}) must match "
                f"--p-list length ({len(args.p_list)})")
    return args


def build_bb(code: BBCode, rounds: int, hidden: int, blocks: int, device):
    layout = code.detector_layout(rounds=rounds)
    flat_idx, grid_shape = make_grid_indexer(layout)
    flat_idx = flat_idx.to(device)
    grid_shape = grid_shape.to(device)
    model = BBCascadeModel(code=code, layout=layout, hidden=hidden, num_blocks=blocks).to(device)
    return model, layout, flat_idx, grid_shape


def build_p_dist(args: argparse.Namespace) -> tuple[PDist | None, float | None, float | None]:
    """Build the mixed-p sampling spec from CLI args (iter-7 plan §3b).

    Returns ``(p_dist, p_min_eff, p_max_eff)``. ``p_dist`` is ``None`` when
    mixed-p is off (no new flags given — single-p path is unaffected).
    ``p_min_eff``/``p_max_eff`` are the low/high monitor-eval points: the
    min/max of an explicit ``--p-list``, or ``--p-min``/``--p-max`` for the
    log-spaced range modes.
    """
    mixed = (args.p_min is not None) or (args.p_list is not None)
    if not mixed:
        return None, None, None
    # Derived, not reused: keeps the p-stream independent of the data seed
    # while staying reproducible run-to-run.
    dist_seed = args.seed + 10_000
    if args.p_list is not None:
        grid = tuple(float(x) for x in args.p_list)
        weights = tuple(float(w) for w in args.p_weights) if args.p_weights is not None else None
        p_dist = PDist(mode="list", grid=grid, weights=weights, seed=dist_seed)
        return p_dist, min(grid), max(grid)
    # Range mode: caller builds the grid — the dataset does not.
    grid = tuple(float(x) for x in np.geomspace(args.p_min, args.p_max, args.p_grid_points))
    if args.p_sampling == "grid":
        # "grid" sampling choice maps to PDist(mode="list", weights=None ->
        # uniform draw over the log-spaced points themselves).
        p_dist = PDist(mode="list", grid=grid, weights=None, seed=dist_seed)
    else:
        p_dist = PDist(mode=args.p_sampling, grid=grid,
                        p_min=args.p_min, p_max=args.p_max, seed=dist_seed)
    return p_dist, args.p_min, args.p_max


def head_weights(bce_ema: torch.Tensor, alpha: float, clamp: float) -> torch.Tensor:
    """Per-head loss weights from the BCE EMA: mean-normalized, clamped ratio^alpha."""
    if alpha <= 0.0:
        return torch.ones_like(bce_ema)
    w = (bce_ema / bce_ema.mean().clamp(min=1e-8)) ** alpha
    w = w.clamp(min=1.0 / clamp, max=clamp)
    return w * (w.numel() / w.sum())


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

    p_dist, mix_p_min, mix_p_max = build_p_dist(args)
    mixed = p_dist is not None
    # Mixed mode: p_max defaults the curriculum's stage-2 anneal target;
    # --p-train keeps separately naming the high-p monitor eval point (they
    # are equal in the v7 launcher, but conceptually distinct knobs).
    curriculum = CurriculumConfig(p1=args.p_warmup, p2=mix_p_max if mixed else args.p_train)
    # Dataset starts in single-p warmup mode regardless of mixed-p — passing
    # p_dist at construction would call set_mixed immediately (mixed from
    # step 1), which is not what we want; set_mixed is called later, once,
    # at the anneal_end stage boundary.
    ds = StimMemoryDataset(code=code, rounds=rounds, p=curriculum.p_at(0, args.steps),
                            batch_size=args.batch, seed=args.seed)
    if mixed:
        print(f"[main] mixed-p: mode={p_dist.mode} grid={len(p_dist.grid)} pts "
              f"over [{mix_p_min:.4g},{mix_p_max:.4g}] (p_sampling={args.p_sampling}), "
              f"curriculum p2->p_max, p_train(high-p monitor)={args.p_train:.4g}, "
              f"dist_seed={p_dist.seed}")
    opts = build_muon_lion(
        model, muon_lr=args.muon_lr, muon_weight_decay=args.weight_decay,
        lion_lr=args.lion_lr, lion_weight_decay=0.0,
    )
    ema = ParamEMA(model, decay=args.ema_decay)

    accum_steps = max(1, int(args.accum_steps))
    effective_batch = args.batch * accum_steps
    obs_count = code.num_logical_observables
    # Per-head training-BCE EMA driving the loss reweighting. Init at ln2
    # (chance) so early weights are uniform until heads differentiate.
    head_bce_ema = torch.full((obs_count,), float(np.log(2.0)), device=device)
    print(f"[main] params={n_params:,}, micro_batch={args.batch}, "
          f"accum={accum_steps}, effective_batch={effective_batch}, "
          f"p1={curriculum.p1:.4f}→p2={curriculum.p2:.4f}, "
          f"H={args.hidden}, L={args.blocks}, ema={args.ema_decay}")
    print(f"[main] head-reweight: alpha={args.head_reweight_alpha}, "
          f"clamp={args.head_reweight_clamp}, bce_ema_decay={args.head_reweight_ema}")

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
            "head_bce_ema": head_bce_ema.detach().cpu(),
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
        if ck.get("head_bce_ema") is not None:
            head_bce_ema = ck["head_bce_ema"].to(device)
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

    # Mixed-p stage switch (plan §3b): stages 1-2 stay on the single-p
    # curriculum (warmup at p1, anneal to p2=mix_p_max) exactly as before; at
    # anneal_end (stage-3 boundary) call ds.set_mixed once, re-create
    # iter_ds, and never call ds.set_noise again. mixed_active is a fresh
    # local (not persisted in the checkpoint) so it self-heals across
    # resumes/segment rollovers: the first loop iteration re-checks
    # step >= anneal_end and re-switches immediately if we resumed past it.
    anneal_end = int(curriculum.warmup_frac * args.steps) + int(curriculum.anneal_frac * args.steps)
    mixed_active = False
    p_draw_tally: dict[float, int] = {}

    for step in range(start_step + 1, args.steps + 1):
        # Curriculum + LR update — once per *opt* step, before accum loop (Risk R3).
        p_now = curriculum.p_at(step, args.steps)
        if mixed and not mixed_active and step >= anneal_end:
            ds.set_mixed(p_dist)
            iter_ds = iter(ds)
            mixed_active = True
            print(f"  [mixed-p] switched dataset to mixed-p sampling at step {step} "
                  f"(anneal_end={anneal_end}, grid={len(p_dist.grid)} pts over "
                  f"[{mix_p_min:.4g},{mix_p_max:.4g}], mode={p_dist.mode})")
        elif not mixed_active and abs(p_now - ds.p) > 1e-6:
            ds.set_noise(p_now)
            iter_ds = iter(ds)

        sched_mult = cosine_with_warmup(step, total_steps=args.steps, warmup_steps=1000)
        for g in opts["muon"].param_groups:
            g["lr"] = args.muon_lr * sched_mult
        for g in opts["lion"].param_groups:
            g["lr"] = args.lion_lr * sched_mult

        # Head weights frozen across the accum loop so every micro-batch of one
        # optimizer step sees the same loss surface.
        w_heads = head_weights(head_bce_ema, args.head_reweight_alpha,
                               args.head_reweight_clamp).detach()

        model.train()
        zero_grad_muon_lion(opts)
        loss_accum = 0.0
        step_head_bce = torch.zeros_like(head_bce_ema)
        for _ in range(accum_steps):
            det, obs = next(iter_ds)
            if mixed_active:
                pd = ds.last_p
                if pd is not None:
                    p_draw_tally[pd] = p_draw_tally.get(pd, 0) + 1
            det = det.to(device, non_blocking=True)
            obs = obs.to(device, non_blocking=True)
            grid = detection_events_to_grid(det, flat_idx, grid_shape)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
                logits = model(grid)
            # Cast to fp32 before scaling so backward stays fp32 — avoids
            # bf16 underflow when accum_steps is large.
            per_head_bce = F.binary_cross_entropy_with_logits(
                logits.float(), obs, reduction="none").mean(dim=0)
            loss = (w_heads * per_head_bce).mean() / accum_steps
            loss.backward()
            loss_accum += loss.item()
            step_head_bce += per_head_bce.detach() / accum_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step_muon_lion(opts)
        ema.update(model)
        # One EMA update per optimizer step, from the effective-batch mean.
        d = args.head_reweight_ema
        head_bce_ema = d * head_bce_ema + (1.0 - d) * step_head_bce
        loss_logged = loss_accum  # already scaled by 1/accum so it's the mean weighted BCE

        if step % 200 == 0 or step == 1:
            mixed_suffix = (f" p_draw={ds.last_p:.4g} cache={ds.sampler_cache_size}"
                             if mixed_active else "")
            print(f"  step {step:6d} | p={p_now:.4f} lr_mult={sched_mult:.3f} "
                  f"loss {loss_logged:.4f} | {(step - start_step) / (time.time() - t_start):.2f} steps/s"
                  f"{mixed_suffix}")

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
            # Low-p monitor eval (mixed mode only, plan §3b) — the key live
            # evidence that generalization is/isn't improving. Runs for the
            # whole training when mixed flags are given (not gated on
            # mixed_active) so the log shows the pre-switch baseline too.
            live_metrics_lowp = None
            ema_metrics_lowp = None
            if mixed:
                live_metrics_lowp = evaluate_bb(model, code, rounds, mix_p_min,
                                                 8192, flat_idx, grid_shape, device,
                                                 batch=args.eval_batch, use_bf16=use_bf16)
                if step >= ema_warm_steps:
                    with ema.applied(model):
                        model.eval()
                        ema_metrics_lowp = evaluate_bb(model, code, rounds, mix_p_min,
                                                        8192, flat_idx, grid_shape, device,
                                                        batch=args.eval_batch, use_bf16=use_bf16)
            history.append(dict(step=step, live=live_metrics, ema=ema_metrics,
                                live_lowp=live_metrics_lowp, ema_lowp=ema_metrics_lowp,
                                head_weights=w_heads.detach().cpu().tolist()))
            ema_str = f"ema p_block={ema_metrics['p_block']:.5f}" if ema_metrics else "ema=cold"
            print(f"  [eval] step={step} live p_block={live_metrics['p_block']:.5f} "
                  f"bce={live_metrics['bce']:.4f}  {ema_str}")
            if mixed:
                ema_lowp_str = (f"ema p_block={ema_metrics_lowp['p_block']:.5f}"
                                 if ema_metrics_lowp else "ema=cold")
                print(f"  [eval low-p] step={step} p={mix_p_min:.4g} "
                      f"live p_block={live_metrics_lowp['p_block']:.5f} "
                      f"bce={live_metrics_lowp['bce']:.4f}  {ema_lowp_str}")
            # Per-logical BCE / std (Risk R1/R2 monitoring)
            pl_bce = live_metrics["per_logical_bce"]
            pl_std = live_metrics["per_logical_std"]
            print("    per-logical BCE: [" +
                  ", ".join(f"{b:.3f}" for b in pl_bce) + "]")
            print("    per-logical std: [" +
                  ", ".join(f"{s:.3f}" for s in pl_std) + "]")
            print("    head loss w:     [" +
                  ", ".join(f"{v:.2f}" for v in w_heads.cpu().tolist()) + "]")
            if mixed_active:
                # Drawn-p variety + cache-bound observability (smoke §9.2).
                tally_str = "{" + ", ".join(
                    f"{k:.4g}: {v}" for k, v in sorted(p_draw_tally.items())) + "}"
                print(f"    mixed-p draws: {tally_str}  cache_size={ds.sampler_cache_size}")
                p_draw_tally.clear()
            best = ema_metrics or live_metrics
            if mixed:
                # Generalization-weighted selection (plan §3b, required, not
                # optional): reward high-p AND low-p jointly so best.pt isn't
                # a high-p-specialized checkpoint. Uses the SAME (EMA-
                # preferred) evals computed above, matching today's
                # preference order.
                best_lowp = ema_metrics_lowp or live_metrics_lowp
                selection_metric = 0.5 * (best["p_block"] + best_lowp["p_block"])
                selection_desc = (
                    f"0.5*(p_block@p_max[{mix_p_max:.4g}]={best['p_block']:.5f} + "
                    f"p_block@p_min[{mix_p_min:.4g}]={best_lowp['p_block']:.5f}) "
                    f"= {selection_metric:.5f}")
            else:
                selection_metric = best["p_block"]
                selection_desc = f"p_block@p_train[{args.p_train:.4g}]={selection_metric:.5f}"
            if selection_metric < best_p_block:
                best_p_block = selection_metric
                best_step = step
                # save checkpoint
                ema_state_dict = None
                if ema_metrics is not None:
                    with ema.applied(model):
                        ema_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
                cfg_dict = vars(args).copy()
                cfg_dict["out"] = str(cfg_dict["out"])
                cfg_dict["effective_batch"] = effective_batch
                if mixed:
                    # Extra observability line — mixed mode only, so single-p
                    # stdout stays byte-identical to today.
                    print(f"  [best.pt] step={step} saved on metric: {selection_desc}")
                torch.save({
                    "model": model.state_dict(),
                    "model_ema": ema_state_dict,
                    "ema": ema.state_dict(),
                    "config": cfg_dict,
                    "history": history,
                    "best_step": best_step,
                    "best_p_block": best_p_block,
                    "best_metric_desc": selection_desc,
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
    print(f"[train v3] done in {elapsed/60:.1f} min "
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
