# Iteration 5 (H=512 finish) + Iteration 6 (BB-144 launch) — Handoff / Resume Notes

Snapshot: 2026-07-03 ~16:45
Supersedes: `iteration_5_h512_handoff.md` (its step 4 is DONE; steps 1-3, 5 pending d9)

---

## TL;DR for a fresh session

Two tracks in flight, all on SLURM (survive terminal close):

1. **iter-5 (surface H=512)**: only d9 deep eval left — job **163003**, started
   16:33 Jul 3, ~19 h (p=0.001 point dominates) → writes
   `results/surface_d9_v5.json` ~**11:30 Jul 4**. Then run the 3 wrap-up steps below.
2. **iter-6 (BB-144 Gross code)**: training chain **163114 → 163115 → 163175**
   (3 × 48 h, afterany, auto-resume from `checkpoints/bb_144_12_12_v6_bb144/last.pt`).
   ETA ~Jul 8 morning. **Watch the dead-head risk** (§ iter-6 below).

`squeue -u leo07010` to see everything. All commands run from
`/work/leo07010/Ray/QEC/cascade` with `source .venv/bin/activate`.

---

## Iter-5 state

### Done
- d5, d7 H=512 train + deep eval: `results/surface_d{5,7}_v5.json` (all p-points
  200+ failures down to p=0.001).
- d9 H=512 train (job 162921, completed 16:33 Jul 3, best ema p_block=0.00244
  — better than d7's 0.00342; checkpoint has BOTH live and ema).
- **KEY NEW RESULT — H=256 live-weight re-eval (was step 4 of old handoff):**
  `results/surface_d{5,7,9}_v4_live.json` + `results/surface_lambda_v4_live.json`
  → **Λ_H256(live) = 7.503 [6.755, 8.212]** vs EMA-weights 6.851 [6.162, 7.508].
  The ema_decay=0.9998 lag was masking ~0.65 of Λ. The fair H512-vs-H256
  baseline is now **7.503**, not 6.85. (Λ_MWPM live = 5.375 [5.054, 5.711].)
- Preliminary 2-pt Λ(d5→d7) from v5: 9.21 @ p=0.002, 12.6 @ p=0.0015,
  20.1 @ p=0.001 — waterfall climbing exactly as iter-4 §5 predicted
  (statistics-limited, not model-limited). NB: 2-pt values are inflated by
  saturated d5; the 3-distance fit will be lower and hinges on d9.
- d7 v5 eval used **ema** weights (prefer auto): correct — d7 best was ema
  0.00342 @ step 39k. The "live > ema on H=512" handoff caveat only applied
  mid-training / to d5 (which peaked pre-warmup and has no ema in best.pt).

### Remaining (after `results/surface_d9_v5.json` appears, ~11:30 Jul 4)

Check first: job 163003 COMPLETED and JSON exists. If the job FAILED, check
`logs/cascade_eval_surf_v5_163003.{out,err}`; resubmit `sbatch slurm/eval_surface_v5.sh 9`.

```bash
# 1. Λ fit at p=0.002 anchor + deeper anchors
python scripts/23_fit_lambda.py \
    --inputs results/surface_d5_v5.json results/surface_d7_v5.json results/surface_d9_v5.json \
    --decoders Cascade MWPM --p 0.002 --weighted \
    --out results/surface_lambda_v5.json
# repeat with --p 0.0015 and --p 0.001 (use only if d9 has 200 failures there;
# check "sufficient": true in the JSON), writing
# results/surface_lambda_v5_p0015.json / _p001.json

# 2. Waterfall figure
python scripts/28_waterfall_fit.py \
    --inputs results/surface_d5_v5.json results/surface_d7_v5.json results/surface_d9_v5.json \
    --p-max 0.006 --decoders Cascade MWPM --weighted \
    --fig figures/surface_waterfall_v5.pdf --out results/surface_waterfall_v5.json

# 3. Write reports/iteration_5_status.md
```

The report must answer: did H=512 lift Λ above the **7.503 live baseline**?
Compare at matched anchors (p=0.002) and highlight the deeper anchors
(0.0015 / 0.001) that v4 could not reach. Include the EMA-lag finding
(6.85 → 7.50 was weight-selection, not capacity) and cite memory
`cascade-ema-warmup-gotcha`. Both outcomes are publishable: Λ up → capacity
helped; Λ flat → H=256 was already at capacity for d≤9 and the paper's 8.4
needs deeper p / larger d (which the deep anchors may already show).

---

## Iter-6 state (BB-144 [[144,12,12]] Gross code)

### Launched 10:49 Jul 3 with two fixes vs the parked recipe
- `slurm/train_bb.sh` case 144: **EMA_DECAY=0.999** (warmup 5k steps = steps/8;
  the old 0.9998 needed 25k and cost iter-5 ~0.65 Λ — memory
  `cascade-ema-warmup-gotcha`), and **TAG=v6_bb144** (lineage matches iter-6).
- 40k steps, micro 48 × accum 69 = eff 3312, H=256, L=12, rounds=12,
  curriculum p 0.001→0.0055, 4.16M params.
- Measured rate 0.10 steps/s → ~111 h → 3 chained segments queued:
  **163114 → 163115 → 163175** (afterany; resume-at-40k exits fast, so a
  spare segment is harmless). If segment 3 ends below 40k steps:
  `sbatch --dependency=afterany:163175 slurm/train_bb.sh 144`.

### ⚠️ Dead-head watch (the one real risk)
step-2000 eval: heads 0-5 healthy (BCE 0.36-0.38, std 2.6-2.9), head 7 waking
(std 1.29), **heads 6, 8-11 still at chance with std 0.02-0.19** — the same
high-lz-weight heads (14-20) that died permanently in iter-2 BB-72
(`iteration_2_status.md` §3.6). Not yet a kill signal: iter-2's L7 woke
between 3k and 40k, and d=12 heads plausibly wake slower; the iter-3 fix
(curriculum + eff-batch 3328) is what's being tested here.

Decision protocol:
- **step 5000** (~00:40 Jul 4): trainer prints a built-in early warning listing
  dead heads. Check `grep -A3 "eval] step=5000" logs/cascade_bb_163114.out`.
- **step ~10000** (~14:30 Jul 4): if heads 8-11 std is still ~0 →
  `scancel 163114 163115 163175`, add per-head loss reweighting (upweight
  high-weight logicals in the BCE mean, `scripts/14_train_bb_v2.py` loss), relaunch.
  If std is climbing (even to ~0.5), let it run.

### After training completes
`slurm/eval_bb_bposd.sh` is hardcoded to the BB-72 v3 checkpoint (line ~42) —
adapt to `checkpoints/bb_144_12_12_v6_bb144/best.pt` (+ BB-144 dims) before
submitting. Paper headline claim to test: Cascade ~4000× lower P_L than BP+OSD
on [[144,12,12]] at p=0.2-0.55%.

---

## Job/artifact map

```
163003            d9 v5 deep eval (RUNNING, ETA ~11:30 Jul 4) → results/surface_d9_v5.json
163114/115/175    BB-144 train chain → checkpoints/bb_144_12_12_v6_bb144/{last,best}.pt
logs/cascade_eval_surf_v5_163003.out    d9 eval progress
logs/cascade_bb_163114.out              BB-144 progress (+dead-head evals)
results/surface_lambda_v4_live.json     H=256 live baseline Λ=7.503  ← compare target
results/surface_d{5,7}_v5.json          H=512 evals done
reports/iteration_4_status.md           H=256 ema baseline (Λ=6.85) + context
reports/iteration_5_h512_handoff.md     previous handoff (superseded by this)
```
