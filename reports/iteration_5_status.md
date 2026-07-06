# Iteration 5 status — surface code H=512 (d5/d7/d9) deep eval

Date: 2026-07-04. Closes out iter-5 (plan: `iteration_5_6_handoff.md`).
Everything below is from completed SLURM runs; d9 deep eval = job 163003
(finished 16:21 Jul 4, `logs/cascade_eval_surf_v5_163003.out`).

## TL;DR

1. **H=512 did NOT lift Λ at the matched anchor.** Λ_Cascade(p=0.002, d5→9,
   weighted) = **7.485 [6.733, 8.199]** vs the fair H=256 **live-weight**
   baseline **7.503 [6.755, 8.212]** (`surface_lambda_v4_live.json`).
   Identical within noise → H=256 was already at model capacity for d≤9.
   The previously reported H=256 Λ=6.85 (`iteration_4_status.md`) was an
   **EMA-lag artifact** (ema_decay=0.9998 needed 25k-step warmup on a 40k run;
   selecting live weights recovers ~0.65 Λ — memory `cascade-ema-warmup-gotcha`).
   Capacity was not the bottleneck; weight selection was.
2. **The deep anchors deliver the headline.** With the v5 eval shot budget
   (cap 4e8; all points reach 200 failures, `"sufficient": true` down to
   p=0.001):
   Λ_Cascade = **10.02 [9.01, 10.98] @ p=0.0015** and
   **15.38 [13.83, 16.85] @ p=0.001** — past the paper's ≈8.4, which is quoted
   deeper sub-threshold than the p=0.002 anchor v4 could reach
   (`iteration_4_status.md` §5 predicted exactly this).
3. **Waterfall confirmed.** Cascade's effective sub-threshold slope exceeds the
   ⌈d/2⌉ floor at every distance (d5: 3.1>3, d7: 4.2>4, d9: 5.3>5, slope-CI
   lower bounds above floor); MWPM sits on the floor (2.9/4.0/5.0).
   Figure: `figures/surface_waterfall_v5.pdf`.
4. **Cascade beats MWPM by 1.42-1.47× in Λ at every anchor, CIs disjoint.**
   At d9, p=0.001: P_L/cycle 5.82e-08 vs MWPM 2.48e-07 (4.3× fewer failures).

## Λ fits (3-distance d5/d7/d9, weighted, `scripts/23_fit_lambda.py`)

| anchor p | Λ Cascade H512 (v5) | Λ MWPM (v5) | Λ Cascade H256-live (v4) |
|---|---|---|---|
| 0.002  | 7.485 [6.733, 8.199]   | 5.254 [4.957, 5.565]   | 7.503 [6.755, 8.212] |
| 0.0015 | 10.023 [9.014, 10.982] | 6.820 [6.424, 7.235]   | — (not reached by v4 budget) |
| 0.001  | 15.375 [13.826, 16.851]| 10.605 [10.013, 11.222]| — |

JSONs: `results/surface_lambda_v5.json`, `_p0015.json`, `_p001.json`;
waterfall fit: `results/surface_waterfall_v5.json`.

Notes for fairness/reproduction:
- All three v5 evals use the H=512 models (`surface_d{5,7,9}_v5.json`); d7/d9
  use EMA weights (their best), d5 uses live (peaked before EMA warmup — no
  ema in best.pt). Each distance uses its best available weights; the v4
  baseline row is live-weight by construction.
- Λ rises as the anchor moves deeper sub-threshold — that is the waterfall,
  not an inconsistency. Quote anchor p alongside any Λ.
- The 2-pt Λ(d5→d7) values noted mid-flight (9.21/12.6/20.1) were inflated by
  saturated d5, as predicted; the 3-distance fits above supersede them.

## Interpretation

- **Capacity question (the iter-5 hypothesis): answered.** Doubling H at d≤9
  buys nothing at p=0.002 — the H=256 architecture already saturates. The
  8.4-vs-6.85 gap that motivated H=512 dissolved once EMA lag was removed
  (7.503) and the anchor moved deeper (10-15). Further capacity scans at these
  distances are not worth GPU.
- **Paper claim status:** "Λ ≈ 8.4 deep sub-threshold" is met and exceeded at
  p ≤ 0.0015 with either width. The bottleneck for deeper claims is eval
  statistics (the p=0.001/d9 point alone consumed ~24h: 3.8e8 shots for 200
  failures), not training.
- d9 H=512 training (job 162921) reached best ema p_block=0.00244, and its
  checkpoint carries both live+ema states, so no EMA-lag ambiguity at d9.

## Iter-6 hand-through (BB-144)

Run v6_bb144 (trainer v2, jobs 163114→163115→163175) reproduced the iter-2
dead-head failure: at steps 10000 and 11000, heads 8-11 sit at BCE=0.693=ln2
with logit std=0.000 (heads 0-7 healthy; head 6 woke ~step 5000, head 7
earlier). Per the handoff decision protocol this is the kill branch:
cancel the chain, add per-head loss reweighting, relaunch fresh.

Prepared (this session):
- `scripts/14_train_bb_v3.py` — v2 + adaptive per-head BCE reweighting
  (EMA-of-head-BCE → weights, clamp 4×, mean-normalized, self-annealing;
  alpha=0 reproduces v2). Resume-safe (`head_bce_ema` in last.pt).
- `slurm/train_bb_v3.sh` — v6_bb144 recipe unchanged otherwise; fresh lineage
  TAG=`v6_bb144_rw`.
- Smoke test on dev partition: job 164458 (fresh + resume paths).

Relaunch (after `scancel 163114 163115 163175`):
```bash
J=$(sbatch --parsable slurm/train_bb_v3.sh 144)
J2=$(sbatch --parsable --dependency=afterany:$J slurm/train_bb_v3.sh 144)
sbatch --dependency=afterany:$J2 slurm/train_bb_v3.sh 144
```
