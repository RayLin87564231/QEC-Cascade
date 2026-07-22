# Iteration 7 status — BB-[[144,12,12]] neural decoder (mixed-p retraining)

Date: training 2026-07-17 to 2026-07-21, formal eval 2026-07-22, closed 2026-07-22.
**FINAL — iteration 7 closed.** Training ran as the 3-segment SLURM `afterany`
chain 188708→188709→188710 (started 2026-07-17 02:44:10, `logs/cascade_bb_v7mixp_188708.out`;
completed 2026-07-21 13:26:09, `logs/cascade_bb_v7mixp_188710.out`), 40000 steps,
TAG `v7_bb144_mixp`, ~0.10-0.11 steps/s on H200. Completion verified 7/7 PASS in
`/home/u2467370/.claude/scratch/v7_completion_check.md` (no NaNs/tracebacks/OOM in
any of the three `.out` logs; both `.err` files are the expected SLURM TIMEOUT
rollover message; v6 checkpoints/results untouched). The formal external eval ran
as job 203966 (2026-07-22 22:07:03-22:15:10, `logs/eval_bb144_v7mixp_203966.out`),
loading `best.pt`@step 36000 (selection metric 0.18213), EMA weights, target-failures
200/point, writing `results/bb144_v7_mixp.json`. **Decision (user, 2026-07-22): do
not extend to a 4th 60k segment — iteration 7 is accepted as-is** (low-p `p_block`
trending to ~0, five-for-five wins over BP+OSD; see §"Verdict against acceptance
criteria" below).

## TL;DR

1. **iter-6 diagnosis (final, not re-litigated here): single-p=0.0055 training does
   not generalize.** `p_block` floored at ~0.40 across p=0.002-0.004, losing to
   BP+OSD by 157x at p=0.002 (`reports/iteration_6_status.md`,
   `results/bb144_mw_v4.json`). Root cause: the model only ever saw dense
   p=0.0055 syndromes in training.
2. **iter-7 fix (planned, `reports/iteration_7_plan.md`): mixed-p training.** Log-uniform
   p drawn per micro-batch over `[0.001, 0.0055]`, snapped to a 16-point log-spaced
   grid of cached stim samplers (TAG `v7_bb144_mixp`), from-scratch, same model/
   optimizer/EMA/reweighting as v6. `best.pt` selection changed to a
   generalization-weighted metric, `0.5*(p_block@p_max=0.0055 + p_block@p_min=0.001)`,
   confirmed live in the training log (`logs/cascade_bb_v7mixp_188708.out`:
   `[best.pt] step=1000 saved on metric: 0.5*(p_block@p_max[0.0055]=... + p_block@p_min[0.001]=...)`).
3. **The fix works: the waterfall is restored.** In the formal eval
   (`results/bb144_v7_mixp.json`) Cascade `p_block` falls monotonically as p
   decreases (0.35 → 0.206 → 0.042 → 0.0058 → 0.00038 for p=0.0055→0.002), and
   Cascade now **beats BP+OSD at all five swept points**, not just the training
   point. The in-training final full-sweep at p=0.001 (below the eval's swept
   range; same `best.pt`@36000/EMA checkpoint, 200000 shots, no BP+OSD baseline
   exists at this p) measured `p_block=0.00002` (3/200000,
   `logs/cascade_bb_v7mixp_188710.out`, quoted in
   `/home/u2467370/.claude/scratch/v7_completion_check.md` §1).
4. **Iter-7 plan acceptance criteria: 1-4 PASS, 5 PARTIAL** (`reports/iteration_7_plan.md`
   §7; a v7 zero-syndrome re-probe analogous to v6's job 185295 was not run —
   criterion 5 rests on std-positivity evidence only); see the itemized check below. No dead-head regression: all 12 per-logical std
   stayed in the 5.9-7.5 range through step 40000
   (`logs/cascade_bb_v7mixp_188710.out`).
5. **Legacy/caveats:** all iter-7 code, checkpoints and results are uncommitted
   (`git status` on 2026-07-22 shows `scripts/14_train_bb_v3.py`,
   `src/cascade/data/stim_dataset.py`, `slurm/train_bb_v6_mw.sh`,
   `slurm/eval_bb144_mw.sh` modified and `results/bb144_v7_mixp.json`,
   `reports/iteration_7_plan.md`, `slurm/{train,eval,smoke}_bb144_v7_mixp*.sh` etc.
   untracked) — pending user go-ahead to commit. `best.pt` is frozen at step
   36000 (no new best in the final 4000 steps once `lr_mult` hit its 0.10 floor);
   this is expected, not a fault (`/home/u2467370/.claude/scratch/v7_completion_check.md` §4).

## BP+OSD baseline (min-weight circuit) — unchanged, reused from iteration 6

Same files as `reports/iteration_6_status.md` (job 166180 + p=0.003 top-up job
167152, `osd_order=4`); re-read and independently re-verified here directly from
the JSONs (only the `"BP+OSD"` decoder entry is used — the `"Cascade"` entries in
these files are iter-6-era live probes, not iter-7 numbers):

| p | BP+OSD P_L/cycle [95% CI] | failures/shots | p_block | source |
|---|---|---|---|---|
| 0.0055 | 5.876e-3 [5.18e-3, 6.64e-3] | 280/500  | 0.560   | `results/bb144_mw_bposd_p0.0055.json` |
| 0.005  | 3.534e-3 [3.09e-3, 4.02e-3] | 236/600  | 0.393   | `results/bb144_mw_bposd_p0.005.json` |
| 0.004  | 1.260e-3 [1.06e-3, 1.49e-3] | 132/800  | 0.165   | `results/bb144_mw_bposd_p0.004.json` |
| 0.003  | 2.557e-4 [2.15e-4, 3.04e-4] | 130/3600 | 0.0361  | `results/bb144_mw_bposd_p0.003_merged.json` (merged; 47/1200 raw + 5 top-up workers) |
| 0.002  | 2.319e-5 [1.17e-5, 4.57e-5] | 8/2400   | 0.00333 | `results/bb144_mw_bposd_p0.002.json` |

Note (carried from iter-6): p=0.002 has only 8 BP+OSD failures, so its CI is wide
(~[1.17e-5, 4.57e-5], roughly [4x, 18x] depending on the inversion recipe against
Cascade's p=0.002 point below) — this was deferred rather than topped up in iter-6 because the
90x+ gap made the CI irrelevant to the verdict; iter-7 narrows that gap to
single digits, so the CI width is flagged here for transparency rather than
re-measured (no new BP+OSD shots were taken for iteration 7).

## Formal Cascade eval — FINAL (job 203966, 2026-07-22)

`slurm/eval_bb144_v7_mixp.sh` (target-failures 200, `--prefer auto` → `best.pt`@
step 36000, EMA weights, SHOTS_CAP 1e7), out `results/bb144_v7_mixp.json`.
Per-cycle P_L with Wilson 95% CI (k=12, rounds=12); **ratio column computed the
same way as iteration_6_status.md's table, i.e. BP+OSD P_L/cycle ÷ Cascade
P_L/cycle** (>1 means Cascade wins):

| p | Cascade P_L/cycle [95% CI] | fails/shots | p_block | ratio vs BP+OSD (P_L/cycle) |
|---|---|---|---|---|
| 0.0055 | 3.038e-3 [2.65e-3, 3.47e-3] | 224/640   | 0.350   | 1.9x |
| 0.005  | 1.615e-3 [1.41e-3, 1.85e-3] | 211/1024  | 0.206   | 2.2x |
| 0.004  | 3.001e-4 [2.61e-4, 3.45e-4] | 200/4736  | 0.0422  | 4.2x |
| 0.003  | 4.016e-5 [3.50e-5, 4.61e-5] | 200/34688 | 0.00577 | 6.4x |
| 0.002  | 2.635e-6 [2.29e-6, 3.03e-6] | 200/527232| 0.00038 | 8.8x |

**The waterfall is restored and Cascade now wins at every swept point** — a
qualitative reversal of iter-6's flat ~0.40 floor. All P_L, p_block and CI values
above were independently re-derived from `results/bb144_v7_mixp.json` and match
the `eval_bb144_v7mixp_203966.out` printout exactly.

**Cross-check flag (as requested):** the commander's initial back-of-envelope
ratios (0.002→8.8x, 0.003→6.3x, 0.004→3.9x, 0.005→1.9x, 0.0055→1.6x) were computed
from the **`p_block` ratio** (BP+OSD p_block ÷ Cascade p_block), not the
P_L/cycle ratio used in the table above and in iteration_6_status.md's headline
column. Both conventions were recomputed independently here and agree with each
other at low p (0.003: 6.3x vs 6.4x; 0.002: 8.8x vs 8.8x — p_block and P_L/cycle
coincide in the small-P_L linear regime) but **diverge by 15-20% at the higher-p
points**, where p_block is no longer ≈ 12×P_L: 0.0055 gives 1.6x by p_block vs
1.9x by P_L/cycle; 0.005 gives 1.9x vs 2.2x; 0.004 gives 3.9x vs 4.2x. Both
conventions agree on the qualitative verdict (Cascade wins at all five points,
by a widening margin toward low p); only the exact multiplier at the top three
points depends on which metric is quoted. p_block-based ratios, for reference:
1.6x, 1.9x, 3.9x, 6.3x, 8.8x (p=0.0055→0.002).

## Verdict against acceptance criteria (`reports/iteration_7_plan.md` §7)

1. **Waterfall shape returns (primary, must-pass): PASS.** `p_block` is
   monotonically non-increasing as p decreases (0.350→0.206→0.0422→0.00577→0.00038).
   Minimum bar `p_block(0.002) ≤ 0.5×p_block(0.0055)`: 0.00038 ≤ 0.175, easily met.
   Stretch bar `p_block(0.002) < 0.10`: 0.00038, far under the stretch target
   (iter-6's flat floor was 0.402).
2. **Low-p gap closes toward BP+OSD (secondary): PASS, exceeds target.** Target
   was within ~10x at p=0.002; measured 8.8x (both conventions) — inside target,
   and Cascade actually **wins** rather than merely narrowing the gap (iter-6 was
   >100x behind at this point).
3. **No regression at the winning point: PASS.** `p_block(0.0055)`=0.350, both
   under the ≤0.56 ceiling and a large improvement over v6's 0.512
   (`results/bb144_mw_v4.json`) — Cascade's margin over BP+OSD at p=0.0055 grew
   from ~1.1x (iter-6) to 1.6-1.9x (iter-7, depending on convention above).
4. **Live evidence (in-train): PASS.** The added low-p monitor (`p=0.001`) shows
   `ema p_block` collapsing from oscillating 0.06-0.66 mid-training (segment
   188708, steps 9000-18000) to pinned at 0.00000-0.00195 by the end (segment
   188710, steps 37000-40000) — `/home/u2467370/.claude/scratch/v7_completion_check.md`
   §2, independently confirmed by grepping `logs/cascade_bb_v7mixp_188710.out`
   directly (steps 37000/38000/39000/40000 all show `ema p_block=0.00000` except
   step 39000 at 0.00195).
5. **No dead-head / bias regression: PARTIAL (PASS on std-positivity; zero-syndrome
   re-probe not run).** All 12
   per-logical std values stayed strongly positive through the final segment
   (step 37000: 6.18-6.96; step 40000: 5.87-6.84;
   `logs/cascade_bb_v7mixp_188710.out`). A dedicated zero-syndrome re-probe
   (analogous to v6's dev job 185295) was not found in the available logs for
   v7 and is **not claimed here** — the std-positivity evidence is what is
   directly observed.

## Training status (v7, TAG `v7_bb144_mixp`) — COMPLETE @ 40k steps

Chain 188708 (TIMEOUT rollover, 47:59:00 wall) → 188709 (TIMEOUT rollover) →
188710 (COMPLETED), started 2026-07-17 02:44:10, ended 2026-07-21 13:26:09
(`logs/cascade_bb_v7mixp_188708/9/10.out`). Config (job header,
`logs/cascade_bb_v7mixp_188708.out`): Hidden 256, Blocks 12, Steps 40000, micro
batch 48, accum 69 (effective 3312), p range [0.001, 0.0055] log-uniform 16 grid
points, `ema_decay`=0.999, head reweight alpha=1.0/clamp=4.0/bce_ema=0.98 —
identical to v6 except the p recipe, as planned. Mixed-p sampling switched on at
step 1600 (`anneal_end`); resumed cleanly across both rollovers (`[resume] loaded
.../last.pt at step 36000 (best p_block=0.18213 @ 36000)`). Checkpoints:
`last.pt`@40000, `best.pt`@36000 (frozen there; no new best in the final 4000
steps once `lr_mult` hit its 0.10 floor — expected fine-tuning behavior, not a
fault). No NaNs, tracebacks, or OOM in any of the three `.out` logs; both non-empty
`.err` files are the expected SLURM TIMEOUT rollover message
(`/home/u2467370/.claude/scratch/v7_completion_check.md` §§3,6,7).

## Figures / table

Auto-generated from `results/bb144_v7_mixp.json` + the unchanged BP+OSD JSONs
(analogous to iter-6's `scripts/29_bb144_iter6_figures.py`):
`figures/bb144_v7_mixp_threshold.pdf`, `figures/bb144_v7_mixp_headline.pdf`,
`reports/bb144_iter7_table.{md,csv}`. These landed mid-drafting (mtime
2026-07-22 22:34, from a parallel session); `reports/bb144_iter7_table.md` was
read back and its `p_block`/`P_L`/fail-total numbers match this report's tables
exactly. **One intentional convention difference:** `bb144_iter7_table.md`'s
"ratio" column is `p_block`-based (8.8x/6.3x/3.9x/1.9x/1.6x, matching the
commander's back-of-envelope figures) and its own footnote explicitly flags
this as "distinct from the iter-6 table's P_L/cycle-based ratio" — i.e. the
companion table independently corroborates the same convention split raised in
the Cross-check flag above; this report's headline table uses the P_L/cycle
convention to stay consistent with `iteration_6_status.md`'s own column
definition. Neither is wrong; readers combining the two documents should note
which ratio they're quoting.

## Reproduction / fairness caveats

- BP+OSD baseline is byte-for-byte reused from iteration 6 (same min-weight
  circuit, same jobs 166180/167152) — no new BP+OSD shots were taken for
  iteration 7, so the comparison is exactly as fair/unfair as it was in
  iteration 6 (see the p=0.002 CI-width note above).
- Cascade v7 numbers come from `best.pt`@step 36000 (the generalization-weighted
  selection metric), not `last.pt`@40000 — the two differ slightly from the
  separate in-training full final-sweep at step 40000 (e.g. p=0.002:
  0.00042/200000 in-training vs 0.00038/527232 in the formal eval), consistent
  with different checkpoints and different (early-stopped vs full) shot counts,
  not a discrepancy requiring resolution.
- All iter-7 work (code diffs, new SLURM scripts, checkpoints, `results/bb144_v7_mixp.json`,
  this report) is uncommitted as of 2026-07-22 — nothing here has been committed
  to git, pending user approval.
- All runs went through SLURM (dev for the smoke, job 188692) — never the login
  node; this report itself was compiled read-only from the login node with no
  compute performed.
