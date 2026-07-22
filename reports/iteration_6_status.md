# Iteration 6 status — BB-[[144,12,12]] neural decoder (min-weight basis)

Date: 2026-07-07, finalized 2026-07-15. **FINAL — iteration 6 closed.**
Training reached 40k steps on 2026-07-10 (job 169333 resumed the v4 chain
165337→165338→165339 after the leo07010 → u2467370 account migration). The
formal eval (`slurm/eval_bb144_mw.sh`, job 185232) ran 2026-07-15; results
below. **Headline: the model is competitive only at the training p=0.0055 and
fails to generalize to lower p** (see "Final Cascade eval"). Iteration 7
(mixed-p retraining) is planned in `iteration_7_plan.md`. Root-cause detail
for the dead-head fix: `iteration_6_deadhead_rootcause.md`.

## TL;DR

1. **The iter-2/iter-6-v2 "dead head" failure was a code-geometry bug, not a
   training-recipe problem.** On BB-144, logical heads 8-11 (and marginally 6)
   collapsed to logit std = 0.000, BCE = ln2 = 0.693 exactly — an all-zero-logit
   predictor — by step ~3000-4000, and no amount of extra steps, bigger model,
   or per-head loss reweighting revived them (v2 and v3 both killed, ~2.4 GPU-days
   spent between them).
2. **Root cause: high-weight logical representatives × a mean-pool readout.**
   `bb.py._logical_basis` chose logical coset representatives by plain Gaussian
   elimination, giving BB-144 `lz` weights `[12,12,12,12,12,12, 24,12,36,34,28,38]`.
   The dead set was *exactly* the weight->12 rows `{6,8,9,10,11}`. The per-head
   readout (`cascade_bb.py:282`) is a mean-pool over the logical support, which
   cannot represent a wide XOR parity — gradient ≈ 0 at the saturated optimum, so
   weight decay zeroes the head. (The iter-2 "labels are 50/50" reading was
   REFUTED: at train `p` all heads have marginal flip ≈ 0.5, healthy ones
   included — representative weight is the discriminator.)
3. **Fix #1 (deployed): a minimum-weight logical basis.**
   `_min_weight_logical_basis` in `bb.py` (randomized information-set search,
   deterministic seed 20260705) makes every BB-144 `lz` AND `lx` representative
   weight = 12 = d (BB-72: all weight 6). Verified by probe 165329
   (`logs/lw_basis_verify_165329.out`: kernel membership, symplectic rank 12,
   independence mod stabilizers all OK). The old basis is kept as
   `_logical_basis_gaussian`; backup `src/cascade/codes/bb.py.backup-20260705`.
4. **The fix works.** At the exact steps where v2/v3 died (3000/4000), the v4 run
   has all 12 heads with logit std 2.2-4.0 and rising, uniform; reweighting is
   barely intervening (head weights ~0.85-1.26). The min-weight basis also speeds
   up the *whole* model (live `p_block` 0.73-0.78 @ step 3-4k vs v2/v3's ~0.97).
   As of step 18000 (Jul 7): live `p_block` 0.558, BCE 0.196, all 12 heads std
   4.4-6.1, still improving. No kill watch needed.
5. **Final verdict (40k eval, job 185232): the min-weight-basis model trains,
   but does NOT generalize below the training p.** `p_block` floors at ~0.40
   across p=0.002-0.004 (P_L/cycle ~3.6e-3), losing to BP+OSD by 157× at
   p=0.002 and winning only at the training point p=0.0055 (1.1×). Probe job
   185295 pinned the failure mode: heads stay correctly silent on the zero
   syndrome (0/12 fire) but misdecode sparse real syndromes (p_block 0.42 even
   at p=0.0005). Root cause is the single-p training recipe, not the harness
   (sampler triple-checked). Fix → iteration 7 mixed-p retraining.

## BP+OSD baseline (min-weight circuit) — FINAL

From `slurm/eval_bb144_mw_bposd.sh` (job 166180) + p=0.003 top-up (job 167152),
`osd_order=4`. Per-cycle P_L with Wilson 95% CI (k=12, rounds=12):

| p | BP+OSD P_L/cycle [95% CI] | failures/shots | p_block |
|---|---|---|---|
| 0.0055 | 5.876e-3 [5.18e-3, 6.64e-3] | 280/500  | 0.560   |
| 0.005  | 3.534e-3 [3.09e-3, 4.02e-3] | 236/600  | 0.393   |
| 0.004  | 1.260e-3 [1.06e-3, 1.49e-3] | 132/800  | 0.165   |
| 0.003  | 2.557e-4 [2.15e-4, 3.04e-4] | 130/3600 | 0.0361  |
| 0.002  | 2.319e-5 [1.17e-5, 4.57e-5] | 8/2400   | 0.00333 |

JSONs: `results/bb144_mw_bposd_p{P}.json`; p=0.003 is the merged file
`bb144_mw_bposd_p0.003_merged.json` (original 47/1200 + 5 top-up workers,
merged with `scripts/merge_bposd_points.py`, CI span 1.77×→1.41×).

Notes:
- The `"Cascade"` entries inside those `*_bposd_*` JSONs are **mid-training live
  probes**, NOT paper numbers — ignore them. The paper Cascade row comes from the
  40k eval below.
- p=0.002 has only 8 BP+OSD failures (CI spans ~4×). Topping it up (~+27k shots
  / ~20 GPU-h for ≥100 failures) is **deferred** until the 40k Cascade eval shows
  whether p=0.002 is the decisive comparison point. → **Resolved 2026-07-15: not
  needed for iteration 6.** Cascade's P_L at p=0.002 sits ~80× above even the
  upper end of BP+OSD's CI, so the CI width cannot change the verdict.

## Final Cascade eval — FINAL (job 185232, 2026-07-15)

`slurm/eval_bb144_mw.sh` (target-failures 200, min-errors 100, `--prefer auto`
→ picked best.pt @ step 38000, EMA weights, ≥5k EMA warmup satisfied;
SHOTS_CAP 1e7, `--no-mwpm`), out `results/bb144_mw_v4.json`. Per-cycle P_L
with Wilson 95% CI (k=12, rounds=12); ratio = BP+OSD P_L / Cascade P_L
(>1 means Cascade wins):

| p | Cascade P_L/cycle [95% CI] | fails/shots | p_block | ratio vs BP+OSD |
|---|---|---|---|---|
| 0.0055 | 5.110e-3 [4.49e-3, 5.79e-3] | 262/512 | 0.512 | 1.1x |
| 0.005  | 4.684e-3 [4.10e-3, 5.32e-3] | 247/512 | 0.482 | 0.75x |
| 0.004  | 3.784e-3 [3.29e-3, 4.34e-3] | 212/512 | 0.414 | 0.33x |
| 0.003  | 3.641e-3 [3.16e-3, 4.18e-3] | 206/512 | 0.402 | 0.07x |
| 0.002  | 3.641e-3 [3.16e-3, 4.18e-3] | 206/512 | 0.402 | 0.0064x |

**The model does not generalize below the training p.** Cascade's P_L is
nearly flat (~3.6e-3 floor) while BP+OSD improves ~250× over the same range:
Cascade wins only at the training point (1.1× at p=0.0055) and loses 157× at
p=0.002.

Diagnosis (final, with observed evidence — dev probe job 185295, 2026-07-15):

- **Not a harness bug.** Each p point gets a fresh circuit + unseeded sampler
  (`src/cascade/eval/decoder_compare.py:66-69`); measured detector click rates
  scale correctly with p (0.0715 @ p=0.002 vs 0.1027 @ p=0.003); two fresh
  samplers on the same circuit draw different shots; BP+OSD runs through the
  same `make_circuit(p)` path and responds strongly to p. The bitwise-identical
  206/512 failure sets at p=0.002 vs 0.003 are a coincidence of a p-independent
  failure floor plus the same 4-batch stopping point.
- **Failure mode: misreading sparse syndromes, not a trigger-happy readout.**
  On the all-zero syndrome 0/12 heads fire (healthy heads sigmoid ~1e-4; weak
  heads 6/7/9/10/11 elevated at 0.02-0.22 but all <0.5). At p=0.0005 — where a
  typical shot still carries ~30+ detector clicks — measured p_block is 0.42.
- **Root cause: single-p training** (all training data at p=0.0055). Secondary,
  additive: weak heads 6/7/9/10/11 with per-head BCE 0.18-0.28 vs ~0.08 for the
  rest (best.pt history). Fix plan: mixed-p training → `iteration_7_plan.md`.

Figures + headline table auto-generate from `results/bb144_mw_v4.json` +
the BP+OSD JSONs via `scripts/29_bb144_iter6_figures.py`
(`figures/bb144_mw_threshold.pdf`, `figures/bb144_mw_headline.pdf`,
`reports/bb144_iter6_table.{md,csv}`).

## Training status (v4, TAG `v6_bb144_mw`) — COMPLETE @ 40k steps

Old account (leo07010): chain 165337 → 165338 → 165339 (3×48h `afterany`,
8gpus) delivered ~24k steps; rollovers verified clean (`[resume] loaded … at
step 18000`, 12/12 heads alive). After the account migration (authoritative
copy now `/work/u2467370/QEC/cascade`), job 169333 resumed from step 24000 and
reached 40000 on Jul 10 (1d18h, node 25a-hgpn013, H200, ~0.105 steps/s);
dependency job 169440 was the expected no-op resume that re-ran the `[final]`
multi-p sweep (numbers consistent within CI). Checkpoints: `last.pt` @ 40000,
`best.pt` @ 38000 (EMA p_block 0.478 at train p). Trainer
`scripts/14_train_bb_v3.py` (recipe unchanged from v3, incl. reweighting),
`_min_weight_logical_basis`.

## Reproduction / fairness caveats

- Per-head numbers are NOT comparable to v2/v3 runs (GF(2) basis change); the
  block-level `p_block` / P_L headline is basis-invariant, so the Cascade-vs-BP+OSD
  comparison is unaffected.
- The old Gaussian-basis BP+OSD baseline (BB-72 era) is superseded by job 166180's
  min-weight-circuit numbers above.
- All runs go through SLURM (dev for smokes) — never the login node.
