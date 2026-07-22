# Iteration 7 plan — BB-[[144,12,12]] neural decoder: mixed-p retraining

Date: 2026-07-15. **PLAN — no training launched yet.** Author: iter-7 planning
session. Prerequisite diagnosis (iter-6) is final; this document turns the agreed
fix (mixed-p training) into an executable recipe. Every code-change anchor below
is a `path:line` verified against the current tree. **All `sbatch` in this plan is
ask-first — the user approves each submission; this planning pass submits nothing.**

Lineage: iter-6 shipped fix #1 (min-weight logical basis) which revived all 12
heads (`reports/iteration_6_status.md`, `reports/iteration_6_deadhead_rootcause.md`).
The resulting run `v6_bb144_mw` trains fine at the single noise p=0.0055 but does
not generalize to low p. Iter-7 changes exactly one thing: the training noise
recipe (single p → mixed p). The min-weight basis, model, optimizer, EMA,
per-head reweighting, and all hyper-parameters are held fixed so the mixed-p
effect is cleanly attributable.

---

## 1. Motivation

The v6_bb144_mw chain reached 40k steps and its final EMA eval
(`results/bb144_mw_v4.json`, `--prefer auto`, target-failures 200) is:

| p | Cascade p_block | BP+OSD p_block (iter-6 final) | Cascade vs BP+OSD |
|---|---|---|---|
| 0.0055 | 0.5117 (262/512) | 0.560  | ~1.1× **better** (only win) |
| 0.005  | 0.4824 (247/512) | 0.393  | worse |
| 0.004  | 0.4141 (212/512) | 0.165  | ~2.5× worse |
| 0.003  | 0.4023 (206/512) | 0.0361 | ~11× worse |
| 0.002  | 0.4023 (206/512) | 0.00333| **>100× worse** |

The Cascade curve is **flat at a ~0.40 floor** — it barely responds to p — while
BP+OSD falls three decades over the same range. The block-error floor is a
generalization failure, not a harness bug: the sampler passed a triple check
(`reports/iteration_6_status.md`), the dev probe (job 185295) showed **0/12 heads
fire on a zero syndrome** (so the floor is not spurious bias firing), and a direct
probe at p=0.0005 measured p_block ≈ 0.42 — the model mis-decodes ~40% of sparse,
genuinely-easy syndromes the instant it sees a noise level away from its single
training point. **Conclusion (final, not re-litigated here): single-p=0.0055
training over-specializes to the dense-syndrome regime and does not transfer to
the low-p syndromes that dominate the paper headline range.** The fix is to train
over a *range* of p so every noise level the eval probes is represented in the
gradient. A secondary, lower-priority issue rides along: weak heads 6/7/9/10/11
carry per-head BCE ≈ 0.178–0.283 vs ~0.078–0.083 for the other heads (in-train
eval at p=0.0055, step 40000, `logs/cascade_bb_169333.out`), the residue of the mean-pool readout
bottleneck (`src/cascade/models/cascade_bb.py:282`,
`reports/iteration_6_deadhead_rootcause.md:63-68`) — see §4.

---

## 2. p-distribution choice

**Recommended (primary): log-uniform over `[p_min, p_max] = [0.001, 0.0055]`,
drawn per micro-batch.**

- **p_max = 0.0055.** Keep the current single training point inside the range so
  the one regime Cascade already wins is not abandoned.
- **p_min = 0.001.** Aligns with (and extends just below) the eval floor. The
  headline sweep bottoms out at p=0.002; training down to 0.001 gives the model
  sub-eval-floor coverage with margin, without entering the extreme-sparse regime
  (see alternative B) where a batch is almost all trivially-decodable shots and
  the gradient signal per sample collapses.
- **Log-uniform, not uniform-in-p.** P_L falls roughly geometrically with p (the
  waterfall), so BP+OSD p_block spans 0.560 → 0.0033 over this range. Uniform-in-p
  sampling would pour most draws into the high-p end (dense failures, easy signal)
  and starve the low-p end — exactly the region the model currently fails.
  Log-uniform spreads draws evenly across the *decades of P_L*, putting comparable
  learning pressure on each order of magnitude and guaranteeing low-p failure
  gradient in essentially every optimizer step.
- **Per-micro-batch draw (not per-optimizer-step).** With accum=69, drawing a
  fresh p for each of the 69 micro-batches means every optimizer step's gradient
  is an average over the whole p range — no step-to-step p oscillation, and low-p
  and high-p signal co-occur in each update. This is strictly better for
  preventing low-p forgetting than putting one p on a whole 3312-sample effective
  batch. It is only affordable because of the sampler cache (see §3): each
  distinct grid p compiles its stim sampler once and is reused.
- **Implementation of "log-uniform" = snap to a log-spaced grid of ~16 points**
  spanning `[p_min, p_max]`. Truly continuous p would force a stim
  `compile_detector_sampler` on every draw (tens of thousands of compiles);
  snapping each log-uniform draw to the nearest of 16 cached grid points bounds
  total compiles to ≤16 while keeping coverage effectively continuous. `--p-grid-points`
  controls the grid density.

**Alternative A — discrete eval-aligned grid** `{0.001, 0.002, 0.003, 0.004,
0.005, 0.0055}`, drawn uniformly (or log-weighted). *Pro:* trains at exactly the
eval points, smallest cache (6 samplers), simplest to reason about. *Con:*
aliasing — the model only ever sees 6 noise levels and could over-fit the grid,
so an eval at an off-grid p (e.g. 0.0035) is no longer a genuine generalization
test. Use this only if the log-uniform pipeline proves fiddly in the smoke.

**Alternative B — p_min = 0.0005** (match the probe floor). *Pro:* full coverage
down to where the probe measured the failure. *Con:* at p=0.0005 the syndrome is
so sparse that almost every shot decodes trivially; failures are rare, so the
per-batch failure gradient is tiny and low-p learning stalls unless we add
importance up-weighting of low-p samples (extra complexity, extra variable).
**Deferred:** revisit for iter-8 with importance weighting only if p=0.001 low-p
generalization is proven but still short of target.

---

## 3. Exact code changes (anchored)

All changes are **backward-compatible**: with no new flag set, `--p-train` drives
the old single-p behaviour unchanged.

### 3a. `src/cascade/data/stim_dataset.py` — mixed-p sampling + sampler cache

- **`__init__` (lines 48-55):** add an optional `p_dist` argument (a small spec:
  `mode ∈ {log-uniform, uniform, list}`, `p_min`, `p_max`, `grid` = the log-spaced
  snap/cache points, optional per-point weights, and an RNG seed). Keep `p` for the
  single-p path. When `p_dist is None` the class behaves exactly as today.
- **Add a compiled-sampler cache.** Generalize `_make_sampler` (lines 65-67) into
  `_get_sampler(p)` that returns `self._cache[p]`, compiling+storing on a miss.
  Give each grid point its own derived seed (e.g. `base_seed + grid_index`) so the
  distinct-p streams are independent, not phase-locked.
- **`__iter__` (lines 69-81):** when `p_dist` is set, draw a p per yielded batch
  (`p_draw = self._draw_p()`, snapped to the nearest grid point), fetch its cached
  sampler, and `sampler.sample(...)` as at lines 76-78. The single-p branch
  (recompile only on `set_noise` change) is untouched.
- **Add `set_mixed(p_dist)`** (sibling to `set_noise`, lines 61-63) so the trainer
  can flip the dataset from single-p warmup into mixed mode at the stage boundary.

### 3b. `scripts/14_train_bb_v3.py` — CLI, dataset wiring, stage switch, eval + best selection

- **CLI (`parse_args`, add near lines 45-48):** new flags
  `--p-min` (float, default `None`), `--p-max` (float, default `None`),
  `--p-sampling {log-uniform,uniform,grid}` (default `log-uniform`),
  `--p-grid-points` (int, default 16), and an explicit-grid escape hatch
  `--p-list` (nargs="+", default `None`) with optional `--p-weights`. **Mixed mode
  is ON iff (`--p-min` and `--p-max`) or `--p-list` is given; otherwise the
  existing single-p `--p-train` (line 45) path runs verbatim** (backward compat).
  When mixed, `p_max` defaults the curriculum's `p2` and the in-train monitor
  point; `--p-train` still names the high-p monitor eval.
- **Dataset construction (lines 161-163):** build the `p_dist` spec from the new
  flags (log-space the grid over `[p_min, p_max]`) and pass it to
  `StimMemoryDataset`. The `CurriculumConfig` (line 161) is still built from
  `--p-warmup`/`p_max`; the dataset starts in **single-p warmup** mode.
- **Stage switch (loop head, lines 239-244):** keep the single-p curriculum for
  stages 1-2 (warmup at `p_warmup`, anneal to `p_max`) exactly as now
  (`curriculum.p_at`, `src/cascade/train/curriculum.py`). At `anneal_end`
  (stage-3 boundary) call `ds.set_mixed(p_dist)` **once**; thereafter stop calling
  `ds.set_noise` and let the dataset draw per micro-batch. The accum loop
  (`next(iter_ds)`, lines 261-262) needs no change — it just receives
  mixed-p batches.
- **Low-p monitor eval (eval block, lines 288-304):** the trainer currently evals
  only at `args.p_train`=0.0055 (live line 290-292, EMA line 297-299). **Add a
  second eval at a low p (`p_min`, or 0.002)** so the log shows the low-p p_block
  descending live — this is the key acceptance signal (§7) and is currently
  invisible during training.
- **best.pt selection criterion (lines 314-336):** today `best` is chosen by
  p_block at `p_train` only (lines 314-315), and the final sweep + external eval
  both load `best.pt` (line 366; `slurm/eval_bb144_mw.sh:46`). Under mixed-p this
  could enshrine a high-p-specialized checkpoint. **Change the selection metric to
  reward generalization**, e.g. `0.5*(p_block@p_max + p_block@p_min)` or the low-p
  p_block, using the new monitor eval. This is a required iter-7 change, not
  optional — otherwise the headline numbers come from the wrong checkpoint.
- **Final sweep (lines 374-386):** already iterates `args.p_eval`; drive it from
  the SLURM script with `--p-eval 0.001 0.002 0.003 0.004 0.005 0.0055` (default
  is lines 47-48). No code change needed.

### 3c. New SLURM launcher `slurm/train_bb_v7_mixp.sh`

Copy `slurm/train_bb_v6_mw.sh` and change only: the `144` case config
(lines 56-67) keeps `HIDDEN=256 BLOCKS=12 STEPS=40000 BATCH=48 ACCUM_STEPS=69
ROUNDS=12 EMA_DECAY=0.999`, sets a **new TAG** (§5), and the python invocation
(lines 97-116) **drops `--p-train`'s single-p role** by adding
`--p-min 0.001 --p-max 0.0055 --p-sampling log-uniform --p-grid-points 16` and
`--p-eval 0.001 0.002 0.003 0.004 0.005 0.0055`. Keep the reweighting knobs
(lines 75-77: `RW_ALPHA=1.0 RW_CLAMP=4.0 RW_EMA=0.98`) and the account/partition
header (lines 25-36). Do **not** edit `train_bb_v6_mw.sh` in place.

### 3d. Eval

`scripts/20_eval_decoder.py` needs **no change** — it already sweeps `--p`
(line 55) and prefers EMA (`--prefer auto`, lines 53/102-110). Copy
`slurm/eval_bb144_mw.sh` to `slurm/eval_bb144_v7_mixp.sh`, repoint `CKPT`
(line 46) to the new tag dir, and write `--out results/bb144_v7_mixp.json`
(line 75). The BP+OSD baseline is unchanged and reused from iter-6
(`results/bb144_mw_bposd_p*.json`) — same min-weight circuit, so the comparison
stays fair.

---

## 4. Weak-head problem — handle in iter-7 or defer?

**Recommendation: do NOT add new per-head machinery in iter-7. Keep the existing
adaptive reweighting at `alpha=1.0` (unchanged from v6) and only change the p
recipe.** Rationale:

- The dead-head structural bug is already fixed (min-weight basis); all 12 heads
  are alive (std 5–6.7, `logs/cascade_bb_165338.out`). What remains is a *soft*
  gap: heads 6/7/9/10/11 sit at BCE ≈ 0.22–0.31 vs ~0.12 for the rest — a residue
  of the mean-pool readout's difficulty with the (still weight-12 but higher-order)
  parities (`cascade_bb.py:282`).
- The adaptive per-head BCE reweighting (`head_weights`, `14_train_bb_v3.py:96-102`;
  EMA at line 281; knobs lines 62-70) is *already active* and already up-weights
  these heads automatically. It is the right, hands-off tool; there is no evidence
  a manual per-head intervention is needed before we've seen mixed-p's effect.
- **Isolate the variable.** Iter-7's whole purpose is to test whether mixed-p
  restores generalization. Simultaneously retuning per-head loss would confound
  attribution — the same single-variable discipline that made the iter-6 diagnosis
  clean. Low-p exposure may itself change the per-head signal balance.

**Watch, don't act:** the trainer already prints per-logical BCE, per-logical std,
and head weights each eval (lines 306-313). If, after iter-7, the low-p p_block is
demonstrably bottlenecked by heads 6/7/9/10/11 (their per-logical BCE dominates
the block error), escalate to iter-8 with a *readout* fix (rootcause fix #2 dense
per-qubit head, or fix #3 parity-capable readout;
`reports/iteration_6_deadhead_rootcause.md:120-161`) — not another reweighting
knob, which the rootcause analysis already argues is near-useless on a saturated
readout.

---

## 5. Training configuration

- **TAG: `v7_bb144_mixp`** (checkpoint dir `checkpoints/bb_144_12_12_v7_bb144_mixp/`).
  **Must never reuse `v6_bb144_mw`** — a fresh tag guarantees the from-scratch run
  cannot auto-resume the v6 `last.pt` (resume keys off the tag dir,
  `14_train_bb_v3.py:189,212`) and cannot overwrite the v6 checkpoints/results.
- **From-scratch vs warm-start: recommend FROM-SCRATCH for the definitive run.**
  - *For from-scratch:* (a) cleanest single-variable comparison against
    `v6_bb144_mw` — only the p recipe differs, matching this project's diagnosis
    methodology and giving the paper a defensible mixed-p-vs-single-p ablation;
    (b) no risk of the warm weights being locked in a p=0.0055-specialized basin
    that mixed-p then has to *unlearn*; (c) the standard cosine LR schedule
    (`14_train_bb_v3.py:246`, warmup 1000) applies as-is; (d) the dead-head fix is
    in the code, so from-scratch also gets 12 healthy heads.
  - *Against from-scratch:* full 40k cost (§6).
  - *Warm-start (from `checkpoints/bb_144_12_12_v6_bb144_mw/best.pt`)* would likely
    converge in fewer steps and cost less, **but** the current resume path only
    reloads a `last.pt` from the *same* tag dir and continues the old step/schedule
    (lines 212-227); a true warm-start needs a **new `--init-from CKPT` flag** that
    loads model (+EMA) weights while **resetting** optimizer state, `start_step`,
    and the LR/curriculum schedule. That is extra code and reintroduces the
    basin/interference risk. **Optional cost hedge:** run a short warm-start
    *preview* (~5–8k steps, mixed-p, from v6 best.pt) purely to get an early read
    on whether mixed-p moves the low-p p_block before committing the 40k
    from-scratch run. The headline run stays from-scratch.
- **Steps: 40,000** (match v6 for the controlled comparison). Mixed-p spreads the
  gradient budget across the range, and low-p failure signal is sparse, so low-p
  convergence *may* lag; keep the run **chain-extensible to 60k** via the existing
  `afterany` + auto-resume mechanism if the low-p monitor BCE is still descending
  at 40k. Decision rule at 40k: extend iff `p_block@p_min` is still trending down
  over the last ~5k steps.
- **SLURM (unchanged header from `train_bb_v6_mw.sh:25-36`):**
  `--account=GOV114009 --partition=8gpus --gres=gpu:1 --cpus-per-task=8
  --nodes=1 --ntasks-per-node=1 --time=47:59:00` (H200, 8gpus partition). Launch
  as a **3-segment `afterany` chain** (48k-steps-worth per ~48h segment, see §6),
  same pattern as v6 (`train_bb_v6_mw.sh:19-22`). The trainer's atomic `save_last`
  + auto-resume (lines 191-227, 336) already survive the segment rollovers.

---

## 6. Cost estimate

Measured v6 throughput on H200, micro=48 × accum=69 (effective ~3312): **~0.11
steps/s** observed (`logs/cascade_bb_165337.out`), plan with the conservative
**0.105 steps/s ≈ 9.5 s/step**.

- **40k from-scratch:**
  `40,000 steps × 9.5 s/step = 380,000 s = 105.6 h ≈ 4.40 days`.
  Cross-check: `40,000 / 0.105 = 380,952 s = 105.8 h`. **≈ 106 GPU-h** (1× H200).
- **Segments:** steps per 48h wall `= 0.105 × 48 × 3600 ≈ 18,144 steps`
  (matches v6's "~18k steps in segment 1", `iteration_6_status.md`).
  `40,000 / 18,144 = 2.2 → 3 segments` (`afterany` chain), leaving rollover slack.
- **If extended to 60k:** `60,000 × 9.5 = 570,000 s = 158.3 h ≈ 6.6 days ≈ 158
  GPU-h`; `60,000 / 18,144 = 3.3 → 4 segments`.
- **Eval (reused sizing from `eval_bb144_mw.sh:10-16`):** target-failures 200,
  cap 1e7, ~≤17 h wall, one 24h `8gpus` job. BP+OSD baseline reused (no new cost).
- **Smoke (§9):** dev partition, ~400–600 steps at reduced size, ≪1 GPU-h,
  inside a 1h dev window.

Mixed-p adds only the one-time compile of ≤16 cached stim samplers (seconds,
amortized over 40k steps) — negligible vs the single-p baseline throughput.

---

## 7. Acceptance criteria / success thresholds

Honest staging: **iter-7 step 1 is to prove generalization is restored — a
descending waterfall — not to beat BP+OSD everywhere.** Checks, all observable:

1. **Waterfall shape returns (primary, must-pass).** In `results/bb144_v7_mixp.json`
   the Cascade p_block is **monotonically non-increasing as p decreases** across
   {0.0055, 0.005, 0.004, 0.003, 0.002}, reversing the current flat ~0.40 floor.
   Minimum bar: `p_block(0.002) ≤ 0.5 × p_block(0.0055)`. Stretch:
   `p_block(0.002) < 0.10` (an order below the current 0.402).
2. **Low-p gap closes toward BP+OSD (secondary).** At p=0.002 the current gap is
   **>100×** worse than BP+OSD (0.402 vs 0.00333 p_block). Target: within **~10×**
   at p=0.002. Beating BP+OSD across the range is explicitly an iter-8+ goal, not
   an iter-7 gate.
3. **No regression at the winning point.** `p_block(0.0055)` stays ≤ ~0.56 (i.e.
   Cascade still ≈ ties/beats BP+OSD's 0.560 at p=0.0055); a >~10% regression vs
   v6's 0.512 triggers the R1 mitigation.
4. **Live evidence (in-train).** The added low-p monitor eval (§3b) shows
   `p_block@p_min` (or @0.002) **descending over training**, not pinned near 0.40.
   Check: `grep "low-p" logs/cascade_bb_<jobid>.out`.
5. **No dead-head / bias regression.** All 12 per-logical std > 0 throughout
   (lines 310-311 print), and the zero-syndrome probe still fires 0/12 heads
   (re-run the dev probe post-training) — confirms the floor is genuinely gone,
   not masked.

---

## 8. Risks and mitigations

- **R1 — mixed-p degrades the p=0.0055 win.** Spreading the gradient budget could
  cost the one point Cascade currently leads. *Mitigate:* p_max stays 0.0055 in
  the range (still trained every step via the cache); best.pt criterion includes
  p_max (§3b); monitor `p_block@0.0055` live; if it regresses >~10% vs v6, raise
  the high-p tail weight (`--p-weights`) or extend steps.
- **R2 — sparse low-p failure signal → slow low-p learning.** At p=0.001–0.002
  most shots decode trivially, so per-batch failure gradient is small and the
  model may just relearn the high-p regime. *Mitigate:* log-uniform (not uniform)
  already up-weights low p; per-micro-batch draws put low-p in every step;
  p_min=0.001 (not 0.0005) avoids the extreme-sparse regime; be ready to extend to
  60k; importance up-weighting of low-p loss held in reserve for iter-8.
- **R3 — sampler-cache correctness / cost.** A cache miss must compile lazily and
  store (else 40k× recompile destroys throughput); distinct-p streams must use
  distinct seeds or they phase-lock. *Mitigate:* smoke asserts cache size ≤
  `p-grid-points` and logs per-batch p variety; per-grid-point seed derivation
  (§3a).
- **R4 — best.pt selects a non-generalizer.** Choosing best on p_train alone would
  save a high-p-tuned checkpoint that the final low-p sweep then reports.
  *Mitigate:* the §3b selection-metric change (generalization-weighted).
- **R5 — weak-head bottleneck at low p.** Heads 6/7/9/10/11's mean-pool weakness
  could dominate low-p block error, capping the win. *Mitigate:* keep adaptive
  reweighting on; monitor per-head BCE; if these heads gate low-p p_block, escalate
  to the iter-8 readout fix (§4) rather than a reweighting knob.
- **R6 — schedule/warm-start pitfalls.** From-scratch sidesteps this (standard
  cosine applies). If warm-start is ever chosen, the `--init-from` flag MUST reset
  optimizer state, `start_step`, and the schedule — otherwise the resume path
  (lines 212-227) silently continues v6's finished schedule at LR≈0.

---

## 9. Smoke-test plan (before the 40k run)

**Goal:** prove the mixed-p pipeline runs end-to-end on dev before spending ~106
GPU-h. Model no dead-head recovery is expected in the dev window — that is fine;
this validates plumbing, not convergence.

- **Job:** copy `slurm/smoke_bb_v3_mw.sh` → `slurm/smoke_bb_v7_mixp.sh`,
  `--partition=dev --time=01:00:00`, reduced size (`--hidden 128 --blocks 6
  --steps 400 --accum-steps 4`, as the existing smoke), `--no-resume`, a separate
  `.smoke_ckpt` out dir, `--eval-every 50`, and the mixed-p flags
  `--p-min 0.001 --p-max 0.0055 --p-sampling log-uniform --p-grid-points 16
  --p-eval 0.001 0.0055`.
- **FIX THE WORKDIR BUG:** `slurm/smoke_bb_v3_mw.sh:25` (and the `--out` at line 65)
  point at `/work/leo07010/Ray/QEC/cascade` — the **old, pre-handoff account**. The
  iter-7 smoke must use `/work/u2467370/QEC/cascade`, or it will fail / touch
  another account (an ask-first cross-account action). Correct this when copying.
- **Pass checks (read from the smoke log):**
  1. runs 400 steps to completion, no crash on the new basis + mixed-p path;
  2. the per-batch p **varies across the log-spaced grid** (add a debug print of
     drawn p) and the sampler cache holds ≤ `p-grid-points` entries;
  3. the eval block prints p_block at **both** p_max=0.0055 **and** p_min=0.001
     (the new low-p monitor, §3b);
  4. all 12 per-logical std > 0 (no dead-head regression);
  5. best.pt is written using the new generalization-weighted metric.
- **Gate:** only after the smoke passes do we launch the 40k from-scratch chain.
  **Every `sbatch` — smoke, training chain, and eval — is ask-first and submitted
  only on the user's explicit approval. This plan submits nothing.**

---

## Appendix — anchor index (verified `path:line`)

- Single-p today: `scripts/14_train_bb_v3.py:45` (`--p-train` 0.0055),
  `:46` (`--p-warmup`), `:47-48` (`--p-eval` default), `:161` (CurriculumConfig),
  `:162-163` (dataset built at one p), `:239-244` (per-step curriculum p / set_noise),
  `:261-262` (accum-loop `next(iter_ds)`), `:288-304` (eval at p_train only),
  `:314-336` (best.pt select on p_train + save_last), `:366-386` (final loads best.pt, sweep).
- Curriculum stages: `src/cascade/train/curriculum.py` (`p_at`, warmup/anneal ends).
- Data pipeline: `src/cascade/data/stim_dataset.py:48-55` (`__init__`),
  `:57-63` (`p` / `set_noise`), `:65-67` (`_make_sampler` uses `self._p`),
  `:69-81` (`__iter__`; sample at `:76-78`).
- Readout bottleneck (weak heads): `src/cascade/models/cascade_bb.py:282` (mean-pool);
  `reports/iteration_6_deadhead_rootcause.md:63-68,120-161`.
- Reweighting (leave at alpha=1.0): `scripts/14_train_bb_v3.py:62-70,96-102,281`.
- SLURM to copy: `slurm/train_bb_v6_mw.sh:19-22,25-36,56-67,75-77,97-116`;
  `slurm/eval_bb144_mw.sh:46,65-75`; `slurm/smoke_bb_v3_mw.sh:25,46-66` (WORKDIR bug at :25).
- Results / evidence: `results/bb144_mw_v4.json` (flat floor), iter-6 BP+OSD in
  `reports/iteration_6_status.md:47-53`, throughput `logs/cascade_bb_165337.out`.
