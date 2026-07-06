# Cascade Neural Decoder — Iteration 3 Status Report

Snapshot date: 2026-05-05
Working dir: `/home/leo07010/Ray/QEC/cascade/`
Reproduces: Andi Gu et al., *Cascade: A neural decoder for ...* (arXiv:2604.08358)
Plan: `~/.claude/plans/concurrent-zooming-papert.md` (Iter-3 P0: Paper-Aligned BB-72 Retrain)

---

## 1. Executive summary

**Both success-bar criteria met.** Hits decision-tree row "✅ DONE / Package demo".

| Criterion | Iter-2 | Iter-3 | Verdict |
|---|---|---|---|
| Logicals alive (std > 0.1) | 7/12 | **12/12** | ✅ |
| Cascade per-cycle P_L vs BP+OSD @ p=0.005 | **10× worse** (5.34e-2 vs 5.30e-3) | **4.1× better** (1.31e-3 vs 5.38e-3) | ✅ |
| Live block error @ p=0.005 | 0.966 | **0.0900** | 10.7× better |

Iter-3 is a pure training-recipe change (gradient accumulation to paper batch 3328 + curriculum re-enabled). Architecture and head untouched from iter-2.

---

## 2. Training run (job 186261)

```
Job ID         : 186261
Node           : hgpn02 → preempted, requeued, scancelled
Code           : BB-72 (rounds=6)
Hidden         : 256, Blocks: 8, Steps: 40000 (target)
Micro batch    : 256, Accum: 13, Effective: 3328  (= paper batch)
p_train        : 0.0055 (warmup 0.001)            (= paper 3-stage curriculum)
Tag            : v3_bb72
Params         : 3,564,044
Start          : 2026-05-04 19:32:06
End            : preempted at 2026-05-05 ~13:35; scancelled at 14:35
Wall completed : 18h+ on H100, reached step ~38,800 / 40,000 (97%)
```

### 2.1 Why the run is still usable

`scripts/14_train_bb_v2.py` saves `best.pt` whenever EMA p_block improves. The most recent improvement landed at step ~38,000 (file mtime May 5 13:35). The training trajectory shows convergence well before then — live p_block plateaued at 0.131–0.140 from step 32,000 onward, so the missing 1,200 steps are cosine-LR fine-tune (lr_mult ≤ 0.13) with diminishing returns.

slurm preempted the job (`Restarts=1`, `Reason=Priority`). The train script has no resume logic (it loads `best.pt` only at the final-evaluation step, line 267, not at startup), so a restart would have begun from step 1. Slurm's revised `StartTime` was 2026-05-06 21:23 (≥ 31h further wait + ~24h re-train), with a non-zero chance of a second preemption. The user chose to `scancel 186261` and proceed with the available `best.pt`.

### 2.2 Training trajectory (selected eval points)

| step | live p_block | EMA p_block | live BCE | per-logical std (k=8–11) |
|---|---|---|---|---|
| 1,000 | 0.922 | cold | 0.553 | 1.63, 1.58, 1.21, 1.63 |
| 5,000 | 0.323 | cold | 0.196 | (early-warning passed) |
| 10,000 | 0.219 | 0.992 | 0.150 | all > 4.5 |
| 20,000 | 0.171 | 0.998 | 0.131 | all > 5.5 |
| 30,000 | 0.145 | 0.706 | 0.114 | 6.4, 6.3, 6.4, 6.2 |
| 35,000 | 0.131 | 0.171 | 0.113 | 6.5, 6.5, 6.5, 6.4 |
| 38,000 | 0.132 | 0.142 | 0.107 | 6.6, 6.5, 6.4, 6.5 |

**Risk R1 (multi-task interference) did not trigger.** All 12 heads had std > 1 by step 1,000 and grew monotonically. The bigger effective batch + curriculum fix from iter-2 (where 5 heads collapsed to identically-zero output by step 31k) was clean.

EMA was cold for ~25k steps because of β=0.9998 and the live model rapidly improving early; once live converged, EMA caught up (0.998 → 0.142 over steps 25k–38k).

---

## 3. Per-logical diagnostic (step 3 of plan)

`scripts/24_bb_per_logical.py --ckpt checkpoints/bb_72_12_6_v3_bb72/best.pt --p 0.005 --shots 8192` (EMA weights):

```
block error rate = 0.0878 (719/8192)
BCE (mean over K logicals): 0.0771
```

| k | logit std | mean P(1) | obs rate | per-logical err | verdict |
|---|---|---|---|---|---|
| 0 | 8.61 | 0.459 | 0.462 | 0.028 | alive |
| 1 | 8.61 | 0.464 | 0.462 | 0.030 | alive |
| 2 | 8.60 | 0.459 | 0.464 | 0.030 | alive |
| 3 | 8.62 | 0.457 | 0.456 | 0.030 | alive |
| 4 | 8.63 | 0.460 | 0.463 | 0.029 | alive |
| 5 | 8.65 | 0.456 | 0.454 | 0.028 | alive |
| 6 | 7.41 | 0.503 | 0.501 | 0.036 | alive |
| 7 | 7.71 | 0.466 | 0.465 | 0.028 | alive |
| 8 | 6.59 | 0.497 | 0.500 | 0.037 | alive |
| 9 | 6.95 | 0.500 | 0.500 | 0.039 | alive |
| 10 | 6.50 | 0.494 | 0.500 | 0.042 | alive |
| 11 | 6.13 | 0.505 | 0.511 | 0.044 | alive |

**alive (std > 0.1): 12/12 logicals.** Light heads (k=0–5, weight 6) have std ~8.6 and per-logical error ~3%; heavy heads (k=8–11, weights 16–20) have std 6.1–7.0 and per-logical error 3.7–4.4%. The expected gradient of difficulty with logical weight is present but mild — heavy heads only ~50% worse per-logical than light heads, vs iter-2's identically-zero collapse.

Saved at `results/bb72_v3_per_logical_p0.005.txt`.

---

## 4. Cascade vs BP+OSD (step 4 of plan)

slurm job 188144 on hgpn05, 35 min wall (`slurm/eval_bb_bposd.sh`). Both decoders evaluated against the same EMA checkpoint at p=0.005; BP+OSD capped at 2000 shots (CPU-bound), Cascade at 200,000.

| decoder | shots | fail | p_block | P_L/cycle | CI95 | ratio vs Cascade |
|---|---|---|---|---|---|---|
| Cascade (EMA) | 200,000 | 18,006 | 0.0900 | **1.31e-03** | [1.29, 1.33]e-03 | 1.0× |
| BP+OSD (osd_order=4) | 2,000 | 637 | 0.3185 | 5.38e-03 | [4.97, 5.82]e-03 | 4.11× worse |

**Cascade beats BP+OSD by 4.1× in per-cycle P_L.** CI bounds do not overlap (Cascade upper 1.33e-3 vs BPOSD lower 4.97e-3 — 3.7× gap), so the verdict is statistically clean.

Block-error gap (0.090 vs 0.319) tracks the per-cycle gap — Cascade's 12-logical product `1 - Π(1 - p_k)` benefits from per-logical errors all in the 3–4% range (no 50% dead head pinning the product).

Saved at `results/bb72_v3_iter3_bposd.json`.

### 4.1 Comparison vs iter-2 numbers

| | iter-2 | iter-3 | improvement |
|---|---|---|---|
| Cascade P_L/cycle @ p=0.005 | 5.34e-2 | 1.31e-3 | **40.8×** |
| Cascade vs BPOSD ratio | 10.1× **worse** | 4.1× **better** | crossed parity |
| Live p_block @ p=0.005 | 0.966 | 0.0900 | 10.7× |

The 40× P_L improvement comes from two sources stacked: (i) all 12 logicals now decode (5 dead heads removed the multiplicative `0.5^5 = 3.1%` block-error floor), (ii) the surviving heads also decode better thanks to a richer backbone trained at the paper's batch size and curriculum.

---

## 5. Decision-tree verdict

From the plan:

| Per-logical | Cascade vs BPOSD | Verdict |
|---|---|---|
| 12/12 alive | Cascade ≤ BPOSD | ✅ DONE — Package demo |

**Iter-3 P0 hits the top row.** The recipe-only change (gradient accumulation to paper batch + curriculum) was sufficient; no architecture changes, no auxiliary losses, no warm-starts needed. This matches paper Methods §Training, which explicitly disclaims those techniques.

---

## 6. Artifacts

```
checkpoints/bb_72_12_6_v3_bb72/best.pt          (43 MB, EMA + live, step ~38,000)
results/bb72_v3_per_logical_p0.005.txt          (per-logical std/err table)
results/bb72_v3_iter3_bposd.json                (Cascade + BPOSD sweep at p=0.005)
logs/cascade_bb_186261.out                      (~22 KB, training log to step 38,800)
logs/eval_bb_bposd_188144.out                   (eval header + result rows)
reports/iteration_3_status.md                   (this file)
```

The iter-2 ckpt `checkpoints/bb_72_12_6_v2_bb72/best.pt` is preserved for comparison.

---

## 7. Open questions / follow-ups

1. **Did the missing 1,200 steps cost anything?** EMA p_block was 0.142 at step 38,000; trajectory suggests it would have settled near 0.10–0.12 at step 40,000 (extrapolating cosine-LR tail). A re-run from the existing `best.pt` for the last ~5% of steps could squeeze ~10–20% more P_L if needed for paper-faithfulness, but the demo verdict doesn't change.
2. **Λ for BB-72.** Plan focused on a single point at p=0.005. A multi-p sweep (matching iter-2's `P_EVAL=(0.001 0.002 0.003 0.004 0.005)`) would give a code-distance suppression curve. The Cascade evaluation cost is cheap (~minutes); BPOSD at lower p gets more shots-per-fail, so 2000-shot CIs widen.
3. **BB-144 [[144,12,12]] (Gross code).** The recipe transfer is the natural next step — paper claims Cascade beats BP+OSD by ~4000× on this code. Iter-2 parked it because of the 5/12-dead-head issue; iter-3 has cleared that blocker. The compute budget is roughly 4× iter-3 (n=144 vs 72, and rounds=12 vs 6); a single H100 should still fit.
4. **80k-step variant** (paper-faithful step count). Iter-3 used 40k for variable isolation; paper used 80k. If iter-3 P_L turned out to be the gating number for a write-up, the marginal Λ improvement of doubling steps is worth a single retrain. Not on the critical path right now.

---

## 8. Recommended next action

**Move to iter-4 = BB-144 [[144,12,12]]** with the same paper-aligned recipe. Required changes:

- `slurm/train_bb.sh` case `144`: enable `ACCUM_STEPS=13`, set `P_WARMUP=0.001` (currently both off, mirroring the iter-2 BB-72 settings that we now know to fix).
- Verify `_BBSpatialWrap` offset table for the 12×6 torus (already implemented; smoke test on login: a single-step forward should match the polynomial layout).
- Eval scripts already accept `--bb-variant 144`; only ckpt path needs updating.

If BB-144 reproduces, the full demo (surface d=5/7/9 + BB-72 + BB-144) can be packaged as the deliverable.
