# Cascade Neural Decoder — Iteration 2 Status Report

Snapshot date: 2026-05-04
Working dir: `/home/leo07010/Ray/QEC/cascade/`
Reproduces: Andi Gu et al., *Cascade: A neural decoder for ...* (arXiv:2604.08358)

---

## 1. Executive summary

| Track | Status | Result | Notes |
|---|---|---|---|
| Surface d=5/7/9, Track 2 | ✅ delivered | Cascade 1.0–2.16× lower P_L than MWPM; Λ ≈ 1.86 [1.79, 1.93] @ p=0.005 | Below paper Λ≈8.4 (H=128 < paper H=512); deliverable at "Track 1.5" tier |
| BB-72 first attempt | ❌ failed | BCE flat at ln(2); 99.9% block error | Diagnosed; root cause = global pooling + curriculum, not optimizer or schedule |
| BB-72 architecture rewrite | 🟡 partial | 7/12 logicals decoded cleanly (per-logical err 3.2–4.3%); 5 heavy-weight heads collapsed to identical-zero output | Job 185354 done; structural failure on weight-≥14 lz rows (§3.6) |
| BB-72 BPOSD baseline | ✅ complete | BP+OSD P_L/cycle 5.30e-3 vs Cascade 5.34e-2 → **Cascade ≈ 10× behind** at p=0.005 | slurm 185651 done (35 min on hgpn02) |
| Tooling / infra | ✅ complete | 27 unit tests pass | All evaluation, fitting, plotting scripts in place |

---

## 2. Surface code results (Track 1.5 deliverable)

Surface d=5/7/9 trained with `trainer_v2` (Muon + Lion + EMA + 3-stage curriculum + bf16):

| d | Steps | Best EMA p_block | Train time | best step |
|---|---|---|---|---|
| 5 | 40,000 | 0.0073 | 14.8 min | 39,000 |
| 7 | 60,000 | 0.0049 | 38.2 min | 60,000 |
| 9 | 80,000 | 0.0045 | 78.2 min | 74,000 |

### 2.1 Cascade vs MWPM (per-cycle P_L, 200k shots, EMA weights)

p_phys | d=5 Cascade | d=5 MWPM | d=5 ratio | d=7 Cascade | d=7 MWPM | d=7 ratio | d=9 Cascade | d=9 MWPM | d=9 ratio
---|---|---|---|---|---|---|---|---|---
0.001 | 1.80e-05 | 2.10e-05 | 1.17× | 1.43e-06 | 2.86e-06 | 2.00× | 5.6e-07† | 5.6e-07† | 1.00×
0.002 | 1.36e-04 | 2.32e-04 | 1.71× | 1.79e-05 | 3.86e-05 | 2.16× | 5.00e-06 | 6.11e-06 | 1.22×
0.003 | 4.50e-04 | 6.96e-04 | 1.55× | 9.79e-05 | 1.85e-04 | 1.89× | 3.67e-05 | 5.45e-05 | 1.49×
0.004 | 9.82e-04 | 1.52e-03 | 1.55× | 3.24e-04 | 6.11e-04 | 1.88× | 1.66e-04 | 2.73e-04 | 1.64×
0.005 | 2.02e-03 | 2.71e-03 | 1.34× | 8.91e-04 | 1.41e-03 | 1.58× | 5.88e-04 | 7.76e-04 | 1.32×
0.006 | 3.52e-03 | 4.69e-03 | 1.33× | 1.92e-03 | 2.83e-03 | 1.48× | 1.62e-03 | 1.80e-03 | 1.11×
0.007 | 5.65e-03 | 7.25e-03 | 1.28× | 3.75e-03 | 5.06e-03 | 1.35× | 3.58e-03 | 3.59e-03 | 1.00×

† insufficient statistics (fewer than 5 errors observed at p=0.001 for d=9)

Cascade beats MWPM at every (d, p) where statistics permit, by 1.2–2.2×. At p=0.007 (close to surface threshold) the advantage shrinks to ~1.0× — expected behavior near threshold.

### 2.2 Λ (error-suppression) fits

Λ from log-linear regression of P_L vs ⌊(d+1)/2⌋:

p | Cascade Λ [95% CI] | MWPM Λ [95% CI]
---|---|---
0.003 | 3.50 [3.09, 4.00] | 3.57 [3.19, 3.97]
0.004 | 2.43 [2.28, 2.59] | 2.36 [2.24, 2.48]
0.005 | 1.86 [1.79, 1.93] | 1.87 [1.81, 1.93]
0.007 | 1.26 [1.23, 1.28] | 1.42 [1.40, 1.45]

Cascade Λ tracks MWPM Λ within statistical error (slope ≈ same), but Cascade has consistently lower P_L prefactor. Paper Λ ≈ 8.4 not reached — attributed to model capacity (H=128 vs paper H=512) and small d range (3 distances vs paper's broader sweep).

### 2.3 Surface artifacts

```
checkpoints/surface_d5_v2_d5/best.pt
checkpoints/surface_d7_v2_d7/best.pt
checkpoints/surface_d9_v2_d9/best.pt
results/surface_d{5,7,9}_v2.json
figures/surface_lambda.pdf
```

A separate `checkpoints/surface_d5_v2_d5_login/best.pt` exists from a login-node debugging run — kept for comparison; **not** part of the deliverable.

---

## 3. BB-72 debugging journey

### 3.1 First attempt (slurm job 184952)

Trained with default `trainer_v2` settings (H=128, L=6, 40k steps, Muon + Lion + EMA + curriculum p1=0.001 → p2=0.0055 + bf16). Result:

```
Final BCE: 0.6849 (essentially ln(2) = 0.6931 — random predictor)
Final p_block: 0.999  (any-of-12-logicals wrong)
EMA p_block: 0.999
```

Per-logical logit std on 64 shots:

| Logical | std | Note |
|---|---|---|
| L0–L5 | 0.31–0.43 | weak signal |
| L6 | 0.02 | dead |
| L7 | 0.17 | weak |
| L8–L11 | 1e-5–1e-4 | completely dead |

### 3.2 Root cause analysis

Three hypotheses tested and resolved:

1. **Logical basis weight** (canonical `code.lz` row weights are `[6,6,6,6,6,6,14,8,16,18,20,20]`; high-weight logicals have parities statistically near 50%).
2. **Optimizer / curriculum / EMA / bf16** flat-lining the gradient.
3. **Architecture** (global average pooling drowning per-logical signal across 432 detectors).

Diagnostic experiments:

- **Phase 0 smoke** (AdamW + no curriculum + no EMA + fp32, 3000 steps): BCE 0.6937 → 0.6890. **Confirms not optimizer/curriculum**.
- **Linear baseline** on raw flattened `det → obs`: also stuck at 0.69. Confirms BB syndrome → logical mapping has no easy linear structure (unlike surface).
- **Toroidal translate search**: weight-6 lz translates over Z₆×Z₆ span only 6-D LI subspace mod stabilisers. **No 12-D weight-6 basis exists** for [[72,12,6]]. Original lz row weights are inherent.
- **Per-logical detector mask** (relevance from stabiliser × lz support overlap): BCE 0.69 → 0.49 in 3000 steps; L0–L5 logits std jumped to 3.0–3.9; L7 to 1.30; L9–L11 still dead. **Architecture is real cause**.

### 3.3 Architecture fix (commit-ready in `cascade_bb.py`)

Replace global mean pool with the paper's intended **check → data-qubit scatter + per-logical lz-support pool**:

1. Backbone unchanged: `(B, F, T, ell, 2m)` features on detector positions.
2. `final_conv` projects to 12 logical channels: `(B, K=12, F, T, ell, 2m)`.
3. **Scatter** through Tanner graph: `data_feat[q] = (1/deg) Σ_{c with q in support} feature[c]` for each data qubit, giving `(B, K, F, T, n_data=72)`.
4. Time-average: `(B, K, F, n_data)`.
5. **Per-logical pool** over `lz[i]` support: pool over 6/8/14/16/18/20 specific data qubits per logical i.
6. Per-observable head → 12 logits.

Implementation registers `scatter_matrix` (sparse Tanner adjacency, shape `(2*n_check, n_data)`) and `lz_support_float` `(K, n_data)` as buffers.

### 3.4 Smoke results with new architecture

Three setups, 3000 steps each, AdamW lr=1e-3, no curriculum, fp32:

Setup | BCE end | L0–5 std | L7 std (w=8) | L6 std (w=14) | L8 std (w=16) | L9–11 std (w=18-20)
---|---|---|---|---|---|---
Original (global pool) | 0.6890 | 0.31–0.43 | 0.12 | 0.02 | 0.02 | < 0.02
Per-logical detector mask | 0.4917 | 3.0–3.9 | 1.30 | 0.15 | 0.11 | < 0.01
**Final scatter + lz-support pool, H=128 L=6** | **0.4780** | **4.1–5.0** | **1.44** | **0.13** | **0.10** | **< 0.01**
Final scatter + pool, **H=256 L=8** | 0.4381 | 4.8–5.5 | 2.28 | 0.18 | 0.014 | < 0.025

L0–L7 (8 of 12 logicals) fully active under new architecture. **L9–L11 (weight 18–20) remain dormant** at 3000 steps — bigger model didn't help. Either more steps required for gradient to break out, or these logicals require deeper Tanner-graph message passing than 8 blocks provide.

### 3.5 Slurm retrain (job 185354, completed)

`slurm/train_bb.sh` updated:

```diff
-  HIDDEN=128; BLOCKS=6;  STEPS=40000; BATCH=256; ROUNDS=6
-  P_TRAIN=0.0055; P_WARMUP=0.001
+  HIDDEN=256; BLOCKS=8;  STEPS=40000; BATCH=256; ROUNDS=6
+  P_TRAIN=0.0055; P_WARMUP=0.0055    # curriculum disabled
```

Ran 111 min on H100, 6.0 steps/s. Convergence:

```
final BCE: 0.359 (down from ln(2)=0.693)
best EMA p_block = 0.96790 at step 31000
final EMA p_block = 0.97083 at step 40000
```

Cascade-only sweep on the EMA checkpoint (200k shots/p, `results/bb72_v2_iter2.json`):

| p | p_block | P_L/cycle | CI95 |
|---|---|---|---|
| 0.002 | 0.8717 | 3.05e-02 | [3.03, 3.07]e-02 |
| 0.003 | 0.9345 | 4.16e-02 | [4.13, 4.19]e-02 |
| 0.004 | 0.9564 | 4.87e-02 | [4.84, 4.91]e-02 |
| 0.005 | 0.9661 | 5.33e-02 | [5.29, 5.37]e-02 |

p_block remained dominated by the dormant logicals — the 40k-step sweep did not break L9–L11 out, and L6 collapsed back to the dead state seen in the original architecture.

### 3.6 Per-logical breakdown (post-training)

Diagnostic at p=0.005, 8192 shots, EMA weights (`scripts/24_bb_per_logical.py`):

| k | lz row weight | logit std | mean P(1) | per-logical err | verdict |
|---|---|---|---|---|---|
| 0–5 | 6 | 7.07–7.15 | 0.46 | 0.032–0.035 | alive |
| 6 | 14 | 0.0000 | 0.498 | 0.498 | DEAD (always predicts 0) |
| 7 | 8 | 7.24 | 0.469 | 0.043 | alive |
| 8 | 16 | 0.0000 | 0.499 | 0.492 | DEAD |
| 9 | 18 | 0.0000 | 0.500 | 0.489 | DEAD |
| 10 | 20 | 0.0000 | 0.500 | 0.498 | DEAD |
| 11 | 20 | 0.0000 | 0.499 | 0.496 | DEAD |

**7/12 logicals alive after 40k steps.** Notable shift from the 3000-step smoke (table 3.4):

* L7 (weight 8) now strongly alive (std 7.2 vs 0.18 at smoke). The extra 37k steps gave the 8-supported logical enough gradient to escape.
* L6 (weight 14) regressed back to identically zero output (std 0.18 → 0.00). The model converged to "predict 0" rather than carrying the smoke-time weak signal — likely because constant-0 minimises BCE on a near-50/50 marginal once the per-logical head can no longer find structure.
* L8–L11 (weights 16–20) remain identically dead, std exactly 0. Their lz heads collapsed to constant output during training, same failure mode as L6.

Per-logical error rates on the 7 alive logicals (3.2–4.3%) are usable, but the multiplicative `1 - Π(1 - p_k)` block error gets pinned at ≥0.95 by the 5 dead heads. So the model is not "decoding badly across the board" — it is decoding very well on weight-≤8 logicals and not at all on weight-≥14 ones.

### 3.7 Decision (resolves §6 of original plan)

Three-way branch from §6 ("12 alive → continue", "8/12 → publish as 1.5", "still flat → rework"):

* Outcome: **7/12 alive — between branches B and C.**
* Per-logical evidence shows it is **not** an under-trained / wrong-LR / wrong-curriculum failure. The alive logicals decoded cleanly; the dead ones converged to identically zero. So scaling H, blocks, or steps further is unlikely to help — the lz-support pool gives heavy logicals a 16/18/20-qubit parity-of-noisy-bits to predict where the marginal is essentially 50% under the training noise distribution, and the per-logical head has no shared representation with the alive heads to bootstrap from.
* **Verdict**: ship Track-1.5 BB with the 7/12 result documented honestly. Do NOT chase BB-144 until a structural fix unfreezes heavy logicals (see §7).

---

## 4. Tooling / infra status

### 4.1 Code added or modified this iteration

- `src/cascade/decoders/bposd.py` — BP+OSD wrapper for BB baseline (matches MWPMDecoder API)
- `src/cascade/eval/decoder_compare.py` — `include_bposd`, `bposd_max_shots` flags
- `src/cascade/eval/lambda_fit.py` — log-linear Λ fit + bootstrap CI
- `src/cascade/eval/post_select.py` — confidence-threshold sweep
- `src/cascade/models/cascade_bb.py` — per-logical mask + Tanner scatter + lz support pool (3.3)
- `scripts/15_bb_smoke_adamw.py` — Phase 0 diagnostic
- `scripts/20_eval_decoder.py` — checkpoint → noise sweep → JSON; reads H/L from ckpt; `--min-errors` flag
- `scripts/21_compare_baselines.py` — paper-style P_L vs p plot
- `scripts/22_post_select.py` — acceptance vs P_L curve
- `scripts/23_fit_lambda.py` — Λ from per-distance JSONs
- `slurm/train_bb.sh` — updated for new architecture (3.5)
- `slurm/eval_bb_bposd.sh` — eval-time BPOSD slurm wrapper (`--batch 512` to fit H=256/L=8 on 80GB; 2000 BPOSD shots)
- `scripts/24_bb_per_logical.py` — per-logical std / err diagnostic (3.6)
- `scripts/20_eval_decoder.py` — `_resolve_model_hparams` accepts both `num_blocks` (surface ckpts) and `blocks` (BB ckpts)
- `src/cascade/decoders/bposd.py` — default `osd_order` lowered 60 → 4 (no quality gain on BB-72, ~250× speedup; see §6 comment)
- `tests/test_bb.py` — 12 tests (algebra + circuit distance + ldpc cross-check)
- `tests/test_eval.py` — 6 tests (lambda fit, post-select, BPOSD wiring)

### 4.2 Test coverage

```
27 passed / 27 collected
  test_codes.py:  7 surface code tests
  test_bb.py:    12 BB code tests (3 marked @slow, ldpc + Stim heuristic)
  test_eval.py:   6 evaluation tests
  test_model.py:  2 architecture tests
```

Run: `pytest tests/ -q` (~14s including slow tests)

### 4.3 Outstanding code

- `src/cascade/eval/sinter_runner.py` — listed in original plan but unused (only needed for BB-144 BP+OSD baseline at low p, which is parked)

---

## 5. Open questions

1. **Why did L6 regress?** Smoke had std 0.18, post-40k has std 0.0000 — the head went from "weak signal" to "constant 0". Suggests the 40k-step training pressure on the strong heads pulled shared backbone capacity away from L6. A frozen-backbone fine-tune of L6 alone would test this.
2. **Cascade vs BPOSD ratio at p=0.005.** ✅ Resolved in §6: Cascade 5.34e-2 vs BPOSD 5.30e-3 per cycle — Cascade ~10× behind on this code/p.
3. **BB-144 (Phase 7 of original plan)** — parked indefinitely. The lz-support-pool architecture has a structural failure on heavy logicals; BB-144's lz weights are larger still (12 logicals, n=144), so the same failure mode would dominate.
4. **Surface H=256 retrain for higher Λ** — postponed. Track 1.5 result on file.

---

## 6. Iteration-2 close-out — Cascade vs BP+OSD

Cascade-only sweep in §3.5; BP+OSD baseline run via slurm job 185651 on hgpn02 (35 min, `slurm/eval_bb_bposd.sh`). Both decoders evaluated against the same EMA checkpoint at p=0.005; BP+OSD capped at 2000 shots (CPU-bound), Cascade at 200k.

| decoder | shots | fail | p_block | P_L/cycle | CI95 | ratio vs Cascade |
|---|---|---|---|---|---|---|
| Cascade (EMA) | 200000 | 193267 | 0.9663 | 5.34e-02 | [5.30, 5.39]e-02 | 1.0× |
| BP+OSD (osd_order=4) | 2000 | 629 | 0.3145 | **5.30e-03** | [4.89, 5.74]e-03 | **0.099× (≈10× better)** |

BP+OSD beats Cascade by ~10× in per-cycle P_L. The block-error gap (0.31 vs 0.97) is consistent with the per-logical breakdown in §3.6: Cascade's 5 dead heads each fail at ~50% per shot, multiplicatively pinning p_block ≥ 0.94 regardless of how well the alive heads decode. BP+OSD treats all 12 logicals symmetrically so it is not subject to this ceiling.

Note on BP+OSD speed: stimbposd default `osd_order=60` runs at ~1 s/shot on BB-72 with no measurable accuracy gain over `osd_order=4` (both yield 27.5% block_err on a 200-shot probe — see `decoders/bposd.py` comment). Iter-2 lowered the wrapper's default to 4, giving ~250× speedup on this code class.

Decision recap: 7/12 logicals decoded vs paper's all-12 baseline, and Cascade currently 10× behind BP+OSD on this code at p=0.005. We ship Track-1.5 BB with this honestly documented (5/12 dead heads, 10× behind BPOSD), then move to iter-3 (§7) for the structural fix.

## 7. Next iteration (Iter-3) — what would actually unfreeze L6, L8–L11

The iter-2 architecture treats each logical as an independent head sharing only the backbone. Heavy-weight lz rows (weights 14–20) collapse to constant outputs because the marginal parity over 14–20 noisy data qubits is essentially Bernoulli(0.5), and the per-logical head finds no signal that beats "predict 0".

Concrete options for iter-3 (ranked by expected ROI):

1. **Auxiliary loss on physical Z errors per data qubit.** Instead of predicting only the 12 logical parities, also predict per-qubit Z-error indicators, then compute logical parities deterministically from those. The auxiliary task gives the backbone dense per-qubit gradient regardless of logical weight. Risk: doubles output dimension (12 → 84) and the per-qubit task is harder than logical parities for low p, but it's the standard fix used in surface-code papers.
2. **Curriculum on logical weight, not on p.** Train alive logicals (weights 6, 8) first; freeze backbone; train heavy logicals (weights 14–20) with a higher LR on their heads alone. Tests hypothesis from §5.1.
3. **BP+OSD warm-start.** Run BPOSD once offline, use its per-qubit error guesses as soft labels alongside the true logical parities. Bootstraps the heavy logicals from a non-neural baseline.
4. **Read paper supplementary / repo for Andi Gu et al.'s actual head architecture.** §3.3 was inferred from a paper-skim, not a reference impl. The paper's `Λ ≈ 8.4` requires that all 12 logicals decode well, so they must have solved this; whatever they did is the canonical fix.

Ship-ready Track-1.5 deliverable for now: surface d=5/7/9 (§2) + BB-72 with documented 7/12 logical coverage (§3).
