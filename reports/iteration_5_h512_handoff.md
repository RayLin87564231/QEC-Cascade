# Iteration 5 (H=512 scale-up) — Handoff / Resume Notes

Snapshot: 2026-07-03 ~07:20
Goal: push measured Λ toward the paper's ≈8.4 by (a) scaling the model H=256→**512**
and (b) measuring **deeper** (down to p=0.001). Baseline to beat: **iter-4 H=256
gave Λ_Cascade = 6.851 [6.162, 7.508] @ p=0.002** (see `iteration_4_status.md`).

---

## Current state (what is running / done)

| distance | H=512 training | deep eval (p→0.001) | status |
|---|---|---|---|
| d5 | ✅ job 162919 done (40k, 3.4 st/s) | ✅ job 162976 → `results/surface_d5_v5.json` | **no gain vs H=256** (saturated, expected) |
| d7 | ✅ job 162920 done 08:57 (best_step=39000, **ema saved**, best_p_block=0.00342 < v4's 0.00378) | 🔄 job **163002** RUNNING → `results/surface_d7_v5.json` | H=512 looks BETTER than H=256 (confirm on deep eval) |
| d9 | 🔄 job **162921** (~49% @ 08:58, ~16:00 ETA) | ⏳ job **163003** (dependency=afterok:162921) | auto-runs after train |

**Dependency chain is set** — the d7/d9 deep evals auto-start when training
succeeds, WITHOUT needing an active Claude session / terminal. SLURM handles it.
`squeue -u leo07010` to watch. If a training job FAILS (not COMPLETED), its eval
stays held on `(DependencyNeverSatisfied)` — cancel it and investigate.

Config: `slurm/train_surface_v5.sh` (H=512, L=d+1, 40k steps, eff batch ≈3328),
`slurm/eval_surface_v5.sh` (p = 0.001 0.0015 0.002 0.003 0.004 0.005, 24h wall,
caps d5=3e7 / d7=1e8 / d9=4e8, target 200 failures self-sizing).

---

## Remaining MANUAL steps (after d7 & d9 evals finish)

All are login-node / fast. Run from `/work/leo07010/Ray/QEC/cascade` with
`source .venv/bin/activate`.

### 1. Confirm all three v5 eval JSONs exist
```
ls -la results/surface_d{5,7,9}_v5.json
```

### 2. Fit Λ (H=512) at the deepest sufficient anchor
```
python scripts/23_fit_lambda.py \
    --inputs results/surface_d5_v5.json results/surface_d7_v5.json results/surface_d9_v5.json \
    --decoders Cascade MWPM --p 0.002 --weighted \
    --out results/surface_lambda_v5.json
# also try --p 0.0015 and --p 0.001 (deeper anchors, if d9 reached 200 failures there)
```
Compare Λ_Cascade(H512) to **6.851** (H256) and the paper **8.4**.

### 3. Waterfall figure (H=512)
```
python scripts/28_waterfall_fit.py \
    --inputs results/surface_d5_v5.json results/surface_d7_v5.json results/surface_d9_v5.json \
    --p-max 0.006 --decoders Cascade MWPM --weighted \
    --fig figures/surface_waterfall_v5.pdf --out results/surface_waterfall_v5.json
```

### 4. (Fairness) live-vs-live H256-vs-H512 cross-check — see caveat below
Re-eval the H=256 v4 checkpoints with `--prefer live` so both H are compared on
live weights (v5 uses live; see caveat). Quick, ~minutes each:
```
for d in 5 7 9; do B=$((d+1)); \
python scripts/20_eval_decoder.py --code surface --distance $d \
  --ckpt checkpoints/surface_d${d}_v4_d${d}/best.pt --prefer live \
  --hidden 256 --blocks $B --p 0.002 0.003 0.004 0.005 \
  --shots 30000000 --target-failures 200 --min-errors 100 --batch 8192 \
  --out results/surface_d${d}_v4_live.json; done
# then fit Λ on the *_v4_live.json set → Λ_H256(live), compare to Λ_H512(live)
```

### 5. Write `reports/iteration_5_status.md`
Answer the question: did H=512 lift Λ from 6.85 toward 8? Two valid outcomes:
- Λ rises materially → capacity was a bottleneck; closer to paper.
- Λ stays ~7 → H=256 already near capacity for this spec; paper's 8.4 needs the
  larger L=14 + larger distances (d up to 15), not just H.

---

## CAVEAT: EMA vs live weights (important, see memory `cascade-ema-warmup-gotcha`)

`ema_decay=0.9998` → EMA "warm" only after **25,000 steps** (62% of a 40k run).
Consequences observed on H=512:
- v5 d5 peaked at step 20k (< 25k) → `best.pt` has `model_ema=None` → eval used
  **live** weights. v4 (H=256) peaked at 28k/34k/35k → all had ema.
- On H=512, **live is consistently better than ema anyway** (ema lags too much at
  0.9998 over 40k; e.g. d7 step 28k: live 0.00403 vs ema 0.00513). So evaluating
  v5 on live is correct (best available), not a bug.
- For a clean capacity comparison, step 4 re-evals H=256 on live too (live-vs-live).

**Do NOT reuse `ema_decay=0.9998` for iter-6 BB-144 or any 40k retrain** — set it
so warmup ≤ steps/5 (e.g. 0.999). Fix location: `src/cascade/train/trainer_v2.py`
line ~213 (`ema_warm_steps`).

---

## Artifacts so far
```
checkpoints/surface_d{5,7,9}_v5_d{5,7,9}/best.pt   (d5 done; d7,d9 writing)
results/surface_d5_v5.json                          (d5 deep eval done)
slurm/train_surface_v5.sh, slurm/eval_surface_v5.sh (new)
reports/iteration_4_status.md                       (H=256 baseline: Λ=6.85)
reports/iteration_5_h512_handoff.md                 (this file)
```
d5 v5 vs v4 (Cascade P_L/cycle): p0.001 1.38e-5 vs 1.36e-5, p0.002 1.07e-4 vs
1.11e-4 — identical within noise (d5 saturated).
