# QEC-Cascade

Reproduction and extension of the **Cascade neural decoder** for quantum error
correction (arXiv:2604.08358), in PyTorch + Stim. The decoder is trained and
evaluated on two code families:

- **Rotated surface code** (d = 5, 7, 9) under circuit-level depolarizing noise,
  benchmarked against MWPM (PyMatching).
- **Bivariate bicycle (BB) codes** — [[72,12,6]] and the [[144,12,12]] "gross
  code" — benchmarked against BP+OSD (stimbposd).

All training/evaluation runs on SLURM (single node, 1–8 GPUs); every result
JSON in `results/` comes from a logged SLURM job.

![Cascade vs BP+OSD on the BB-144 gross code](figures/fig1b_scalable_bb144_v7.png)

*Logical error rate per logical qubit per cycle on the [[144,12,12]] gross code
(12 syndrome rounds, circuit-level depolarizing noise), our data, in the style
of Fig. 1(b) of the Cascade paper. Blue: our iteration-7 mixed-p Cascade model;
red: BP+OSD (osd_order=4). Error bars are Wilson 95% CIs; lines are single
power-law fits with the measured exponents (~p^7.0 vs ~p^5.4). Over
p = 0.2%–0.55% a two-component (waterfall + floor) fit is not yet resolvable
from our data, so we plot the honest single-slope fits. Reproduce with
`scripts/34_scalable_fig1bc.py`.*

## Headline results so far

### BB-144 [[144,12,12]]: Cascade beats BP+OSD at all five swept points

Final evaluation of the iteration-7 model (`results/bb144_v7_mixp.json`, EMA
weights, best checkpoint at step 36k of 40k) against the BP+OSD baseline
(`results/bb144_mw_bposd_p*.json`), P_L = logical error rate per logical qubit
per cycle (k = 12, 12 rounds):

| p (phys) | Cascade P_L/cycle | BP+OSD P_L/cycle | ratio |
|----------|-------------------|------------------|-------|
| 0.002    | 2.63e-6           | 2.32e-5          | 8.8×  |
| 0.003    | 4.02e-5           | 2.56e-4          | 6.4×  |
| 0.004    | 3.00e-4           | 1.26e-3          | 4.2×  |
| 0.005    | 1.62e-3           | 3.53e-3          | 2.2×  |
| 0.0055   | 3.04e-3           | 5.88e-3          | 1.9×  |

Every Cascade point has ≥200 block failures. Caveat on the p = 0.002 row: the
BP+OSD baseline has only 8 block failures there, so the 8.8× ratio carries a
wide CI (roughly 4×–18×), but the win direction is stable. Fitted power-law
exponents over this range: Cascade ~p^7.00 (95% CI [6.88, 7.13]) vs BP+OSD
~p^5.45 (95% CI [5.23, 5.65]) — Cascade's curve is not just lower but steeper,
so the advantage grows as p decreases.

A zero-syndrome sanity probe of the final model (`scripts/33_probe_bb144_zero_v7.py`)
fires 0/12 readout heads on the all-zero syndrome, and at p = 0.0005 the
measured p_block is 0 — i.e. no spurious low-p error floor (see below for why
this probe exists).

### Surface code: error-suppression factor Λ (H=512, d5→d9 weighted fit)

Λ = factor by which the logical error rate per cycle drops when d → d+2.
Fits over d = 5, 7, 9 with ≥200 block failures per point
(`results/surface_lambda_v5*.json`, figure `figures/surface_waterfall_v5.pdf`).

| p (phys) | Λ Cascade [95% CI] | Λ MWPM [95% CI] |
|----------|--------------------|-----------------|
| 0.002    | 7.48 [6.73, 8.20]  | 5.25 [4.96, 5.57] |
| 0.0015   | 10.02 [9.01, 10.98] | 6.82 [6.42, 7.23] |
| 0.001    | 15.38 [13.83, 16.85] | 10.60 [10.01, 11.22] |

The Λ-vs-p "waterfall" confirms the deep sub-threshold scaling regime: Cascade's
suppression factor grows markedly faster than MWPM's as p decreases.

### BB-72 [[72,12,6]]

@ p=0.005 (6 rounds): Cascade P_L/cycle = 1.31e-3 vs BP+OSD 5.38e-3
(`results/bb72_v3_iter3_bposd.json`).

## What it took to make BB-144 work

Two failure → root-cause → fix cycles, both documented in `reports/`:

1. **Dead readout heads (iteration 6 prep).** With logical representatives
   picked by plain Gaussian elimination, the high-weight logicals (weight
   24–38) produce permanently dead readout heads — a mean-pool readout cannot
   represent a wide XOR parity. Fixed by a **minimum-weight logical basis**
   (randomized information-set search; all 12 logicals at weight d = 12),
   after which all 12 heads train. See
   `reports/iteration_6_deadhead_rootcause.md`.

2. **Single-p training does not generalize (iteration 6 → 7).** The
   iteration-6 model, trained only at p = 0.0055, evaluated fine near its
   training point but hit a p_block floor of ~0.40 at low p — losing to BP+OSD
   by ~157× at p = 0.002. Probes showed the model misreads sparse low-p
   syndromes (even at p = 0.0005 it declared failures ~42% of the time).
   Iteration 7 retrained from scratch with **mixed-p training**: p drawn
   log-uniformly from [0.001, 0.0055] per micro-batch (16-point cached circuit
   grid), model selection on the average of p_block at the two endpoint noise
   levels. That single change removed the floor entirely and produced the
   five-point sweep above. Full narrative: `reports/iteration_6_status.md`,
   `reports/iteration_7_plan.md`, `reports/iteration_7_status.md`.

## Layout

```
src/cascade/        package: codes/ (surface, BB), models/, decoders/ (MWPM,
                    BP+OSD wrappers), data/ (Stim sampling), train/, eval/
scripts/            numbered pipeline entry points (train / eval / fits / figures)
slurm/              SLURM batch scripts for every train/eval/smoke job
tests/              pytest suite
reports/            per-iteration lab notes (status, root-cause analyses,
                    handoffs, literature surveys)
results/            evaluation outputs (JSON) — small, committed
figures/            generated figures (PDF/PNG)
```

Beyond the lab notes, `reports/universal_decoder_survey.md` is a verified
literature map on universal (code-agnostic) neural decoders and FPGA decoding,
with every citation checked against the source
(`reports/universal_decoder_survey_evidence.json`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # or: pip install -e ".[dev]" for pytest
```

Requires Python ≥ 3.10 and a CUDA GPU for training (H100/H200-class used here;
evaluation of small models runs on CPU too, slowly).

### External code

`external/BivariateBicycleCodes/` (BB-code distance utilities used for
cross-checks) is a third-party clone and is not committed. To restore:

```bash
git clone https://github.com/sbravyi/BivariateBicycleCodes external/BivariateBicycleCodes
```

Model checkpoints and raw SLURM logs are intentionally not committed
(~1 GB / run); they live on the cluster.

## Reproducing the figures

Figure scripts are pure numpy/matplotlib over the committed `results/` JSONs —
no GPU or checkpoints needed:

```bash
.venv/bin/python scripts/34_scalable_fig1bc.py   # headline figure above
.venv/bin/python scripts/32_bb144_iter7_figures.py  # iter-7 threshold/headline plots + tables
```

`scripts/34_scalable_fig1bc.py` also contains a stub for the companion
accuracy-vs-latency panel (Fig. 1(c) of the paper); it is not implemented yet
because no BB-144 inference-latency measurements exist for either decoder.

## Status

Research in progress (2026-07). Surface-code results (iter-5) and the BB-144
mixed-p result (iter-7, five-point win over BP+OSD) are final. Candidate next
steps: BB-144 latency measurements for the accuracy–latency panel, a next
iteration / different code, or paper writing.
