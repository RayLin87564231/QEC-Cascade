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

## Headline results so far

### Surface code: error-suppression factor Λ (H=512, d5→d9 weighted fit)

Λ = factor by which the logical error rate per cycle drops when d → d+2.
Fits over d = 5, 7, 9 with ≥200 block failures per point
(`results/surface_lambda_v5*.json`, figure `figures/surface_waterfall_v5.pdf`).

| p (phys) | Λ Cascade [95% CI] | Λ MWPM [95% CI] |
|----------|--------------------|-----------------|
| 0.002    | 7.49 [6.73, 8.20]  | 5.25 [4.96, 5.57] |
| 0.0015   | 10.02 [9.01, 10.98] | 6.82 [6.42, 7.23] |
| 0.001    | 15.38 [13.83, 16.85] | 10.61 [10.01, 11.22] |

The Λ-vs-p "waterfall" confirms the deep sub-threshold scaling regime: Cascade's
suppression factor grows markedly faster than MWPM's as p decreases.

### BB codes

- **BB-72 [[72,12,6]]** @ p=0.005 (6 rounds): Cascade P_L/cycle = 1.31e-3 vs
  BP+OSD 5.38e-3 (`results/bb72_v3_iter3_bposd.json`).
- **BB-144 [[144,12,12]]** — in progress. Key finding: with logical
  representatives picked by plain Gaussian elimination, the high-weight
  logicals (weight 24–38) produce permanently dead readout heads (a mean-pool
  readout cannot represent a wide XOR parity). Fixed by a **minimum-weight
  logical basis** (randomized information-set search; all 12 logicals at
  weight d=12), after which all 12 heads train. See
  `reports/iteration_6_deadhead_rootcause.md`. Final trained numbers and the
  BP+OSD comparison on the min-weight circuit will land in `results/` when the
  current training chain finishes.

Full experiment narrative, decisions, and negative results are in
`reports/iteration_*_status.md`.

## Layout

```
src/cascade/        package: codes/ (surface, BB), models/, decoders/ (MWPM,
                    BP+OSD wrappers), data/ (Stim sampling), train/, eval/
scripts/            numbered pipeline entry points (train / eval / fits / figures)
slurm/              SLURM batch scripts for every train/eval/smoke job
tests/              pytest suite
reports/            per-iteration lab notes (status, root-cause analyses, handoffs)
results/            evaluation outputs (JSON) — small, committed
figures/            generated figures (PDF)
```

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

## Status

Research in progress (2026-07). The BB-144 min-weight-basis training chain and
its BP+OSD baseline are currently running; surface-code results (iter-5) are
final.
