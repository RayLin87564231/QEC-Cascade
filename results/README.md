# Results layout

Reorganized 2026-05-05. Top-level symlinks preserve old paths used by scripts
and slurm jobs.

## `deliverable/`
JSON / TXT cited in `reports/iteration_2_status.md` and `iteration_3_status.md`.

- `bb72_v3_iter3_bposd.json` — iter-3 BB-72 Cascade vs BP+OSD @ p=0.005.
- `bb72_v3_per_logical_p0.005.txt` — iter-3 per-logical std/err table.
- `bb72_v2_iter2.json` — iter-2 BB-72 Cascade sweep (4 noise points).
- `bb72_v2_iter2_bposd.json` — iter-2 BB-72 BP+OSD baseline.
- `surface_d{5,7,9}_v2.json` — iter-2 surface Cascade vs MWPM sweeps.

## `archive/`
Auxiliary experiments not in the main report.

- `latency_*.json` — inference latency benchmarks.
- `postselect_*.json` — confidence-threshold acceptance curves.
- `reliability_surf_d7.json` — d=7 reliability diagram data.
- `surface_d{5,7,9,11}_v3.json` — iter-3 surface retrains (undocumented;
  d=11 underperforms MWPM).

## Symlinks (backward-compat)
```
bb72_v3_iter3_bposd.json -> deliverable/bb72_v3_iter3_bposd.json
surface_d5_v2.json       -> deliverable/surface_d5_v2.json
surface_d7_v2.json       -> deliverable/surface_d7_v2.json
surface_d9_v2.json       -> deliverable/surface_d9_v2.json
reliability_surf_d7.json -> archive/reliability_surf_d7.json
```
