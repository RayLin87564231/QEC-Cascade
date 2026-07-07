#!/usr/bin/env python3
"""Merge independent BP+OSD baseline runs for a single p point.

The BB-144 BP+OSD baseline is sampled across separate SLURM invocations
(main job + top-up workers). Sampling is NOT fixed-seed across invocations,
so the runs are statistically independent and merge by summing
(n_failures, n_total); the merged P_L / 95% CI is then recomputed with the
same estimator the eval uses (cascade.eval.pblock_to_pl.per_cycle_pl_with_ci).

Only the BP+OSD column is merged — the Cascade column in the baseline JSONs
is a mid-training probe and is intentionally ignored here.

Usage:
    python scripts/merge_bposd_points.py --p 0.003
    python scripts/merge_bposd_points.py --p 0.002 --k 12 --rounds 12

By default it globs results/bb144_mw_bposd_p{P}.json and
results/bb144_mw_bposd_p{P}_topup_w*.json (skips any *_merged.json), sums the
BP+OSD counts, and writes results/bb144_mw_bposd_p{P}_merged.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cascade.eval.pblock_to_pl import per_cycle_pl_with_ci  # noqa: E402


def load_bposd(path: Path) -> tuple[int, int]:
    d = json.loads(path.read_text())
    sweep = d["sweep"]
    if len(sweep) != 1:
        raise ValueError(f"{path}: expected a single-p sweep, got {len(sweep)}")
    r = sweep[0]["results"].get("BP+OSD")
    if r is None:
        raise ValueError(f"{path}: no BP+OSD result present")
    return int(r["n_failures"]), int(r["n_total"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", required=True, help="p point, e.g. 0.003")
    ap.add_argument("--k", type=int, default=12, help="# logical qubits (BB-144=12)")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    base = rd / f"bb144_mw_bposd_p{args.p}.json"
    files: list[Path] = []
    if base.exists():
        files.append(base)
    for f in sorted(glob.glob(str(rd / f"bb144_mw_bposd_p{args.p}_topup_w*.json"))):
        files.append(Path(f))
    files = [f for f in files if "_merged" not in f.name]
    if not files:
        raise SystemExit(f"no input files matched for p={args.p} under {rd}/")

    total_fail = total_shots = 0
    per_file = []
    for f in files:
        nf, nt = load_bposd(f)
        total_fail += nf
        total_shots += nt
        per_file.append((f.name, nf, nt))

    pl, lo, hi = per_cycle_pl_with_ci(total_fail, total_shots, k=args.k, rounds=args.rounds)
    out = {
        "code": "bb_144_12_12",
        "p": float(args.p),
        "k": args.k,
        "rounds": args.rounds,
        "decoder": "BP+OSD",
        "merged_from": [{"file": n, "n_failures": nf, "n_total": nt} for n, nf, nt in per_file],
        "n_failures": total_fail,
        "n_total": total_shots,
        "p_block": total_fail / total_shots if total_shots else float("nan"),
        "p_l_per_cycle": pl,
        "p_l_lo": lo,
        "p_l_hi": hi,
        "sufficient": total_fail >= 100,
    }
    out_path = Path(args.out) if args.out else rd / f"bb144_mw_bposd_p{args.p}_merged.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"[merge] p={args.p}  files={len(files)}")
    for n, nf, nt in per_file:
        print(f"    {n}: {nf}/{nt}")
    print(f"  TOTAL BP+OSD: {total_fail}/{total_shots}  "
          f"(p_block={out['p_block']:.5g})")
    print(f"  P_L/cycle = {pl:.4g}  [{lo:.4g}, {hi:.4g}]  "
          f"sufficient(>=100 fail)={out['sufficient']}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
