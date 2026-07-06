"""Fit surface-code error-suppression factor Lambda from sweep JSONs.

Reads multiple ``20_eval_decoder.py`` JSON outputs (one per code distance) and
fits Lambda for each requested decoder (default: Cascade AND MWPM in one run),
so the reproduction claim "Lambda_Cascade > Lambda_MWPM" can be checked with
its confidence intervals side by side.

By default it fits at the *deepest* physical error rate where every distance
has sufficient statistics (``--select low``) — that is where the Cascade
advantage is largest and where the paper reports Lambda (p=0.002). Use ``--p``
to force a specific rate.

Example
-------

    python scripts/23_fit_lambda.py \\
        --inputs results/surface_d5_v4.json \\
                 results/surface_d7_v4.json \\
                 results/surface_d9_v4.json \\
        --p 0.002 --weighted --out results/surface_lambda_v4.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cascade.eval.lambda_fit import fit_lambda


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", required=True,
                   help="20_eval_decoder JSONs, one per distance")
    p.add_argument("--decoders", type=str, nargs="+", default=["Cascade", "MWPM"],
                   help="decoder names to fit (default: Cascade MWPM)")
    p.add_argument("--min-errors", type=int, default=100,
                   help="minimum n_failures per (decoder, p) point to be eligible")
    p.add_argument("--p", type=float, default=None,
                   help="force a specific physical error rate instead of auto-select")
    p.add_argument("--select", choices=["low", "high"], default="low",
                   help="when auto-selecting p: 'low' = deepest sub-threshold "
                        "(largest Lambda separation, paper regime); 'high' = "
                        "near-threshold (best statistics). Default low.")
    p.add_argument("--weighted", action="store_true",
                   help="weight the log-linear fit by n_failures (inverse-variance)")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="optional JSON dump of the fit result")
    return p.parse_args()


def _eligible_p_set(payload: dict, decoder: str, min_errors: int) -> set[float]:
    out: set[float] = set()
    for entry in payload["sweep"]:
        r = entry["results"].get(decoder)
        if r is None:
            continue
        if r["n_failures"] >= min_errors:
            out.add(round(float(entry["p"]), 6))
    return out


def _entry_for_p(payload: dict, decoder: str, p: float) -> dict | None:
    for entry in payload["sweep"]:
        if math.isclose(entry["p"], p, abs_tol=1e-9):
            return entry["results"].get(decoder)
    return None


def _distance_from_payload(payload: dict) -> int:
    name = payload["code"]
    if name.startswith("surface_d"):
        return int(name.split("_d")[1])
    raise ValueError(f"cannot infer distance from code name {name!r}")


def _select_p(payloads, distances, decoder, args) -> float:
    if args.p is not None:
        p_pick = float(args.p)
        for d, payload in zip(distances, payloads):
            r = _entry_for_p(payload, decoder, p_pick)
            if r is None:
                raise ValueError(f"d={d} has no sweep entry at p={p_pick}")
            if r["n_failures"] < args.min_errors:
                print(f"[fit] WARNING: {decoder} d={d} at p={p_pick} only has "
                      f"{r['n_failures']} errors (< {args.min_errors})")
        return p_pick
    eligible_per = [_eligible_p_set(pl, decoder, args.min_errors) for pl in payloads]
    common = set.intersection(*eligible_per) if eligible_per else set()
    if not common:
        print(f"[fit] {decoder}: no common p with >= {args.min_errors} failures at every distance.")
        for d, s in zip(distances, eligible_per):
            print(f"  d={d}: {sorted(s)}")
        raise SystemExit(1)
    p_pick = min(common) if args.select == "low" else max(common)
    print(f"[fit] {decoder}: common eligible p = {sorted(common)}; "
          f"select={args.select} -> p={p_pick}")
    return p_pick


def _fit_for_decoder(payloads, distances, decoder, p_pick, args) -> dict:
    n_failures, n_total, rounds_per_d = [], [], []
    for d, payload in zip(distances, payloads):
        r = _entry_for_p(payload, decoder, p_pick)
        if r is None:
            raise ValueError(f"{decoder} d={d} has no sweep entry at p={p_pick}")
        n_failures.append(int(r["n_failures"]))
        n_total.append(int(r["n_total"]))
        rounds_per_d.append(int(payload["rounds"]))
    res = fit_lambda(
        distances=distances, n_failures=n_failures, n_total=n_total,
        rounds_per_d=rounds_per_d, k=1, n_bootstrap=args.n_bootstrap,
        seed=args.seed, weighted=args.weighted,
    )
    return {
        "decoder": decoder,
        "p": p_pick,
        "distances": list(res.distances),
        "rounds": rounds_per_d,
        "n_failures": n_failures,
        "n_total": n_total,
        "p_l_per_cycle": list(res.p_l_per_cycle),
        "Lambda": res.Lambda,
        "Lambda_lo": res.Lambda_lo,
        "Lambda_hi": res.Lambda_hi,
        "all_sufficient": all(f >= args.min_errors for f in n_failures),
    }


def main() -> None:
    args = parse_args()
    payloads = [json.load(open(p)) for p in args.inputs]
    distances = [_distance_from_payload(pl) for pl in payloads]
    if len(set(distances)) != len(distances):
        raise ValueError(f"duplicate distances among inputs: {distances}")
    # Sort by distance for stable reporting.
    order = sorted(range(len(distances)), key=lambda i: distances[i])
    payloads = [payloads[i] for i in order]
    distances = [distances[i] for i in order]

    fits: dict[str, dict] = {}
    for decoder in args.decoders:
        p_pick = _select_p(payloads, distances, decoder, args)
        fit = _fit_for_decoder(payloads, distances, decoder, p_pick, args)
        fits[decoder] = fit
        print(f"\n[{decoder}] p={fit['p']} weighted={args.weighted}")
        for d, f, n, pl in zip(fit["distances"], fit["n_failures"],
                               fit["n_total"], fit["p_l_per_cycle"]):
            print(f"    d={d}: {f}/{n} block errors  P_L/cycle={pl:.4e}")
        print(f"    Lambda = {fit['Lambda']:.3f}  "
              f"95% CI [{fit['Lambda_lo']:.3f}, {fit['Lambda_hi']:.3f}]"
              f"{'' if fit['all_sufficient'] else '  [UNDER-SAMPLED]'}")

    # Reproduction check: Cascade advantage over MWPM.
    verdict = None
    if "Cascade" in fits and "MWPM" in fits:
        c, m = fits["Cascade"], fits["MWPM"]
        gap_clean = c["Lambda_lo"] > m["Lambda_hi"]
        verdict = {
            "cascade_beats_mwpm": c["Lambda"] > m["Lambda"],
            "non_overlapping_ci": bool(gap_clean),
            "lambda_ratio": (c["Lambda"] / m["Lambda"]) if m["Lambda"] else None,
        }
        print("\n[verdict] Lambda_Cascade={:.3f} vs Lambda_MWPM={:.3f}  "
              "(ratio {:.2f}x); CIs {}overlap".format(
                  c["Lambda"], m["Lambda"],
                  verdict["lambda_ratio"] or float("nan"),
                  "do NOT " if gap_clean else "DO "))
        if not gap_clean:
            print("[verdict] CIs overlap -> advantage NOT statistically established; "
                  "do not overclaim (need more shots or higher capacity).")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "inputs": [str(p) for p in args.inputs],
                "select": args.select, "weighted": args.weighted,
                "min_errors": args.min_errors,
                "fits": fits, "verdict": verdict,
            }, f, indent=2)
        print(f"\n[fit] wrote {args.out}")


if __name__ == "__main__":
    main()
