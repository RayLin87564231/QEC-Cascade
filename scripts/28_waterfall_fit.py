"""Waterfall-scaling fit + figure for surface-code logical error rates.

Reads multiple ``20_eval_decoder.py`` sweep JSONs (one per distance, each with
several physical error rates) and, for each decoder, fits the effective
sub-threshold suppression slope ``m`` of ``P_L ~ p^m`` per distance. The
"waterfall" signature is ``m`` markedly steeper than the distance floor
exponent ``floor((d+1)/2)`` and Cascade steeper than MWPM. A two-component
``A p^a + B p^b`` fit is attempted where the data support it; the floor below
the measured range is drawn as extrapolation.

Example
-------

    python scripts/28_waterfall_fit.py \\
        --inputs results/surface_d5_v4.json \\
                 results/surface_d7_v4.json \\
                 results/surface_d9_v4.json \\
        --p-max 0.006 --decoders Cascade MWPM \\
        --fig figures/surface_waterfall.pdf \\
        --out results/surface_waterfall.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cascade.eval.waterfall import (
    fit_powerlaw, fit_two_component, floor_exponent, two_component_curve,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True,
                    help="20_eval_decoder JSONs, one per distance")
    ap.add_argument("--decoders", type=str, nargs="+", default=["Cascade", "MWPM"])
    ap.add_argument("--p-max", type=float, default=None,
                    help="only fit sub-threshold points with p <= p-max")
    ap.add_argument("--min-errors", type=int, default=100,
                    help="only fit points with n_failures >= this (sufficient)")
    ap.add_argument("--weighted", action="store_true",
                    help="weight the log-log slope fit by n_failures")
    ap.add_argument("--fig", type=Path, default=Path("figures/surface_waterfall.pdf"))
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args()


def _distance(payload: dict) -> int:
    name = payload["code"]
    if name.startswith("surface_d"):
        return int(name.split("_d")[1])
    raise ValueError(f"cannot infer distance from {name!r}")


def _series(payload: dict, decoder: str, p_max, min_errors):
    """Return sorted (p, P_L, P_L_lo, P_L_hi, n_fail) arrays for sufficient points."""
    rows = []
    for e in payload["sweep"]:
        r = e["results"].get(decoder)
        if r is None:
            continue
        if p_max is not None and e["p"] > p_max + 1e-12:
            continue
        if r["n_failures"] < min_errors:
            continue
        rows.append((float(e["p"]), r["p_l_per_cycle"], r["p_l_lo"],
                     r["p_l_hi"], r["n_failures"]))
    rows.sort()
    if not rows:
        return (np.array([]),) * 5
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]


def main() -> None:
    args = parse_args()
    payloads = [json.load(open(p)) for p in args.inputs]
    dists = [_distance(pl) for pl in payloads]
    order = sorted(range(len(dists)), key=lambda i: dists[i])
    payloads = [payloads[i] for i in order]
    dists = [dists[i] for i in order]

    styles = {"Cascade": dict(ls="-", marker="o"),
              "MWPM": dict(ls="--", marker="s")}
    cmap = plt.get_cmap("viridis")
    colors = {d: cmap(i / max(1, len(dists) - 1)) for i, d in enumerate(dists)}

    fig, ax = plt.subplots(figsize=(7, 5))
    result: dict = {"decoders": {}}

    for decoder in args.decoders:
        result["decoders"][decoder] = {}
        for d, payload in zip(dists, payloads):
            p, pl, lo, hi, nf = _series(payload, decoder, args.p_max, args.min_errors)
            if p.size == 0:
                continue
            weights = nf if args.weighted else None
            fit = fit_powerlaw(p, pl, weights=weights)
            floor_e = floor_exponent(d)
            two = fit_two_component(p, pl, floor_exp=floor_e)
            result["decoders"][decoder][str(d)] = {
                "p": p.tolist(), "p_l": pl.tolist(),
                "slope_m": fit.slope, "slope_lo": fit.slope_lo, "slope_hi": fit.slope_hi,
                "floor_exponent": floor_e,
                "waterfall_steeper_than_floor": bool(fit.slope_lo > floor_e),
                "n_points": fit.n_points,
                "two_component": two,
            }
            # Plot data + fitted slope line.
            style = styles.get(decoder, dict(ls="-", marker="o"))
            yerr = np.vstack([np.clip(pl - lo, 0, None), np.clip(hi - pl, 0, None)])
            label = f"{decoder} d={d} (m={fit.slope:.1f})"
            ax.errorbar(p, pl, yerr=yerr, color=colors[d], **style,
                        markersize=4, capsize=2, lw=1.2,
                        label=label if decoder == "Cascade" else None)
            pp = np.linspace(p.min(), p.max(), 50)
            ax.plot(pp, np.exp(fit.log_intercept) * pp ** fit.slope,
                    color=colors[d], ls=style["ls"], lw=0.8, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("per-cycle logical error rate $P_L$")
    ax.set_title("Surface-code waterfall: Cascade (solid) vs MWPM (dashed)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7, loc="lower right", title="steep slope $m$ (Cascade)")
    fig.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig)
    print(f"[waterfall] wrote {args.fig}")

    # Verdict summary.
    print("\n[waterfall] effective sub-threshold slope m per distance "
          "(floor exponent in brackets):")
    for decoder in args.decoders:
        dd = result["decoders"].get(decoder, {})
        parts = []
        for d in dists:
            e = dd.get(str(d))
            if e is None:
                continue
            flag = "*" if e["waterfall_steeper_than_floor"] else " "
            parts.append(f"d={d}: m={e['slope_m']:.1f}[{e['floor_exponent']}]{flag}")
        print(f"  {decoder:>8}: " + "  ".join(parts))
    print("  (* = slope CI lower bound exceeds the distance floor exponent = waterfall)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result["inputs"] = [str(p) for p in args.inputs]
        result["p_max"] = args.p_max
        result["weighted"] = args.weighted
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[waterfall] wrote {args.out}")


if __name__ == "__main__":
    main()
