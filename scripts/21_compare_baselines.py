"""Plot Cascade vs baseline decoders from one or more 20_eval_decoder JSONs.

Paper-style: log y-axis, P_L per cycle vs p_phys, error bars from Wilson 95% CI.
Multiple JSONs are overlaid (e.g. surface d=5, d=7, d=9 on one figure, or
BB-72 vs BB-144).

Examples
--------

    # Surface code Lambda figure: three distances together
    python scripts/21_compare_baselines.py \
        --inputs results/surface_d5_v2.json results/surface_d7_v2.json results/surface_d9_v2.json \
        --labels "d=5" "d=7" "d=9" \
        --out figures/surface_lambda.pdf

    # BB-72 with BP+OSD baseline
    python scripts/21_compare_baselines.py \
        --inputs results/bb72_v2.json \
        --labels "BB[[72,12,6]]" \
        --out figures/bb72_compare.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_DECODER_STYLE = {
    "Cascade": {"marker": "o", "linestyle": "-"},
    "MWPM":    {"marker": "s", "linestyle": "--"},
    "BP+OSD":  {"marker": "^", "linestyle": ":"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--labels", type=str, nargs="+", default=None,
                   help="series label per input (default: code name)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--xlabel", type=str, default=r"physical error rate $p$")
    p.add_argument("--ylabel", type=str, default=r"per-cycle $P_L$")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("--labels must match --inputs in length")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for series_idx, in_path in enumerate(args.inputs):
        with open(in_path) as f:
            payload = json.load(f)
        label_prefix = (args.labels[series_idx] if args.labels else payload["code"])
        color = color_cycle[series_idx % len(color_cycle)]

        # Group sweep entries by decoder name
        by_decoder: dict[str, dict[str, list[float]]] = {}
        for entry in payload["sweep"]:
            p_phys = entry["p"]
            for decoder, r in entry["results"].items():
                acc = by_decoder.setdefault(decoder, {"p": [], "p_l": [], "lo": [], "hi": []})
                if r["n_total"] <= 0 or not np.isfinite(r["p_l_per_cycle"]):
                    continue
                acc["p"].append(p_phys)
                acc["p_l"].append(r["p_l_per_cycle"])
                acc["lo"].append(r["p_l_lo"])
                acc["hi"].append(r["p_l_hi"])

        for decoder, acc in by_decoder.items():
            if not acc["p"]:
                continue
            order = np.argsort(acc["p"])
            x = np.array(acc["p"])[order]
            y = np.array(acc["p_l"])[order]
            lo = np.array(acc["lo"])[order]
            hi = np.array(acc["hi"])[order]
            yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
            style = _DECODER_STYLE.get(decoder, {"marker": "x", "linestyle": "-."})
            ax.errorbar(
                x, y, yerr=yerr,
                marker=style["marker"], linestyle=style["linestyle"],
                color=color,
                label=f"{label_prefix} · {decoder}",
                capsize=2.5, markersize=5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    if args.title:
        ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
