"""Generate the iter-3 BB-72 demo figures.

Three panels written as separate PDFs under figures/:

1. demo_bb72_iter3_vs_bposd.pdf — headline bar plot of per-cycle P_L,
   Cascade (iter-3) vs BP+OSD at p=0.005, with Wilson 95% CI.
2. demo_bb72_iter2_vs_iter3.pdf — progress plot showing the 40.8x P_L drop
   from iter-2 to iter-3, with the BP+OSD baseline as a horizontal band.
3. demo_bb72_iter3_per_logical.pdf — per-logical std + per-logical error
   bars across all 12 heads, demonstrating 12/12 alive.

Usage
-----
    python scripts/27_bb_iter3_figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _load(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _entry(payload: dict, p_phys: float, decoder: str) -> dict:
    for sweep in payload["sweep"]:
        if abs(sweep["p"] - p_phys) < 1e-9 and decoder in sweep["results"]:
            return sweep["results"][decoder]
    raise KeyError(f"{decoder} @ p={p_phys} not found in {payload.get('code', '?')}")


def fig_headline() -> None:
    """Cascade (iter-3) vs BP+OSD at p=0.005."""
    data = _load(RES / "deliverable" / "bb72_v3_iter3_bposd.json")
    cas = _entry(data, 0.005, "Cascade")
    bp = _entry(data, 0.005, "BP+OSD")

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    labels = ["Cascade\n(iter-3, EMA)", "BP+OSD\n(osd_order=4)"]
    vals = np.array([cas["p_l_per_cycle"], bp["p_l_per_cycle"]])
    err_lo = vals - np.array([cas["p_l_lo"], bp["p_l_lo"]])
    err_hi = np.array([cas["p_l_hi"], bp["p_l_hi"]]) - vals
    colors = ["#1f77b4", "#d62728"]

    bars = ax.bar(labels, vals, yerr=[err_lo, err_hi],
                  color=colors, capsize=6, edgecolor="black", linewidth=0.7)
    ax.set_yscale("log")
    ax.set_ylabel(r"per-cycle $P_L$  (lower is better)")
    ax.set_title("BB-[[72,12,6]] @ p = 0.005\n"
                 f"Cascade {vals[1]/vals[0]:.2f}x better than BP+OSD")
    ax.grid(True, axis="y", which="both", alpha=0.3)

    for bar, v, n_fail, n_tot in zip(
        bars, vals,
        [cas["n_failures"], bp["n_failures"]],
        [cas["n_total"], bp["n_total"]],
    ):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.6,
                f"{v:.2e}\n({n_fail}/{n_tot} shots)",
                ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    out = FIG / "demo_bb72_iter3_vs_bposd.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def fig_progress() -> None:
    """Iter-2 vs iter-3 Cascade, with BP+OSD horizontal reference."""
    iter2 = _load(RES / "deliverable" / "bb72_v2_iter2.json")
    iter3 = _load(RES / "deliverable" / "bb72_v3_iter3_bposd.json")
    bp = _entry(iter3, 0.005, "BP+OSD")

    sweep2 = sorted(iter2["sweep"], key=lambda x: x["p"])
    p2 = [s["p"] for s in sweep2]
    cas2 = [_entry(iter2, p, "Cascade") for p in p2]
    y2 = [c["p_l_per_cycle"] for c in cas2]
    lo2 = [c["p_l_lo"] for c in cas2]
    hi2 = [c["p_l_hi"] for c in cas2]

    cas3 = _entry(iter3, 0.005, "Cascade")

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    yerr2 = np.vstack([np.array(y2) - np.array(lo2), np.array(hi2) - np.array(y2)])
    ax.errorbar(p2, y2, yerr=yerr2, marker="s", linestyle="--",
                color="#7f7f7f", label="Cascade iter-2 (5/12 dead)",
                capsize=3, markersize=6)
    ax.errorbar([0.005], [cas3["p_l_per_cycle"]],
                yerr=[[cas3["p_l_per_cycle"] - cas3["p_l_lo"]],
                      [cas3["p_l_hi"] - cas3["p_l_per_cycle"]]],
                marker="o", linestyle="", color="#1f77b4", markersize=10,
                capsize=4, label="Cascade iter-3 (12/12 alive)")
    ax.errorbar([0.005], [bp["p_l_per_cycle"]],
                yerr=[[bp["p_l_per_cycle"] - bp["p_l_lo"]],
                      [bp["p_l_hi"] - bp["p_l_per_cycle"]]],
                marker="^", linestyle="", color="#d62728", markersize=9,
                capsize=4, label="BP+OSD (osd_order=4)")

    ratio = cas2[-1]["p_l_per_cycle"] / cas3["p_l_per_cycle"]
    ax.annotate(f"{ratio:.1f}x improvement\n(recipe-only fix)",
                xy=(0.005, cas3["p_l_per_cycle"]),
                xytext=(0.0035, cas3["p_l_per_cycle"] / 6),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"physical error rate $p$")
    ax.set_ylabel(r"per-cycle $P_L$")
    ax.set_title("BB-[[72,12,6]] progress: iter-2 → iter-3")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    out = FIG / "demo_bb72_iter2_vs_iter3.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def fig_per_logical() -> None:
    """Per-logical std and per-logical error across all 12 heads."""
    txt = (RES / "deliverable" / "bb72_v3_per_logical_p0.005.txt").read_text()
    rows = []
    for line in txt.splitlines():
        m = re.match(r"\s*(\d+)\s+\S+\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+(YES|NO)",
                     line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    rows.sort()
    k = np.array([r[0] for r in rows])
    std = np.array([r[1] for r in rows])
    err = np.array([r[2] for r in rows])
    weights = [6, 6, 6, 6, 6, 6, 14, 8, 16, 18, 20, 20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), sharex=True)

    bars1 = ax1.bar(k, std, color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax1.axhline(0.1, color="red", linestyle=":", linewidth=1, label="alive threshold (std=0.1)")
    ax1.set_xlabel("logical index k")
    ax1.set_ylabel("logit std (over 8192 shots)")
    ax1.set_title("Per-logical signal — 12/12 alive")
    ax1.set_xticks(k)
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(True, axis="y", alpha=0.3)
    for b, w in zip(bars1, weights):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                 f"w={w}", ha="center", va="bottom", fontsize=7.5)

    ax2.bar(k, err * 100, color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax2.axhline(50, color="red", linestyle=":", linewidth=1, label="random predictor (50%)")
    ax2.set_xlabel("logical index k")
    ax2.set_ylabel("per-logical error rate (%)")
    ax2.set_title("Per-logical error @ p = 0.005")
    ax2.set_xticks(k)
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, 8)

    fig.suptitle("BB-72 iter-3 per-logical breakdown (EMA, 8192 shots)",
                 fontsize=11)
    fig.tight_layout()
    out = FIG / "demo_bb72_iter3_per_logical.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main() -> None:
    fig_headline()
    fig_progress()
    fig_per_logical()


if __name__ == "__main__":
    main()
