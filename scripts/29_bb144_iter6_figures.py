"""Generate the iter-6 BB-[[144,12,12]] figures + headline table.

Consumes two independent result sources (they are produced by separate SLURM
jobs and never live in one JSON):

  * Cascade final numbers  -> results/bb144_mw_v4.json  (produced by
    slurm/eval_bb144_mw.sh once the v4 min-weight-basis chain reaches 40k).
    Schema: {"sweep": [{"p": .., "results": {"Cascade": {..}}}, ..]} — the
    final eval runs --no-mwpm and no --bposd, so each p carries ONLY "Cascade".

  * BP+OSD baseline        -> results/bb144_mw_bposd_p{P}.json  (one per p,
    produced by slurm/eval_bb144_mw_bposd.sh). Two schemas are handled:
      - sweep style (p=0.002/0.004/0.005/0.0055): take results["BP+OSD"].
      - flat merged style (p=0.003 -> bb144_mw_bposd_p0.003_merged.json,
        produced by scripts/merge_bposd_points.py): the point IS the top level.
    WARNING: the "Cascade" entry inside the *_bposd_* files is a mid-training
    live probe, NOT a paper number — this script never reads it.

Outputs (all paths overridable so a dev dry-run can target a scratch dir and
leave the real figures/ + reports/ untouched):

  1. <outdir>/bb144_mw_threshold.pdf  — per-cycle P_L vs physical p, log-log,
     Cascade (final) vs BP+OSD baseline, Wilson 95% CI.
  2. <outdir>/bb144_mw_headline.pdf   — bar chart at the decisive p
     (--headline-p, default 0.002), Cascade vs BP+OSD, log-y, honest ratio.
  3. <table-out>{.md,.csv}            — 5-row comparison table.

If results/bb144_mw_v4.json is absent (before the 40k eval lands), the script
does NOT crash: it prints a warning, plots/tabulates the BP+OSD baseline only
(a preview), and skips anything that needs Cascade numbers.

Usage
-----
    # After the 40k eval lands (writes into the real figures/ + reports/):
    python scripts/29_bb144_iter6_figures.py

    # Dev dry-run against a synthetic Cascade JSON, scratch outputs:
    python scripts/29_bb144_iter6_figures.py \
        --cascade /path/to/synthetic_cascade.json \
        --outdir  /path/to/scratch --table-out /path/to/scratch/table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless SLURM node — no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

# Paper headline sweep for iter-6 (low p first).
P_POINTS = [0.002, 0.003, 0.004, 0.005, 0.0055]

CAS_COLOR = "#1f77b4"
BP_COLOR = "#d62728"


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _p_tag(p: float) -> str:
    """0.0055 -> '0.0055', 0.002 -> '0.002' (matches the JSON filenames)."""
    return f"{p:g}"


# --------------------------------------------------------------------------- #
# BP+OSD baseline loading (handles both the sweep and flat-merged schemas)
# --------------------------------------------------------------------------- #
def _bposd_point(res_dir: Path, p: float) -> dict | None:
    """Return the BP+OSD result dict for one p, or None if no file is found.

    Prefers the merged file (more shots) when present.
    """
    tag = _p_tag(p)
    merged = res_dir / f"bb144_mw_bposd_p{tag}_merged.json"
    plain = res_dir / f"bb144_mw_bposd_p{tag}.json"

    if merged.exists():
        d = _load(merged)
        # flat schema: the point is the top-level object
        return _std_point(d)
    if plain.exists():
        d = _load(plain)
        # sweep schema
        for sweep in d.get("sweep", []):
            if abs(sweep["p"] - p) < 1e-9 and "BP+OSD" in sweep["results"]:
                return _std_point(sweep["results"]["BP+OSD"])
        return None
    return None


def _std_point(d: dict) -> dict:
    """Normalise a result dict to the five fields the plots/table need."""
    return {
        "p_l_per_cycle": d["p_l_per_cycle"],
        "p_l_lo": d["p_l_lo"],
        "p_l_hi": d["p_l_hi"],
        "n_failures": d["n_failures"],
        "n_total": d["n_total"],
        "sufficient": d.get("sufficient", True),
    }


# --------------------------------------------------------------------------- #
# Cascade final loading
# --------------------------------------------------------------------------- #
def _cascade_points(cascade_json: Path) -> dict[float, dict]:
    """Map p -> normalised Cascade point. Empty dict if the file is absent."""
    if not cascade_json.exists():
        print(f"[warn] Cascade JSON not found: {cascade_json}")
        print("[warn] -> BP+OSD-only preview (no Cascade curve / column).")
        return {}
    payload = _load(cascade_json)
    out: dict[float, dict] = {}
    for sweep in payload.get("sweep", []):
        p = float(sweep["p"])
        res = sweep.get("results", {})
        if "Cascade" in res:
            out[p] = _std_point(res["Cascade"])
    print(f"[ok] loaded Cascade final ({payload.get('weights', '?')} weights) "
          f"for p in {sorted(out)}")
    return out


def _yerr(pt: dict) -> list[float]:
    """[lo_err, hi_err] for a single point (matplotlib errorbar convention)."""
    v = pt["p_l_per_cycle"]
    return [v - pt["p_l_lo"], pt["p_l_hi"] - v]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_threshold(cas: dict[float, dict], bp: dict[float, dict],
                  outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.4))

    bp_p = [p for p in P_POINTS if p in bp]
    if bp_p:
        y = np.array([bp[p]["p_l_per_cycle"] for p in bp_p])
        err = np.array([_yerr(bp[p]) for p in bp_p]).T
        ax.errorbar(bp_p, y, yerr=err, marker="^", linestyle="-",
                    color=BP_COLOR, capsize=3, markersize=8,
                    label="BP+OSD (osd_order=4)")

    cas_p = [p for p in P_POINTS if p in cas]
    if cas_p:
        y = np.array([cas[p]["p_l_per_cycle"] for p in cas_p])
        err = np.array([_yerr(cas[p]) for p in cas_p]).T
        ax.errorbar(cas_p, y, yerr=err, marker="o", linestyle="-",
                    color=CAS_COLOR, capsize=3, markersize=8,
                    label="Cascade (iter-6, min-weight basis)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"physical error rate $p$")
    ax.set_ylabel(r"per-cycle $P_L$  (lower is better)")
    ax.set_title("BB-[[144,12,12]] iter-6: Cascade vs BP+OSD")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    out = outdir / "bb144_mw_threshold.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def fig_headline(cas: dict[float, dict], bp: dict[float, dict],
                 headline_p: float, outdir: Path) -> None:
    if headline_p not in bp:
        print(f"[warn] no BP+OSD point at p={headline_p}; skipping headline bar.")
        return
    if headline_p not in cas:
        print(f"[warn] no Cascade point at p={headline_p}; skipping headline bar "
              "(BP+OSD-only preview).")
        return

    c = cas[headline_p]
    b = bp[headline_p]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    labels = ["Cascade\n(iter-6)", "BP+OSD\n(osd_order=4)"]
    vals = np.array([c["p_l_per_cycle"], b["p_l_per_cycle"]])
    err = np.array([_yerr(c), _yerr(b)]).T
    bars = ax.bar(labels, vals, yerr=err, color=[CAS_COLOR, BP_COLOR],
                  capsize=6, edgecolor="black", linewidth=0.7)
    ax.set_yscale("log")
    ax.set_ylabel(r"per-cycle $P_L$  (lower is better)")

    # Honest ratio: state which decoder wins and by how much.
    ratio = b["p_l_per_cycle"] / c["p_l_per_cycle"]
    if ratio >= 1:
        verdict = f"Cascade {ratio:.1f}x better than BP+OSD"
    else:
        verdict = f"BP+OSD {1 / ratio:.1f}x better than Cascade"
    ax.set_title(f"BB-[[144,12,12]] @ p = {headline_p:g}\n{verdict}")
    ax.grid(True, axis="y", which="both", alpha=0.3)

    for bar, v, pt in zip(bars, vals, [c, b]):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.6,
                f"{v:.2e}\n({pt['n_failures']}/{pt['n_total']})",
                ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout()
    out = outdir / "bb144_mw_headline.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def _fmt(pt: dict | None) -> str:
    if pt is None:
        return "—"
    flag = "" if pt["sufficient"] else " *"
    return (f"{pt['p_l_per_cycle']:.3e} "
            f"[{pt['p_l_lo']:.2e}, {pt['p_l_hi']:.2e}] "
            f"({pt['n_failures']}/{pt['n_total']}){flag}")


def write_table(cas: dict[float, dict], bp: dict[float, dict],
                table_out: Path) -> None:
    table_out.parent.mkdir(parents=True, exist_ok=True)
    md_rows = ["| p | Cascade P_L/cycle [95% CI] (fail/tot) | "
               "BP+OSD P_L/cycle [95% CI] (fail/tot) | ratio (Cascade× better) |",
               "|---|---|---|---|"]
    csv_rows = ["p,cascade_pl,cascade_lo,cascade_hi,cascade_fail,cascade_tot,"
                "bposd_pl,bposd_lo,bposd_hi,bposd_fail,bposd_tot,ratio_cascade_better"]
    for p in P_POINTS:
        c = cas.get(p)
        b = bp.get(p)
        if c and b:
            r = b["p_l_per_cycle"] / c["p_l_per_cycle"]
            ratio = f"{r:.1f}x" if r >= 1 else f"{r:.2g}x"
        else:
            ratio = "—"
        md_rows.append(f"| {p:g} | {_fmt(c)} | {_fmt(b)} | {ratio} |")
        cv = (lambda d, k: f"{d[k]:.6e}" if d else "")
        cn = (lambda d, k: f"{d[k]}" if d else "")
        rnum = (f"{b['p_l_per_cycle'] / c['p_l_per_cycle']:.4f}"
                if (c and b) else "")
        csv_rows.append(
            f"{p:g},{cv(c,'p_l_per_cycle')},{cv(c,'p_l_lo')},{cv(c,'p_l_hi')},"
            f"{cn(c,'n_failures')},{cn(c,'n_total')},"
            f"{cv(b,'p_l_per_cycle')},{cv(b,'p_l_lo')},{cv(b,'p_l_hi')},"
            f"{cn(b,'n_failures')},{cn(b,'n_total')},{rnum}"
        )

    note = ("\n\n\\* = point flagged insufficient (n_failures < min_errors). "
            "ratio = BP+OSD P_L / Cascade P_L (>1 means Cascade wins). "
            "Cascade = 40k eval, best.pt @ step 38000, EMA weights "
            "(results/bb144_mw_v4.json, job 185232).\n")
    md_path = table_out.with_suffix(".md")
    csv_path = table_out.with_suffix(".csv")
    md_path.write_text("\n".join(md_rows) + note)
    csv_path.write_text("\n".join(csv_rows) + "\n")
    print(f"[table] wrote {md_path}")
    print(f"[table] wrote {csv_path}")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade", type=Path, default=RES / "bb144_mw_v4.json",
                    help="Cascade final eval JSON (may be absent pre-40k)")
    ap.add_argument("--bposd-dir", type=Path, default=RES,
                    help="directory holding bb144_mw_bposd_p*.json")
    ap.add_argument("--outdir", type=Path, default=FIG,
                    help="where to write the PDFs")
    ap.add_argument("--table-out", type=Path,
                    default=ROOT / "reports" / "bb144_iter6_table",
                    help="table path stem (.md and .csv are appended)")
    ap.add_argument("--headline-p", type=float, default=0.002,
                    help="physical p for the headline bar chart")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    bp = {p: pt for p in P_POINTS
          if (pt := _bposd_point(args.bposd_dir, p)) is not None}
    missing = [p for p in P_POINTS if p not in bp]
    if missing:
        print(f"[warn] no BP+OSD file for p in {missing}")
    cas = _cascade_points(args.cascade)

    fig_threshold(cas, bp, args.outdir)
    fig_headline(cas, bp, args.headline_p, args.outdir)
    write_table(cas, bp, args.table_out)
    print("[done] iter-6 BB-144 figures + table")


if __name__ == "__main__":
    main()
