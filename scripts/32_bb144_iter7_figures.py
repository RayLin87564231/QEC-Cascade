"""Generate the iter-7 (mixed-p) BB-[[144,12,12]] figures + headline table.

Sibling of scripts/29_bb144_iter6_figures.py, adapted for the v7 mixed-p
Cascade run. Consumes THREE independent result sources (never combined in one
JSON):

  * v7 Cascade final numbers -> results/bb144_v7_mixp.json (produced by
    slurm/eval_bb144_v7_mixp.sh once the mixed-p 40k chain lands).
    Schema: {"sweep": [{"p": .., "results": {"Cascade": {..}}}, ..]} — each
    p carries ONLY "Cascade" (final eval, --no-mwpm, no --bposd).

  * v6 Cascade numbers (reference/comparison only) -> results/bb144_mw_v4.json
    Same schema as above. This is the iter-6 headline run being superseded;
    plotted as a dashed reference line, never used in the win/ratio numbers.

  * BP+OSD baseline -> results/bb144_mw_bposd_p{P}.json (one per p). Two
    schemas handled, identical to the iter-6 script:
      - sweep style (p=0.002/0.004/0.005/0.0055): take results["BP+OSD"].
      - flat merged style (p=0.003 -> bb144_mw_bposd_p0.003_merged.json,
        produced by scripts/merge_bposd_points.py): the point IS the top
        level object (has "decoder": "BP+OSD").
    WARNING: the "Cascade" entry embedded in the *_bposd_* files is a
    mid-training live probe (old v6 checkpoint), NOT a paper number — this
    script never reads it.

Outputs (all paths overridable so a dev dry-run can target a scratch dir and
leave the real figures/ + reports/ untouched). NOTE: these are new filenames
distinct from the iter-6 outputs — nothing under figures/ or reports/ that
iter-6 wrote is ever opened for writing by this script.

  1. <outdir>/bb144_v7_mixp_threshold.pdf — per-cycle P_L vs physical p,
     log-log, three curves: v7 Cascade (final), BP+OSD baseline, v6 Cascade
     (dashed, reference-only), Wilson 95% CI error bars.
  2. <outdir>/bb144_v7_mixp_headline.pdf  — bar chart at the decisive p
     (--headline-p, default 0.002), v7 Cascade vs BP+OSD only (same 2-bar
     layout as the iter-6 headline chart), log-y, honest ratio.
  3. <table-out>{.md,.csv} — one row per p: v7 Cascade p_block + P_L/cycle
     [95% CI] (fail/tot), BP+OSD same, v6 Cascade p_block (reference), and
     the p_block-based Cascade/BP+OSD win ratio.

If results/bb144_v7_mixp.json is absent, the script does NOT crash: it prints
a warning and skips anything that needs v7 Cascade numbers (BP+OSD-only /
v6-only preview).

Usage
-----
    # After the v7 40k eval lands (writes into the real figures/ + reports/):
    python scripts/32_bb144_iter7_figures.py

    # Dev dry-run against scratch outputs:
    python scripts/32_bb144_iter7_figures.py \
        --outdir /path/to/scratch --table-out /path/to/scratch/table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless SLURM/login node — no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

# Paper headline sweep (low p first) — same grid as iter-6.
P_POINTS = [0.002, 0.003, 0.004, 0.005, 0.0055]

V7_COLOR = "#1f77b4"
BP_COLOR = "#d62728"
V6_COLOR = "#7f7f7f"


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
    """Normalise a result dict to the fields the plots/table need."""
    return {
        "p_block": d["p_block"],
        "p_l_per_cycle": d["p_l_per_cycle"],
        "p_l_lo": d["p_l_lo"],
        "p_l_hi": d["p_l_hi"],
        "n_failures": d["n_failures"],
        "n_total": d["n_total"],
        "sufficient": d.get("sufficient", True),
    }


# --------------------------------------------------------------------------- #
# Cascade (v7 final / v6 reference) loading — same schema, different files.
# --------------------------------------------------------------------------- #
def _cascade_points(cascade_json: Path, label: str) -> dict[float, dict]:
    """Map p -> normalised Cascade point. Empty dict if the file is absent."""
    if not cascade_json.exists():
        print(f"[warn] {label} Cascade JSON not found: {cascade_json}")
        print(f"[warn] -> figures/table skip anything needing {label} Cascade.")
        return {}
    payload = _load(cascade_json)
    out: dict[float, dict] = {}
    for sweep in payload.get("sweep", []):
        p = float(sweep["p"])
        res = sweep.get("results", {})
        if "Cascade" in res:
            out[p] = _std_point(res["Cascade"])
    print(f"[ok] loaded {label} Cascade ({payload.get('weights', '?')} weights) "
          f"for p in {sorted(out)}")
    return out


def _yerr(pt: dict) -> list[float]:
    """[lo_err, hi_err] for a single point (matplotlib errorbar convention)."""
    v = pt["p_l_per_cycle"]
    return [v - pt["p_l_lo"], pt["p_l_hi"] - v]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_threshold(v7: dict[float, dict], bp: dict[float, dict],
                  v6: dict[float, dict], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.4))

    v6_p = [p for p in P_POINTS if p in v6]
    if v6_p:
        y = np.array([v6[p]["p_l_per_cycle"] for p in v6_p])
        err = np.array([_yerr(v6[p]) for p in v6_p]).T
        ax.errorbar(v6_p, y, yerr=err, marker="s", linestyle="--",
                    color=V6_COLOR, capsize=3, markersize=6, alpha=0.8,
                    label="Cascade (iter-6, reference)")

    bp_p = [p for p in P_POINTS if p in bp]
    if bp_p:
        y = np.array([bp[p]["p_l_per_cycle"] for p in bp_p])
        err = np.array([_yerr(bp[p]) for p in bp_p]).T
        ax.errorbar(bp_p, y, yerr=err, marker="^", linestyle="-",
                    color=BP_COLOR, capsize=3, markersize=8,
                    label="BP+OSD (osd_order=4)")

    v7_p = [p for p in P_POINTS if p in v7]
    if v7_p:
        y = np.array([v7[p]["p_l_per_cycle"] for p in v7_p])
        err = np.array([_yerr(v7[p]) for p in v7_p]).T
        ax.errorbar(v7_p, y, yerr=err, marker="o", linestyle="-",
                    color=V7_COLOR, capsize=3, markersize=8,
                    label="Cascade (iter-7, mixed-p)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"physical error rate $p$")
    ax.set_ylabel(r"per-cycle $P_L$  (lower is better)")
    ax.set_title("BB-[[144,12,12]] iter-7 (mixed-p): Cascade vs BP+OSD")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    out = outdir / "bb144_v7_mixp_threshold.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def fig_headline(v7: dict[float, dict], bp: dict[float, dict],
                 headline_p: float, outdir: Path) -> None:
    if headline_p not in bp:
        print(f"[warn] no BP+OSD point at p={headline_p}; skipping headline bar.")
        return
    if headline_p not in v7:
        print(f"[warn] no v7 Cascade point at p={headline_p}; skipping headline "
              "bar (BP+OSD-only preview).")
        return

    c = v7[headline_p]
    b = bp[headline_p]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    labels = ["Cascade\n(iter-7, mixed-p)", "BP+OSD\n(osd_order=4)"]
    vals = np.array([c["p_l_per_cycle"], b["p_l_per_cycle"]])
    err = np.array([_yerr(c), _yerr(b)]).T
    bars = ax.bar(labels, vals, yerr=err, color=[V7_COLOR, BP_COLOR],
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
    out = outdir / "bb144_v7_mixp_headline.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def _fmt_pl(pt: dict | None) -> str:
    if pt is None:
        return "—"
    flag = "" if pt["sufficient"] else " *"
    return (f"{pt['p_l_per_cycle']:.3e} "
            f"[{pt['p_l_lo']:.2e}, {pt['p_l_hi']:.2e}] "
            f"({pt['n_failures']}/{pt['n_total']}){flag}")


def _fmt_pblock(pt: dict | None) -> str:
    if pt is None:
        return "—"
    flag = "" if pt["sufficient"] else " *"
    return f"{pt['p_block']:.4g} ({pt['n_failures']}/{pt['n_total']}){flag}"


def write_table(v7: dict[float, dict], bp: dict[float, dict],
                v6: dict[float, dict], table_out: Path) -> None:
    table_out.parent.mkdir(parents=True, exist_ok=True)
    md_rows = [
        "| p | v7 Cascade p_block (fail/tot) | v7 Cascade P_L/cycle [95% CI] "
        "(fail/tot) | BP+OSD p_block (fail/tot) | BP+OSD P_L/cycle [95% CI] "
        "(fail/tot) | v6 Cascade p_block (fail/tot, ref) | ratio "
        "(Cascade× better, p_block) |",
        "|---|---|---|---|---|---|---|",
    ]
    csv_rows = [
        "p,v7_cascade_pblock,v7_cascade_pl,v7_cascade_pl_lo,v7_cascade_pl_hi,"
        "v7_cascade_fail,v7_cascade_tot,"
        "bposd_pblock,bposd_pl,bposd_pl_lo,bposd_pl_hi,bposd_fail,bposd_tot,"
        "v6_cascade_pblock,v6_cascade_fail,v6_cascade_tot,"
        "ratio_cascade_better_pblock"
    ]
    for p in P_POINTS:
        c = v7.get(p)
        b = bp.get(p)
        v6pt = v6.get(p)
        if c and b:
            r = b["p_block"] / c["p_block"]
            ratio = f"{r:.1f}x" if r >= 1 else f"{r:.2g}x"
        else:
            ratio = "—"
        md_rows.append(
            f"| {p:g} | {_fmt_pblock(c)} | {_fmt_pl(c)} | {_fmt_pblock(b)} | "
            f"{_fmt_pl(b)} | {_fmt_pblock(v6pt)} | {ratio} |"
        )
        cv = (lambda d, k: f"{d[k]:.6e}" if d else "")
        cn = (lambda d, k: f"{d[k]}" if d else "")
        cb = (lambda d: f"{d['p_block']:.6f}" if d else "")
        rnum = f"{b['p_block'] / c['p_block']:.4f}" if (c and b) else ""
        csv_rows.append(
            f"{p:g},{cb(c)},{cv(c,'p_l_per_cycle')},{cv(c,'p_l_lo')},"
            f"{cv(c,'p_l_hi')},{cn(c,'n_failures')},{cn(c,'n_total')},"
            f"{cb(b)},{cv(b,'p_l_per_cycle')},{cv(b,'p_l_lo')},"
            f"{cv(b,'p_l_hi')},{cn(b,'n_failures')},{cn(b,'n_total')},"
            f"{cb(v6pt)},{cn(v6pt,'n_failures')},{cn(v6pt,'n_total')},"
            f"{rnum}"
        )

    note = (
        "\n\n\\* = point flagged insufficient (n_failures < min_errors). "
        "ratio = BP+OSD p_block / v7 Cascade p_block (>1 means Cascade wins; "
        "this is the block-level win margin, distinct from the iter-6 table's "
        "P_L/cycle-based ratio). v7 Cascade = mixed-p 40k eval, EMA weights "
        "(results/bb144_v7_mixp.json). v6 Cascade column is reference-only "
        "(results/bb144_mw_v4.json, the flat ~0.40 floor iter-7 fixes) and is "
        "not part of the ratio. BP+OSD = results/bb144_mw_bposd_p{P}.json "
        "(p=0.003 uses the *_merged.json top-up); only decoder=\"BP+OSD\" "
        "entries are used, never the live-probe \"Cascade\" entry embedded in "
        "those files.\n"
    )
    md_path = table_out.with_suffix(".md")
    csv_path = table_out.with_suffix(".csv")
    md_path.write_text("\n".join(md_rows) + note)
    csv_path.write_text("\n".join(csv_rows) + "\n")
    print(f"[table] wrote {md_path}")
    print(f"[table] wrote {csv_path}")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade-v7", type=Path, default=RES / "bb144_v7_mixp.json",
                    help="v7 mixed-p Cascade final eval JSON")
    ap.add_argument("--cascade-v6", type=Path, default=RES / "bb144_mw_v4.json",
                    help="v6 Cascade final eval JSON (reference/dashed line)")
    ap.add_argument("--bposd-dir", type=Path, default=RES,
                    help="directory holding bb144_mw_bposd_p*.json")
    ap.add_argument("--outdir", type=Path, default=FIG,
                    help="where to write the PDFs")
    ap.add_argument("--table-out", type=Path,
                    default=ROOT / "reports" / "bb144_iter7_table",
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
    v7 = _cascade_points(args.cascade_v7, "v7")
    v6 = _cascade_points(args.cascade_v6, "v6")

    fig_threshold(v7, bp, v6, args.outdir)
    fig_headline(v7, bp, args.headline_p, args.outdir)
    write_table(v7, bp, v6, args.table_out)
    print("[done] iter-7 (mixed-p) BB-144 figures + table")


if __name__ == "__main__":
    main()
