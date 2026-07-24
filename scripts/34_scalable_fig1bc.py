"""Reproduce Fig 1(b) of "Scalable Neural Decoders for Practical
Fault-Tolerant Quantum Computation" (Gu et al., arXiv:2604.08358) using our
own BB-[[144,12,12]] iteration-7 results: v7 Cascade (mixed-p, EMA weights)
vs the BP+OSD baseline (osd_order=4).

Sibling of scripts/32_bb144_iter7_figures.py (same JSON schemas / loading
logic, reused verbatim -- see _std_point/_bposd_point/_cascade_points below)
and scripts/28_waterfall_fit.py (same power-law fitting machinery, imported
directly from cascade.eval.waterfall -- not reinvented).

Data sources (full audit trail:
~/.claude/scratch/fig1bc_data_inventory.md):
  * v7 Cascade (mixed-p, EMA weights) -> results/bb144_v7_mixp.json
  * BP+OSD baseline (osd_order=4)     -> results/bb144_mw_bposd_p{P}.json
    (p=0.003 uses the *_merged.json top-up, more shots)
Both already carry precomputed p_l_per_cycle / p_l_lo / p_l_hi (Wilson 95%
CI, mapped through the paper's Eq.(1) inversion by
cascade.eval.pblock_to_pl.per_cycle_pl_with_ci with k=12, rounds=12) -- this
script does not recompute that conversion, only loads and plots it.

Honesty constraints (per task spec -- numerical honesty over visual
similarity to the paper):
  - We only have 5 points per curve (p = 0.2%-0.55%). The paper's two-regime
    "waterfall + distance-limited floor" decomposition is not necessarily
    resolvable in this narrow range. We ALWAYS fit a single power law per
    curve (log-log linear regression via cascade.eval.waterfall.fit_powerlaw)
    and only switch to a two-segment fit
    (cascade.eval.waterfall.fit_two_component) if it converges well away
    from its parameter bounds with a tightly constrained steep exponent --
    see fit_curve() for the exact acceptance test. On the actual data (see
    module docstring history / stdout at runtime) the two-component fit is
    degenerate (unconstrained or pinned to its upper bound) for BOTH curves,
    so both are drawn as a single power law.
  - We NEVER reuse the paper's reported exponents (10.8 / 6.4 / 5.4). Every
    exponent printed/plotted here is fit from our own data.
  - No "Distance-limited floor" / "Waterfall" regime text labels and no
    lattice inset are drawn -- that narrative belongs to the paper's own
    data, not ours (per task spec).

Panel (c) (accuracy-latency trade-off) is a stub: see
fig_accuracy_latency_stub() below. No BB-144 latency data exists yet in this
repo for either decoder, so nothing is plotted for panel (c).

Usage
-----
    .venv/bin/python scripts/34_scalable_fig1bc.py

    # Dev dry-run against scratch outputs:
    .venv/bin/python scripts/34_scalable_fig1bc.py --outdir /path/to/scratch

No GPU / torch / checkpoints/ access required or performed: this script only
reads results/*.json (plain JSON, already containing the derived P_L
numbers) and plots with matplotlib (Agg backend, headless-safe).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless login node -- no display
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

from cascade.eval.waterfall import (  # noqa: E402
    fit_powerlaw, fit_two_component, floor_exponent, two_component_curve,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

# Paper headline sweep (low p first) -- same grid as scripts/32.
P_POINTS = [0.002, 0.003, 0.004, 0.005, 0.0055]

# BB-[[144,12,12]]: rounds=12 == code distance (see inventory notes on
# scripts/24_bb_per_logical.py / slurm/eval_bb144_v7_mixp.sh). Used only to
# fix the shallow exponent `b` in a two-component fit attempt -- the fit
# itself still runs on our own data, not the paper's numbers.
BB144_DISTANCE = 12

CASCADE_COLOR = "#1f77b4"  # blue diamond -- repo convention + paper panel(b)
BPOSD_COLOR = "#d62728"    # red diamond

# Inventory spot-check values (~/.claude/scratch/fig1bc_data_inventory.md),
# used purely as a runtime sanity check that this script reads the same
# numbers a human already transcribed by hand.
SPOT_CHECKS = [
    ("v7 Cascade", 0.002, 2.634838e-06),
    ("v7 Cascade", 0.004, 3.000848e-04),
    ("BP+OSD", 0.0055, 5.875897e-03),
]


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _p_tag(p: float) -> str:
    """0.0055 -> '0.0055', 0.002 -> '0.002' (matches the JSON filenames)."""
    return f"{p:g}"


def _std_point(d: dict) -> dict:
    """Normalise a result dict to the fields the plot/fit need."""
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
# Loading (mirrors scripts/32_bb144_iter7_figures.py -- same two JSON
# schemas: sweep-style and flat-merged BP+OSD).
# --------------------------------------------------------------------------- #
def _bposd_point(res_dir: Path, p: float) -> dict | None:
    """Return the BP+OSD result dict for one p, or None if no file is found.

    Prefers the merged file (more shots) when present.
    """
    tag = _p_tag(p)
    merged = res_dir / f"bb144_mw_bposd_p{tag}_merged.json"
    plain = res_dir / f"bb144_mw_bposd_p{tag}.json"

    if merged.exists():
        return _std_point(_load(merged))
    if plain.exists():
        d = _load(plain)
        for sweep in d.get("sweep", []):
            if abs(sweep["p"] - p) < 1e-9 and "BP+OSD" in sweep["results"]:
                return _std_point(sweep["results"]["BP+OSD"])
        return None
    return None


def _cascade_points(cascade_json: Path, label: str) -> dict[float, dict]:
    """Map p -> normalised Cascade point. Empty dict if the file is absent."""
    if not cascade_json.exists():
        print(f"[warn] {label} JSON not found: {cascade_json}")
        return {}
    payload = _load(cascade_json)
    out: dict[float, dict] = {}
    for sweep in payload.get("sweep", []):
        p = float(sweep["p"])
        res = sweep.get("results", {})
        if "Cascade" in res:
            out[p] = _std_point(res["Cascade"])
    print(f"[ok] loaded {label} ({payload.get('weights', '?')} weights) "
          f"for p in {sorted(out)}")
    return out


def _yerr(pt: dict) -> list[float]:
    """[lo_err, hi_err] for a single point (matplotlib errorbar convention)."""
    v = pt["p_l_per_cycle"]
    return [max(v - pt["p_l_lo"], 0.0), max(pt["p_l_hi"] - v, 0.0)]


# --------------------------------------------------------------------------- #
# Spot check (acceptance criterion: 3 points, script value vs inventory).
# --------------------------------------------------------------------------- #
def run_spot_checks(v7: dict[float, dict], bp: dict[float, dict]) -> None:
    print("\n[spot-check] script-computed P_L vs inventory-transcribed P_L:")
    lookup = {"v7 Cascade": v7, "BP+OSD": bp}
    all_ok = True
    for label, p, expected in SPOT_CHECKS:
        pt = lookup[label].get(p)
        if pt is None:
            print(f"  [MISSING] {label} p={p}")
            all_ok = False
            continue
        got = pt["p_l_per_cycle"]
        ok = math.isclose(got, expected, rel_tol=1e-4)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} p={p}: "
              f"script={got:.6e}  inventory={expected:.6e}")
        all_ok = all_ok and ok
    if not all_ok:
        raise SystemExit("[spot-check] mismatch against inventory -- aborting")
    print("[spot-check] all 3 points match inventory.\n")


# --------------------------------------------------------------------------- #
# Fitting: single power law always; two-component only if decisively
# supported (see acceptance test below). Never uses the paper's exponents.
# --------------------------------------------------------------------------- #
def fit_curve(label: str, p_arr: np.ndarray, pl_arr: np.ndarray) -> dict:
    single = fit_powerlaw(p_arr, pl_arr)
    print(f"[fit] {label}: single power law P_L ~ p^{single.slope:.2f} "
          f"(95% CI [{single.slope_lo:.2f}, {single.slope_hi:.2f}], "
          f"n={single.n_points})")

    floor_exp = floor_exponent(BB144_DISTANCE)
    two = fit_two_component(p_arr, pl_arr, floor_exp=floor_exp)
    use_two = False
    if two is not None:
        a, a_err, b = two["a"], two["a_err"], two["b"]
        near_upper_bound = a > (floor_exp + 20) - 2.0  # fit_two_component's bound
        well_constrained = np.isfinite(a_err) and 0 < a_err < 0.3 * abs(a - b)
        steep_gap = abs(a - b) > 2.0
        use_two = well_constrained and steep_gap and not near_upper_bound
        verdict = "USE two-segment" if use_two else "reject -> keep single"
        reason = "pinned to fit bound" if near_upper_bound else (
            "unconstrained (a_err too large)" if not well_constrained and not near_upper_bound
            else ("insufficient exponent gap" if not steep_gap else "")
        )
        print(f"[fit] {label}: two-component attempt A*p^{a:.2f} + B*p^{b:.2f} "
              f"(a_err={a_err:.2g}) -> {verdict}"
              + (f" ({reason})" if not use_two else ""))
    else:
        print(f"[fit] {label}: two-component fit unavailable/did not "
              f"converge (need scipy + >=4 points) -> keep single")

    return {"single": single, "two": two if use_two else None,
            "floor_exp": floor_exp}


def _fit_curve_xy(fit_result: dict, p_lo: float, p_hi: float) -> tuple[np.ndarray, np.ndarray]:
    pp = np.linspace(p_lo, p_hi, 100)
    two = fit_result["two"]
    if two is not None:
        yy = two_component_curve(pp, two["A"], two["a"], two["B"], two["b"])
    else:
        single = fit_result["single"]
        yy = np.exp(single.log_intercept) * pp ** single.slope
    return pp, yy


def _fit_label(fit_result: dict) -> str:
    two = fit_result["two"]
    if two is not None:
        return f"~p^{two['a']:.1f} + p^{two['b']:.1f}"
    return f"~p^{fit_result['single'].slope:.1f}"


# --------------------------------------------------------------------------- #
# Panel (b) figure
# --------------------------------------------------------------------------- #
def fig_panel_b(v7: dict[float, dict], bp: dict[float, dict],
                 outdir: Path) -> dict:
    p7 = np.array(sorted(v7))
    pl7 = np.array([v7[p]["p_l_per_cycle"] for p in p7])
    pb = np.array(sorted(bp))
    plb = np.array([bp[p]["p_l_per_cycle"] for p in pb])

    fit7 = fit_curve("v7 Cascade", p7, pl7)
    fitb = fit_curve("BP+OSD", pb, plb)

    fig, ax = plt.subplots(figsize=(5.8, 4.6))

    # BP+OSD (red diamond, plotted first so it sits behind Cascade at
    # overlapping x if any).
    yerr_b = np.array([_yerr(bp[p]) for p in pb]).T
    ax.errorbar(pb * 100, plb, yerr=yerr_b, marker="D", linestyle="none",
                color=BPOSD_COLOR, markersize=7, capsize=3,
                markeredgecolor="black", markeredgewidth=0.4,
                label="BP+OSD (osd_order=4)", zorder=3)

    # Cascade (blue diamond).
    yerr_7 = np.array([_yerr(v7[p]) for p in p7]).T
    ax.errorbar(p7 * 100, pl7, yerr=yerr_7, marker="D", linestyle="none",
                color=CASCADE_COLOR, markersize=7, capsize=3,
                markeredgecolor="black", markeredgewidth=0.4,
                label="Cascade (iter-7, mixed-p)", zorder=3)

    # Fit lines, extended slightly beyond the data range, with the fitted
    # exponent labelled beside each line.
    x_lo = min(p7.min(), pb.min()) * 0.85
    x_hi = max(p7.max(), pb.max()) * 1.2

    pp7, yy7 = _fit_curve_xy(fit7, x_lo, x_hi)
    ax.plot(pp7 * 100, yy7, color=CASCADE_COLOR, lw=1.2, alpha=0.6, zorder=1)
    i7 = int(len(pp7) * 0.62)
    ax.annotate(_fit_label(fit7), xy=(pp7[i7] * 100, yy7[i7]),
                xytext=(6, -12), textcoords="offset points",
                fontsize=9.5, color=CASCADE_COLOR, ha="left")

    ppb, yyb = _fit_curve_xy(fitb, x_lo, x_hi)
    ax.plot(ppb * 100, yyb, color=BPOSD_COLOR, lw=1.2, alpha=0.6, zorder=1)
    ib = int(len(ppb) * 0.35)
    ax.annotate(_fit_label(fitb), xy=(ppb[ib] * 100, yyb[ib]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=9.5, color=BPOSD_COLOR, ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")

    all_lo = np.concatenate([[v7[p]["p_l_lo"] for p in p7],
                              [bp[p]["p_l_lo"] for p in pb]])
    all_hi = np.concatenate([[v7[p]["p_l_hi"] for p in p7],
                              [bp[p]["p_l_hi"] for p in pb]])
    ymin = float(all_lo.min()) / 10.0
    ymax = float(all_hi.max()) * 10.0
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0.15, 0.7)

    ax.set_xlabel("Physical error rate $p$ (%)")
    ax.set_ylabel(r"Logical error rate $P_L$ (per logical qubit per cycle)")
    ax.set_title("BB-[[144,12,12]]: Cascade (iter-7, mixed-p) vs BP+OSD")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks([0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    ax.grid(True, which="major", alpha=0.25)
    # Reorder legend to Cascade-then-BP+OSD (matches paper panel (b) and
    # the task spec) regardless of draw order above.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: "Cascade" not in labels[i])
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=9, frameon=False, loc="upper left")

    fig.tight_layout()
    png = outdir / "fig1b_scalable_bb144_v7.png"
    pdf = outdir / "fig1b_scalable_bb144_v7.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {png}")
    print(f"[plot] wrote {pdf}")

    return {"v7": fit7, "bp": fitb}


# --------------------------------------------------------------------------- #
# Panel (c) -- stub. Do not fabricate data; see docstring for what's needed.
# --------------------------------------------------------------------------- #
def fig_accuracy_latency_stub(
    cascade_latency_json: Path = RES / "latency_bb144_v7mixp.json",
    bposd_latency_json: Path = RES / "latency_bb144_bposd.json",
    outdir: Path = FIG,
) -> None:
    """Panel (c): accuracy-latency trade-off, Cascade (GPU, amortized
    latency across batch sizes) vs BP+OSD (single-shot CPU latency), at a
    fixed physical error rate p.

    STUB -- deliberately not implemented. Per the panel-(c) audit in
    ~/.claude/scratch/fig1bc_data_inventory.md, this repo currently has NO
    BB-[[144,12,12]] latency measurement for either decoder:

      * Cascade: needs a GPU job running
        ``scripts/25_inference_latency.py --code bb --bb-variant 144
        --ckpt checkpoints/bb_144_12_12_v7_bb144_mixp/best.pt --use-bf16
        --out results/latency_bb144_v7mixp.json`` (the only BB-144 v7
        checkpoint we have P_L numbers for -- see fig_panel_b() above).
      * BP+OSD: needs a new, purpose-built single-shot timing script -- the
        existing eval pipeline (scripts/20_eval_decoder.py) never records
        per-shot decode time, only coarse SLURM job wall time
        (logs/eval_bb144_bposd_166180.out has an ~11h job wall time across
        5 p-points and no per-shot breakdown -- unusable as a latency
        number).

    ``cascade_latency_json`` / ``bposd_latency_json`` / ``outdir`` are
    parameterised so the plotting call site is ready once those
    measurements exist; this function does not read them yet and refuses to
    run rather than fabricate placeholder numbers.
    """
    raise NotImplementedError(
        "panel (c) stub: BB-144 latency data does not exist yet for "
        "Cascade or BP+OSD in this repo -- see this function's docstring "
        "and the Panel (c) section of fig1bc_data_inventory.md for what "
        "measurements are needed before this can be implemented honestly."
    )


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cascade-v7", type=Path, default=RES / "bb144_v7_mixp.json",
                     help="v7 mixed-p Cascade final eval JSON")
    ap.add_argument("--bposd-dir", type=Path, default=RES,
                     help="directory holding bb144_mw_bposd_p*.json")
    ap.add_argument("--outdir", type=Path, default=FIG,
                     help="where to write fig1b_scalable_bb144_v7.{png,pdf}")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    bp = {p: pt for p in P_POINTS
          if (pt := _bposd_point(args.bposd_dir, p)) is not None}
    missing_bp = [p for p in P_POINTS if p not in bp]
    if missing_bp:
        print(f"[warn] no BP+OSD file for p in {missing_bp}")

    v7 = _cascade_points(args.cascade_v7, "v7 Cascade")
    missing_v7 = [p for p in P_POINTS if p not in v7]
    if missing_v7:
        print(f"[warn] no v7 Cascade point for p in {missing_v7}")

    if not v7 or not bp:
        raise SystemExit("[error] panel (b) needs both v7 Cascade and "
                          "BP+OSD points; aborting")

    run_spot_checks(v7, bp)

    fits = fig_panel_b(v7, bp, args.outdir)

    print("\n[summary] fitted single power-law exponents (P_L ~ p^x, fit "
          "from our own data only -- paper exponents 10.8/6.4/5.4 NOT used):")
    s7, sb = fits["v7"]["single"], fits["bp"]["single"]
    two7, twob = fits["v7"]["two"], fits["bp"]["two"]
    print(f"  Cascade (iter-7 mixed-p): x = {s7.slope:.2f} "
          f"(95% CI [{s7.slope_lo:.2f}, {s7.slope_hi:.2f}])"
          + ("  [plotted as two-segment instead]" if two7 else "  [plotted]"))
    print(f"  BP+OSD (osd_order=4):     x = {sb.slope:.2f} "
          f"(95% CI [{sb.slope_lo:.2f}, {sb.slope_hi:.2f}])"
          + ("  [plotted as two-segment instead]" if twob else "  [plotted]"))
    print("\n[done] panel (b) reproduced. panel (c) is a stub -- see "
          "fig_accuracy_latency_stub() (not called; no BB-144 latency data "
          "exists yet in this repo).")


if __name__ == "__main__":
    main()
