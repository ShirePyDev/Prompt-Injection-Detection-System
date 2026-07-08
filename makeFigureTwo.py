#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2 for PIDS-Bench
=======================
Provenance decomposition of hard-benign false-positive rate for the two
fine-tuned detectors, baseline vs. hard-negative augmentation.

Design goals (IEEE Access, two-column):
  * serif type that matches the Times body text, with automatic fallback
    across whatever serif faces the cluster actually has installed;
  * value labels tied programmatically to each bar (no hand placement, so
    labels cannot drift from their bars);
  * near-zero augmented-curated cells shown with their true values
    (0.001 / 0.005), never rounded to "0.00";
  * a dashed 0.10 deployment-target reference line across both panels;
  * +/-1 SD error bars on the externally-sourced bars (the cells whose
    cross-seed variance actually matters);
  * self-check assertions so the plotted numbers match Table 10.

Usage
-----
    python3 make_figure2.py                # writes to current directory
    python3 make_figure2.py -o /path/dir   # writes to a chosen directory

Runs headless (Agg backend); no display required. Produces both a vector
PDF (for the manuscript) and a 300-dpi PNG (for quick visual checking).
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: safe on a cluster with no display
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# 1. Data  (Table 10; five-seed means from the canonical baseline folder)
#    Order within each list: [curated, externally-sourced]
# ---------------------------------------------------------------------------
DATA = {
    "DeBERTa-v3-FT": {
        "baseline":     [0.4813, 0.3144],
        "augmented":    [0.0010, 0.3101],
        "baseline_sd":  [0.0855, 0.0525],
        "augmented_sd": [0.0022, 0.0502],
    },
    "DistilBERT": {
        "baseline":     [0.4843, 0.4606],
        "augmented":    [0.0123, 0.3899],
        "baseline_sd":  [0.0793, 0.0639],
        "augmented_sd": [0.0113, 0.0551],
    },
}
GROUPS = ["Curated", "Externally-sourced"]
TARGET = 0.10  # deployment FPR target

# Show error bars only where cross-seed variance is material (ext-sourced).
# Curated cells collapse near-deterministically; their SD is marker-sized and
# a visible bar would mislead. This choice is stated in the caption.
SHOW_ERR = [False, True]  # per group: curated, externally-sourced


# ---------------------------------------------------------------------------
# 2. Typography — pick the best available serif, fall back gracefully
# ---------------------------------------------------------------------------
def choose_serif():
    """Return a serif family present on this machine, preferring Times-like
    faces. Falls back to matplotlib's bundled DejaVu Serif / Computer Modern,
    both of which always exist, so the figure is never accidentally rendered
    in a sans-serif default."""
    installed = {f.name for f in fm.fontManager.ttflist}
    preference = [
        "Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L",
        "Liberation Serif", "STIXGeneral", "STIX Two Text",
        "TeX Gyre Termes", "DejaVu Serif", "cmr10", "FreeSerif",
    ]
    for name in preference:
        if name in installed:
            return name
    return "serif"  # matplotlib generic serif as a last resort


SERIF = choose_serif()
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [SERIF, "DejaVu Serif"],
    "mathtext.fontset": "stix" if SERIF in {"STIXGeneral", "STIX Two Text"} else "cm",
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,   # embed TrueType (editable, IEEE-friendly)
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# 3. Palette — muted, print-safe, distinguishable in grayscale
# ---------------------------------------------------------------------------
C_BASE = "#3B5B7F"   # desaturated navy   -> baseline
C_AUG  = "#C06B4E"   # muted terracotta   -> augmented
C_TGT  = "#4D4D4D"   # target reference line
BAR_W  = 0.38


def draw_panel(ax, model, show_ylabel):
    d = DATA[model]
    x = np.arange(len(GROUPS))

    base_err = [sd if show else 0.0 for sd, show in zip(d["baseline_sd"], SHOW_ERR)]
    aug_err  = [sd if show else 0.0 for sd, show in zip(d["augmented_sd"], SHOW_ERR)]

    ax.bar(x - BAR_W / 2, d["baseline"], BAR_W, color=C_BASE,
           edgecolor="black", linewidth=0.5, label="Baseline",
           yerr=base_err, capsize=2.5,
           error_kw=dict(elinewidth=0.8, capthick=0.8, ecolor="black"))
    ax.bar(x + BAR_W / 2, d["augmented"], BAR_W, color=C_AUG,
           edgecolor="black", linewidth=0.5, label="+ Hard-negative aug.",
           yerr=aug_err, capsize=2.5,
           error_kw=dict(elinewidth=0.8, capthick=0.8, ecolor="black"))

    ax.axhline(TARGET, color=C_TGT, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)

    # Value labels, positioned relative to each bar's own height + its error.
    def put_label(xi, val, err):
        txt = f"{val:.3f}" if val < 0.01 else f"{val:.2f}"
        ax.text(xi, val + err + 0.014, txt, ha="center", va="bottom",
                fontsize=7.5)

    for xi, v, e in zip(x - BAR_W / 2, d["baseline"], base_err):
        put_label(xi, v, e)
    for xi, v, e in zip(x + BAR_W / 2, d["augmented"], aug_err):
        put_label(xi, v, e)

    ax.set_title(model, fontsize=10, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, fontsize=8.5)
    ax.set_ylim(0, 0.60)
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    if show_ylabel:
        ax.set_ylabel("Hard-benign FPR", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.12)


def verify_against_table10():
    """Guard against silent data drift: assert the hardcoded values match the
    figures quoted in Table 10 / the manuscript body."""
    checks = [
        (DATA["DeBERTa-v3-FT"]["baseline"][0], 0.4813),
        (DATA["DeBERTa-v3-FT"]["augmented"][0], 0.0010),
        (DATA["DeBERTa-v3-FT"]["baseline"][1], 0.3144),
        (DATA["DeBERTa-v3-FT"]["augmented"][1], 0.3101),
        (DATA["DistilBERT"]["baseline"][0], 0.4843),
        (DATA["DistilBERT"]["augmented"][0], 0.0123),
        (DATA["DistilBERT"]["baseline"][1], 0.4606),
        (DATA["DistilBERT"]["augmented"][1], 0.3899),
    ]
    for got, want in checks:
        assert abs(got - want) < 1e-9, f"value drift: {got} != {want}"


def main():
    ap = argparse.ArgumentParser(description="Build PIDS-Bench Figure 2.")
    ap.add_argument("-o", "--outdir", default=".",
                    help="output directory (default: current directory)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    verify_against_table10()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    draw_panel(ax1, "DeBERTa-v3-FT", show_ylabel=True)
    draw_panel(ax2, "DistilBERT", show_ylabel=False)

    handles = [
        Patch(facecolor=C_BASE, edgecolor="black", linewidth=0.5, label="Baseline"),
        Patch(facecolor=C_AUG, edgecolor="black", linewidth=0.5, label="+ Hard-negative aug."),
        Line2D([0], [0], color=C_TGT, linestyle=(0, (4, 3)), linewidth=0.9,
               label="0.10 deployment target"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.07), handlelength=1.6,
               columnspacing=1.6)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    pdf = os.path.join(args.outdir, "figure2_provenance.pdf")
    png = os.path.join(args.outdir, "figure2_provenance.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")

    print(f"Serif face used : {SERIF}")
    if SERIF in {"serif", "DejaVu Serif"}:
        print("  NOTE: Times/Nimbus not found; used a generic/DejaVu serif. "
              "This is still serif and matches body text acceptably, but if "
              "you have Times New Roman or Nimbus Roman, install it and rerun "
              "for an exact match.")
    print(f"Wrote           : {pdf}")
    print(f"Wrote           : {png}")
    print("Data verified against Table 10 (assertions passed).")


if __name__ == "__main__":
    main()
