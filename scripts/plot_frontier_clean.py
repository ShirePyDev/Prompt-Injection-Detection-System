"""Regenerate threshold_sweep.pdf from VERIFIED 5-seed cluster sweeps.
Reads the actual clean sweep files; plots mean F1-vs-aggregate-hb-FPR
with +/-1 std band per architecture; marks the forbidden region."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("outputs/multi_seed_runs")
SEEDS = [13, 42, 123, 2024, 7777]
# (label, dir, sweep filename, color)
MODELS = [
    ("TF-IDF + LR",   ROOT/"tfidf",                 "threshold_sweep_full.csv", "#888888"),
    ("DistilBERT",    ROOT/"distilbert_cluster",    "threshold_sweep_full.csv", "#1f77b4"),
    ("DeBERTa-v3-FT", ROOT/"deberta_baseline_clean","threshold_sweep.csv",      "#d62728"),
]

fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)

# common tau grid for averaging
grid = np.round(np.arange(0.0, 1.0001, 0.001), 3)

for label, d, fname, color in MODELS:
    f1s, fprs = [], []
    for s in SEEDS:
        df = pd.read_csv(d/f"seed_{s}"/fname)
        # align to grid on tau
        df = df.set_index("tau").reindex(grid).interpolate().reset_index()
        f1s.append(df["test_F1"].values)
        fprs.append(df["hb_FPR_agg"].values)
    f1 = np.mean(f1s, axis=0)
    fpr_mean = np.mean(fprs, axis=0)
    fpr_std  = np.std(fprs, axis=0)
    # plot F1 (y) vs aggregate hb-FPR (x): trace the curve
    ax.plot(fpr_mean, f1, color=color, label=label, lw=1.4)
    ax.fill_betweenx(f1, fpr_mean-fpr_std, fpr_mean+fpr_std, color=color, alpha=0.15)

# forbidden-region reference lines
ax.axvline(0.10, ls="--", color="black", lw=0.8)
ax.axhline(0.95, ls="--", color="black", lw=0.8)
ax.set_xlabel("Aggregate hard-benign FPR")
ax.set_ylabel(r"Test $F_1$")
ax.set_xlim(0, 1); ax.set_ylim(0.5, 1.0)
ax.legend(fontsize=6, loc="lower right")
ax.text(0.02, 0.965, r"target region ($F_1\geq0.95,\ \mathrm{FPR}\leq0.10$)",
        fontsize=5, color="black")
fig.tight_layout()
fig.savefig("outputs/threshold_sweep.pdf", bbox_inches="tight")
print("saved -> outputs/threshold_sweep.pdf")
print("Verify: no curve should enter the upper-left box (FPR<=0.10 AND F1>=0.95).")
