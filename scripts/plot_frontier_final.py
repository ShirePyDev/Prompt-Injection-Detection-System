"""Final Figure 1: threshold frontier from VERIFIED 5-seed cluster sweeps,
styled to IEEE figure conventions (markers + reference lines + no title)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("outputs/multi_seed_runs")
SEEDS = [13, 42, 123, 2024, 7777]

# (label, dir, sweep_filename, summary_dir_for_taus, color, marker)
MODELS = [
    ("TF-IDF + LR",   ROOT/"tfidf",
        "threshold_sweep_full.csv", "#2ca02c", "^"),
    ("DistilBERT",    ROOT/"distilbert_cluster",
        "threshold_sweep_full.csv", "#ff7f0e", "s"),
    ("DeBERTa-v3-FT", ROOT/"deberta_baseline_clean",
        "threshold_sweep.csv",      "#1f77b4", "o"),
]

fig, ax = plt.subplots(figsize=(3.6, 2.9), dpi=300)
grid = np.round(np.arange(0.0, 1.0001, 0.001), 3)

for label, d, fname, color, marker in MODELS:
    f1s, fprs, op_pts = [], [], []
    for s in SEEDS:
        df = pd.read_csv(d/f"seed_{s}"/fname)
        df = df.set_index("tau").reindex(grid).interpolate().reset_index()
        f1s.append(df["test_F1"].values)
        fprs.append(df["hb_FPR_agg"].values)
        # operating point at this seed's validation-tuned tau (from summary.json)
        try:
            j = json.load(open(d/f"seed_{s}"/"summary.json"))
            tau_op = j["val_tuned_threshold"]
            row = df.iloc[(df["tau"] - tau_op).abs().idxmin()]
            op_pts.append((float(row["hb_FPR_agg"]), float(row["test_F1"])))
        except Exception:
            pass
    f1 = np.mean(f1s, axis=0)
    fpr_mean = np.mean(fprs, axis=0)
    fpr_std  = np.std(fprs, axis=0)
    ax.plot(fpr_mean, f1, color=color, label=label, lw=1.5)
    ax.fill_betweenx(f1, fpr_mean-fpr_std, fpr_mean+fpr_std, color=color, alpha=0.15)
    # operating-point markers (one per seed)
    for x, y in op_pts:
        ax.plot(x, y, marker=marker, color=color, markersize=4,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)

# reference lines + labels
ax.axvline(0.10, ls="--", color="0.35", lw=0.8)
ax.axhline(0.95, ls="--", color="0.35", lw=0.8)
ax.text(0.11, 0.05, "FPR = 0.10", fontsize=6, color="0.3")
ax.text(0.55, 0.955, "F1 = 0.95", fontsize=6, color="0.3")

ax.set_xlabel("Aggregate hard-benign FPR", fontsize=8)
ax.set_ylabel(r"Test $F_1$", fontsize=8)
ax.set_xlim(0, 1.0); ax.set_ylim(0.0, 1.02)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.5, loc="lower right", frameon=False)
ax.grid(True, alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig("outputs/threshold_sweep.pdf", bbox_inches="tight")
print("saved -> outputs/threshold_sweep.pdf")
