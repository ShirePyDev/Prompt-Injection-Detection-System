import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
})

# Verified values from Table (tab:ablation): baseline vs +HardNeg, per subset
# DeBERTa-v3-FT and DistilBERT, curated vs externally-sourced hard-benign FPR
data = {
    "DeBERTa-v3-FT": {
        "curated":       (0.4813, 0.0010),   # (baseline, +HardNeg)
        "ext-sourced":   (0.3144, 0.3101),
    },
    "DistilBERT": {
        "curated":       (0.4843, 0.0123),
        "ext-sourced":   (0.4606, 0.3899),
    },
}

fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)

col_base = "#B0B0B0"   # baseline (grey)
col_aug  = "#4C72B0"   # +HardNeg (blue)

for ax, (model, subsets) in zip(axes, data.items()):
    labels = ["Curated", "Externally sourced"]
    base = [subsets["curated"][0], subsets["ext-sourced"][0]]
    aug  = [subsets["curated"][1], subsets["ext-sourced"][1]]

    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w/2, base, w, label="Baseline", color=col_base, edgecolor="black", lw=0.5)
    ax.bar(x + w/2, aug,  w, label="+ Hard-negative aug.", color=col_aug, edgecolor="black", lw=0.5)

    # value labels on bars
    for xi, b, a in zip(x, base, aug):
        ax.text(xi - w/2, b + 0.012, f"{b:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + w/2, a + 0.012, f"{a:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_title(model, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.58)
    ax.axhline(0.10, color="0.4", ls="--", lw=0.8, zorder=0)  # 0.10 target reference

axes[0].set_ylabel("Hard-benign FPR")
axes[0].legend(loc="upper right", fontsize=8)
# annotate the 0.10 target once
axes[0].text(1.48, 0.115, "0.10 target", fontsize=7, color="0.4", ha="right")

fig.tight_layout(pad=0.5)
fig.savefig("figure2_provenance.pdf", bbox_inches="tight", dpi=300)
fig.savefig("figure2_provenance.png", bbox_inches="tight", dpi=300)
print("saved figure2_provenance.pdf / .png")
