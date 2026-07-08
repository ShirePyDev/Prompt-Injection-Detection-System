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

# Verified externally-sourced hard-benign FPR by pool size (tau=0.5, 5-seed mean +/- std)
# Pool-size runs are independent trainings from the main augmentation experiment.
pool_sizes = [100, 200, 300, 419]

deberta_mean = [0.340, 0.274, 0.262, 0.321]
deberta_std  = [0.056, 0.035, 0.065, 0.056]

distil_mean  = [0.351, 0.342, 0.378, 0.390]
distil_std   = [0.059, 0.059, 0.024, 0.055]

fig, ax = plt.subplots(figsize=(4.2, 3.0))

col_deb = "#4C72B0"   # DeBERTa (blue)
col_dis = "#DD8452"   # DistilBERT (orange)

x = np.arange(len(pool_sizes))

ax.errorbar(x, deberta_mean, yerr=deberta_std, marker="o", ms=4, lw=1.2,
            color=col_deb, capsize=3, label="DeBERTa-v3-FT")
ax.errorbar(x, distil_mean, yerr=distil_std, marker="s", ms=4, lw=1.2,
            color=col_dis, capsize=3, label="DistilBERT")

ax.axhline(0.10, color="0.4", ls="--", lw=0.8, zorder=0)
ax.text(len(pool_sizes)-1, 0.115, "0.10 target", fontsize=7, color="0.4", ha="right")

ax.set_xticks(x)
ax.set_xticklabels(pool_sizes)
ax.set_xlabel("Hard-negative pool size")
ax.set_ylabel("Externally-sourced hard-benign FPR")
ax.set_ylim(0, 0.50)
ax.legend(loc="upper left", fontsize=8)

fig.tight_layout(pad=0.5)
fig.savefig("figure3_poolsweep.pdf", bbox_inches="tight", dpi=300)
fig.savefig("figure3_poolsweep.png", bbox_inches="tight", dpi=300)
print("saved figure3_poolsweep.pdf / .png")
