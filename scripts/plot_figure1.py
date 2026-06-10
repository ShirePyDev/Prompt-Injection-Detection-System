import numpy as np, pandas as pd, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS=[13,42,123,2024,7777]
MODELS=[
    ("TF-IDF + LR",   "outputs/multi_seed_runs/tfidf",                 "#2ca02c", None),
    ("DistilBERT",    "outputs/multi_seed_runs/distilbert_cluster",    "#ff7f0e", "s"),
    ("DeBERTa-v3-FT", "outputs/multi_seed_runs/deberta_baseline_clean","#1f77b4", "o"),
]
fig, ax = plt.subplots(figsize=(3.6,3.2), dpi=300)
for name, base, color, marker in MODELS:
    curves=[]; taus=None
    for s in SEEDS:
        df=pd.read_csv(f"{base}/seed_{s}/threshold_sweep.csv").sort_values("tau")
        if taus is None: taus=df["tau"].values
        f1=np.interp(taus, df["tau"].values, df["test_F1"].values)
        fpr=np.interp(taus, df["tau"].values, df["hb_FPR_agg"].values)
        curves.append((fpr,f1))
    fprs=np.array([c[0] for c in curves]); f1s=np.array([c[1] for c in curves])
    fpr_mean=fprs.mean(axis=0); f1_mean=f1s.mean(axis=0); fpr_std=fprs.std(axis=0, ddof=1)
    ax.plot(fpr_mean, f1_mean, color=color, lw=1.6, label=name, zorder=3)
    ax.fill_betweenx(f1_mean, fpr_mean-fpr_std, fpr_mean+fpr_std, color=color, alpha=0.15, zorder=1)
    if marker is not None:
        for s in SEEDS:
            try:
                j=json.load(open(f"{base}/seed_{s}/summary.json"))
                t=j.get("val_tuned_threshold", j.get("tuned_threshold"))
            except Exception:
                t=None
            df=pd.read_csv(f"{base}/seed_{s}/threshold_sweep.csv").sort_values("tau")
            if t is not None:
                x=np.interp(t, df["tau"].values, df["hb_FPR_agg"].values)
                y=np.interp(t, df["tau"].values, df["test_F1"].values)
                ax.plot(x,y, marker=marker, color=color, markersize=4, markeredgecolor="white",
                        markeredgewidth=0.4, linestyle="None", zorder=4)
ax.axvline(0.10, color="0.5", ls="--", lw=0.8, label="FPR = 0.10")
ax.axhline(0.95, color="0.3", ls="--", lw=0.8, label="F1 = 0.95")
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xlabel("Hard-benign FPR (false-positive rate)", fontsize=8)
ax.set_ylabel("IID test F1", fontsize=8)
ax.tick_params(labelsize=7); ax.grid(True, alpha=0.25, lw=0.5)
ax.legend(loc="lower left", fontsize=6, framealpha=0.9)
plt.tight_layout()
plt.savefig("outputs/figure1_threshold_sweep.png", bbox_inches="tight")
print("Saved outputs/figure1_threshold_sweep.png")
