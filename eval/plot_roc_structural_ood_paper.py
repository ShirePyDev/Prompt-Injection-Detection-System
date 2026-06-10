#!/usr/bin/env python3
"""
Generate the structural-OOD ROC curve figure for the IEEE Access paper.

Three models only (those with calibrated probability scores):
  - DeBERTa-v3
  - DistilBERT
  - TF-IDF + LR

Output
------
  outputs/roc_pr_curves/roc_structural_ood.png   (300 DPI, 3.5 × 3.0 in)
  outputs/roc_pr_curves/roc_structural_ood.pdf

Usage
-----
  python eval/plot_roc_structural_ood_paper.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs" / "roc_pr_curves"

# ── Model specs ────────────────────────────────────────────────────────────────
MODELS = [
    {
        "label":    "DeBERTa-v3",
        "csv":      ROOT / "outputs" / "deberta_v3" / "structural_ood_probability_scores.csv",
        "color":    "#1a6faf",
        "ls":       "-",
        "lw":       1.6,
    },
    {
        "label":    "DistilBERT",
        "csv":      ROOT / "outputs" / "distilbert_baseline" / "structural_ood_probability_scores.csv",
        "color":    "#e07b39",
        "ls":       "--",
        "lw":       1.6,
    },
    {
        "label":    "TF-IDF + LR",
        "csv":      ROOT / "outputs" / "tfidf_baseline" / "structural_ood_probability_scores.csv",
        "color":    "#2ca02c",
        "ls":       "-.",
        "lw":       1.6,
    },
]


def load_probs(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "probability_score" not in df.columns:
        raise ValueError(f"No 'probability_score' column in {csv_path}")
    y_prob = pd.to_numeric(df["probability_score"], errors="coerce").to_numpy()
    if not np.isfinite(y_prob).all():
        raise ValueError(f"Non-finite probability scores in {csv_path}")
    y_prob = np.clip(y_prob, 0.0, 1.0)

    for col in ("true_label", "label"):
        if col in df.columns:
            y_true = df[col].astype(int).to_numpy()
            return y_true, y_prob
    raise ValueError(f"No label column found in {csv_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Single-column IEEE figure: 3.5 × 3.0 inches ──────────────────────────
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Random baseline
    ax.plot(
        [0, 1], [0, 1],
        lw=1.0, ls=":", color="#999999",
        label="Random (AUC = 0.500)",
        zorder=1,
    )

    for spec in MODELS:
        y_true, y_prob = load_probs(spec["csv"])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = float(roc_auc_score(y_true, y_prob))
        print(f"  {spec['label']:20s}  AUC = {auc:.4f}")
        ax.plot(
            fpr, tpr,
            color=spec["color"],
            ls=spec["ls"],
            lw=spec["lw"],
            label=f"{spec['label']} (AUC = {auc:.3f})",
            zorder=2,
        )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate", fontsize=7)
    ax.set_ylabel("True Positive Rate", fontsize=7)
    ax.set_title("ROC Curves — Structural OOD Split", fontsize=7.5, pad=4)
    ax.tick_params(labelsize=6)

    # No gridlines, clean frame
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    # ── Legend ────────────────────────────────────────────────────────────────
    leg = ax.legend(
        loc="lower right",
        fontsize=5.8,
        framealpha=0.9,
        edgecolor="#cccccc",
        handlelength=2.0,
        labelspacing=0.3,
    )
    leg.get_frame().set_linewidth(0.5)

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.tight_layout(pad=0.5)

    out_png = OUT_DIR / "roc_structural_ood.png"
    out_pdf = OUT_DIR / "roc_structural_ood.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_png.relative_to(ROOT)}")
    print(f"Saved: {out_pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
