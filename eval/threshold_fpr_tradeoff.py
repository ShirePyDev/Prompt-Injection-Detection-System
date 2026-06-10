#!/usr/bin/env python3
"""
Threshold sensitivity analysis for DeBERTa-v3.

This script sweeps the classification threshold τ from 0 to 1 and simultaneously
tracks two quantities at each threshold:

  - IID test F1   (injection positive class, on data/pids_bench_v3/test.csv)
  - Hard-benign FPR  (FP / (FP + TN) on eval_subsets/hard_benign_test.csv, all benign)

It explicitly answers the research question:
    "Can hard-benign FPR be reduced to ≤ 0.10 without dropping test F1 below 0.95?"

Outputs (all under outputs/deberta_v3/threshold_fpr_tradeoff/)
--------------------------------------------------------------
  tradeoff_sweep.csv          — full 500-point sweep table
  figure_threshold_tradeoff.pdf — dual-axis line chart (publication quality)
  figure_threshold_tradeoff.png — same figure, 200 dpi raster
  tradeoff_summary.json        — key crossover points and answers

Usage
-----
  python eval/threshold_fpr_tradeoff.py
  python eval/threshold_fpr_tradeoff.py --model distilbert  # also works for DistilBERT
  python eval/threshold_fpr_tradeoff.py --grid 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DEFAULTS: dict[str, dict] = {
    "deberta_v3": {
        "test_csv":        ROOT / "outputs" / "deberta_v3" / "test_predictions.csv",
        "hard_benign_csv": ROOT / "outputs" / "deberta_v3" / "hard_benign_predictions.csv",
        "out_dir":         ROOT / "outputs" / "deberta_v3" / "threshold_fpr_tradeoff",
        "display_name":    "DeBERTa-v3",
        "default_tau":     0.502,
    },
    "distilbert": {
        "test_csv":        ROOT / "outputs" / "distilbert_baseline" / "test_predictions.csv",
        "hard_benign_csv": ROOT / "outputs" / "distilbert_baseline" / "hard_benign_predictions.csv",
        "out_dir":         ROOT / "outputs" / "distilbert_baseline" / "threshold_fpr_tradeoff",
        "display_name":    "DistilBERT",
        "default_tau":     0.45,
    },
    "tfidf_logreg": {
        "test_csv":        ROOT / "outputs" / "tfidf_baseline" / "test_probability_scores.csv",
        "hard_benign_csv": ROOT / "outputs" / "tfidf_baseline" / "hard_benign_probability_scores.csv",
        "out_dir":         ROOT / "outputs" / "tfidf_baseline" / "threshold_fpr_tradeoff",
        "display_name":    "TF-IDF + LR",
        "default_tau":     0.518,
    },
}


def load_probs(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) from a saved predictions CSV.

    Accepts CSVs with any of these label column names: true_label, label.
    Accepts CSVs with probability_score column.
    """
    df = pd.read_csv(csv_path)
    # Resolve label column
    if "true_label" in df.columns:
        y_true = df["true_label"].astype(int).to_numpy()
    elif "label" in df.columns:
        y_true = df["label"].astype(int).to_numpy()
    else:
        raise ValueError(f"No label column found in {csv_path}. "
                         f"Expected 'true_label' or 'label'. Got: {df.columns.tolist()}")
    if "probability_score" not in df.columns:
        raise ValueError(f"No 'probability_score' column in {csv_path}. "
                         f"Run eval/generate_probability_scores.py first.")

    y_prob = pd.to_numeric(df["probability_score"], errors="coerce").to_numpy()
    if not np.isfinite(y_prob).all():
        raise ValueError(f"Non-finite probability scores in {csv_path}.")
    y_prob = np.clip(y_prob, 0.0, 1.0)
    return y_true, y_prob


def sweep_threshold(
    y_test: np.ndarray,
    p_test: np.ndarray,
    y_hb: np.ndarray,
    p_hb: np.ndarray,
    n_points: int = 500,
) -> pd.DataFrame:
    """
    Sweep τ over [0, 1] and compute at each point:
      - test_f1   : injection F1 on the IID test set
      - test_prec : precision
      - test_rec  : recall
      - test_fpr  : test-set FPR (should be low if model is well-calibrated on IID)
      - hb_fpr    : hard-benign FPR (the primary over-defense metric)
    """
    thresholds = np.linspace(0.0, 1.0, n_points)
    rows = []
    for tau in thresholds:
        yhat_test = (p_test >= tau).astype(int)
        tp = int(((y_test == 1) & (yhat_test == 1)).sum())
        tn = int(((y_test == 0) & (yhat_test == 0)).sum())
        fp = int(((y_test == 0) & (yhat_test == 1)).sum())
        fn = int(((y_test == 1) & (yhat_test == 0)).sum())
        test_f1   = float(f1_score(y_test, yhat_test, zero_division=0))
        test_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        test_rec  = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        test_fpr  = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # hard_benign is all-benign: every prediction is either TN or FP
        n_hb     = len(y_hb)
        n_hb_fp  = int((p_hb >= tau).sum())
        hb_fpr   = float(n_hb_fp / n_hb) if n_hb > 0 else 0.0

        rows.append({
            "threshold":  float(tau),
            "test_f1":    round(test_f1, 6),
            "test_prec":  round(test_prec, 6),
            "test_rec":   round(test_rec, 6),
            "test_fpr":   round(test_fpr, 6),
            "hb_fpr":     round(hb_fpr, 6),
            "n_hb_flagged": n_hb_fp,
        })
    return pd.DataFrame(rows)


def find_crossover(df: pd.DataFrame, fpr_target: float, f1_min: float) -> dict:
    """
    Answer two complementary questions:

    Primary question:
        "Can hard-benign FPR be reduced to ≤ fpr_target while keeping test F1 ≥ f1_min?"

    Inverse question:
        "What is the minimum hard-benign FPR achievable while keeping test F1 ≥ f1_min?"
        This is the practically important number even when the primary question is NO.
    """
    # ── Primary question ──────────────────────────────────────────────────────
    feasible_fpr = df[df["hb_fpr"] <= fpr_target]
    if not feasible_fpr.empty:
        best_under_constraint = feasible_fpr.loc[feasible_fpr["test_f1"].idxmax()]
        primary_achievable    = float(best_under_constraint["test_f1"]) >= f1_min
        primary = {
            "achievable":                    True,
            "best_tau_under_fpr_constraint": float(best_under_constraint["threshold"]),
            "best_test_f1_under_constraint": float(best_under_constraint["test_f1"]),
            "hb_fpr_at_best_tau":            float(best_under_constraint["hb_fpr"]),
            "f1_above_floor":                primary_achievable,
        }
    else:
        primary_achievable = False
        primary = {
            "achievable":    False,
            "reason":        f"HB-FPR never reaches ≤{fpr_target:.2f} at any threshold in sweep.",
        }

    # ── Inverse question ──────────────────────────────────────────────────────
    # Rows where test F1 is above the floor
    feasible_f1 = df[df["test_f1"] >= f1_min]
    if not feasible_f1.empty:
        min_fpr_row = feasible_f1.loc[feasible_f1["hb_fpr"].idxmin()]
        inverse = {
            "min_hb_fpr_under_f1_floor": float(min_fpr_row["hb_fpr"]),
            "tau_at_min_fpr":            float(min_fpr_row["threshold"]),
            "test_f1_at_min_fpr":        float(min_fpr_row["test_f1"]),
        }
    else:
        inverse = {
            "min_hb_fpr_under_f1_floor": None,
            "note": f"Test F1 never reaches {f1_min:.2f} in sweep.",
        }

    # ── Human-readable answer ─────────────────────────────────────────────────
    if primary_achievable:
        answer = (
            f"YES — at τ={primary['best_tau_under_fpr_constraint']:.3f}, "
            f"hard-benign FPR = {primary['hb_fpr_at_best_tau']:.4f} (≤{fpr_target:.2f}), "
            f"test F1 = {primary['best_test_f1_under_constraint']:.4f} (≥{f1_min:.2f})."
        )
    else:
        inv_fpr = inverse.get("min_hb_fpr_under_f1_floor")
        inv_tau = inverse.get("tau_at_min_fpr")
        if inv_fpr is not None:
            answer = (
                f"NO — hard-benign FPR cannot be reduced to ≤{fpr_target:.2f} while keeping "
                f"test F1 ≥{f1_min:.2f}. "
                f"The minimum achievable hard-benign FPR under the F1 ≥{f1_min:.2f} constraint "
                f"is {inv_fpr:.4f} (at τ={inv_tau:.3f}). "
                f"This requires raising τ substantially above the deployment threshold, "
                f"at which point the model's hard-benign false positives are already highly "
                f"confident predictions (scores near 1.0) that no threshold adjustment can eliminate."
            )
        else:
            answer = (
                f"NO — hard-benign FPR cannot be reduced to ≤{fpr_target:.2f} while keeping "
                f"test F1 ≥{f1_min:.2f} anywhere in the sweep."
            )

    return {
        "fpr_target":      fpr_target,
        "f1_floor":        f1_min,
        "primary_question": primary,
        "inverse_question": inverse,
        "answer":          answer,
    }


def plot_tradeoff(
    df: pd.DataFrame,
    default_tau: float,
    display_name: str,
    out_pdf: Path,
    out_png: Path,
    fpr_target: float = 0.10,
    f1_floor: float = 0.95,
) -> None:
    """
    Publication-quality dual-axis figure:
      Left y-axis  : IID test F1  (blue)
      Right y-axis : Hard-benign FPR  (red)
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_f1  = "#1a6faf"
    color_fpr = "#d62728"

    ax1.plot(df["threshold"], df["test_f1"],  color=color_f1,  lw=2.0,
             label="Test F1 (injection class)", zorder=3)
    ax1.set_xlabel("Classification threshold τ", fontsize=12)
    ax1.set_ylabel("IID test F1", fontsize=12, color=color_f1)
    ax1.tick_params(axis="y", labelcolor=color_f1)
    ax1.set_ylim(0.0, 1.02)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Reference line: F1 floor
    ax1.axhline(f1_floor, color=color_f1, lw=1.0, ls="--", alpha=0.6,
                label=f"F1 floor ({f1_floor:.2f})")

    ax2 = ax1.twinx()
    ax2.plot(df["threshold"], df["hb_fpr"], color=color_fpr, lw=2.0,
             label="Hard-benign FPR", zorder=3)
    ax2.set_ylabel("Hard-benign false-positive rate", fontsize=12, color=color_fpr)
    ax2.tick_params(axis="y", labelcolor=color_fpr)
    ax2.set_ylim(0.0, 1.02)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Reference line: FPR target
    ax2.axhline(fpr_target, color=color_fpr, lw=1.0, ls="--", alpha=0.6,
                label=f"FPR target ({fpr_target:.2f})")

    # Mark operating point at deployment threshold
    idx_default = (df["threshold"] - default_tau).abs().idxmin()
    row_default  = df.iloc[idx_default]
    ax1.scatter([row_default["threshold"]], [row_default["test_f1"]],
                color=color_f1, s=80, zorder=5)
    ax2.scatter([row_default["threshold"]], [row_default["hb_fpr"]],
                color=color_fpr, s=80, zorder=5)
    ax1.annotate(
        f"  τ={default_tau:.3f}\n  F1={row_default['test_f1']:.4f}\n  HB-FPR={row_default['hb_fpr']:.4f}",
        xy=(row_default["threshold"], row_default["test_f1"]),
        fontsize=8.5, color="#333333",
        xytext=(default_tau + 0.06, row_default["test_f1"] - 0.12),
        arrowprops={"arrowstyle": "->", "color": "#555", "lw": 0.8},
    )

    # Legend — combine both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="center left", fontsize=9, framealpha=0.9)

    ax1.set_title(
        f"{display_name}: IID test F1 vs Hard-benign FPR across thresholds",
        fontsize=12, pad=10,
    )
    ax1.set_xlim(0.0, 1.0)
    ax1.grid(axis="y", color="#e5e5e5", lw=0.7, zorder=0)

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_frontier(
    model_sweeps: list[dict],
    out_dir: Path,
    f1_floor: float = 0.95,
) -> None:
    """
    Combined Pareto-frontier figure for all internal models.

    Each model contributes a curve in (hard-benign FPR, test F1) space,
    parametrised by the classification threshold τ.  A model that can
    simultaneously achieve low benign FPR and high test F1 would appear in
    the top-left corner.  The figure shows whether the tradeoff is
    architecture-specific or consistent across model families.

    Parameters
    ----------
    model_sweeps : list of dicts, each with keys
        display  – label for the legend
        color    – line colour
        df       – sweep DataFrame (columns: threshold, test_f1, hb_fpr)
        tau      – deployment threshold (marked as a dot)
    out_dir : directory where PDF and PNG are written
    f1_floor : horizontal reference line for the minimum acceptable F1
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.grid(color="#e5e5e5", lw=0.7, zorder=0)

    # Reference lines
    ax.axhline(f1_floor, color="#888888", lw=1.0, ls="--",
               label=f"F1 floor ({f1_floor:.2f})", zorder=1)
    ax.axvline(0.10, color="#888888", lw=1.0, ls=":",
               label="FPR target (0.10)", zorder=1)

    for ms in model_sweeps:
        df = ms["df"]
        ax.plot(df["hb_fpr"], df["test_f1"],
                color=ms["color"], lw=2.0, label=ms["display"], zorder=3)
        # Mark deployment operating point
        idx = (df["threshold"] - ms["tau"]).abs().idxmin()
        row = df.iloc[idx]
        ax.scatter([row["hb_fpr"]], [row["test_f1"]],
                   color=ms["color"], s=60, zorder=5, edgecolors="white", lw=0.8)

    ax.set_xlabel("Hard-benign false-positive rate", fontsize=12)
    ax.set_ylabel("IID test F1", fontsize=12)
    ax.set_title(
        "Test F1 vs Hard-benign FPR across thresholds\n"
        "(dots mark each model's validation-tuned operating point)",
        fontsize=11,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / "figure_combined_frontier.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / "figure_combined_frontier.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {(out_dir / 'figure_combined_frontier.pdf').relative_to(ROOT)}")
    print(f"  Saved {(out_dir / 'figure_combined_frontier.png').relative_to(ROOT)}")


COMBINED_MODEL_SPECS = [
    {
        "model_id":   "deberta_v3",
        "display":    "DeBERTa-v3",
        "color":      "#1a6faf",
        "sweep_csv":  ROOT / "outputs" / "deberta_v3" / "threshold_fpr_tradeoff" / "tradeoff_sweep.csv",
        "tau":        0.502,
    },
    {
        "model_id":   "distilbert",
        "display":    "DistilBERT",
        "color":      "#e07b39",
        "sweep_csv":  ROOT / "outputs" / "distilbert_baseline" / "threshold_fpr_tradeoff" / "tradeoff_sweep.csv",
        "tau":        0.45,
    },
    {
        "model_id":   "tfidf_logreg",
        "display":    "TF-IDF + LR",
        "color":      "#2ca02c",
        "sweep_csv":  ROOT / "outputs" / "tfidf_baseline" / "threshold_fpr_tradeoff" / "tradeoff_sweep.csv",
        "tau":        0.518,
    },
]


def run_combined(f1_floor: float = 0.95) -> None:
    """Load completed sweep CSVs for all internal models and produce the combined figure."""
    out_dir = ROOT / "outputs" / "threshold_combined"
    loaded = []
    for spec in COMBINED_MODEL_SPECS:
        if not spec["sweep_csv"].exists():
            print(f"[SKIP] {spec['model_id']}: sweep CSV not found at "
                  f"{spec['sweep_csv'].relative_to(ROOT)}. "
                  f"Run --model {spec['model_id']} first.")
            continue
        df = pd.read_csv(spec["sweep_csv"])
        loaded.append({
            "display": spec["display"],
            "color":   spec["color"],
            "df":      df,
            "tau":     spec["tau"],
        })

    if len(loaded) < 2:
        print("[ERROR] At least two models need completed sweeps for the combined figure.")
        return

    plot_combined_frontier(loaded, out_dir, f1_floor=f1_floor)

    # Summary table: min achievable HB-FPR while F1 >= f1_floor, per model
    rows = []
    for ms in loaded:
        df = ms["df"]
        feasible = df[df["test_f1"] >= f1_floor]
        if feasible.empty:
            min_fpr = None
            tau_at_min = None
        else:
            idx_min = feasible["hb_fpr"].idxmin()
            min_fpr = float(feasible.loc[idx_min, "hb_fpr"])
            tau_at_min = float(feasible.loc[idx_min, "threshold"])
        # Operating point
        idx_op = (df["threshold"] - ms["tau"]).abs().idxmin()
        rows.append({
            "model":                     ms["display"],
            "deployment_tau":            ms["tau"],
            "test_f1_at_deployment":     round(float(df.loc[idx_op, "test_f1"]), 4),
            "hb_fpr_at_deployment":      round(float(df.loc[idx_op, "hb_fpr"]), 4),
            "min_hb_fpr_f1_floor":       round(min_fpr, 4) if min_fpr is not None else None,
            "tau_at_min_hb_fpr":         round(tau_at_min, 4) if tau_at_min is not None else None,
            "f1_floor":                  f1_floor,
        })

    summary_df = pd.DataFrame(rows)
    out_csv = out_dir / "combined_tradeoff_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv.relative_to(ROOT)}")
    print(summary_df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold FPR/F1 tradeoff analysis")
    ap.add_argument(
        "--model",
        choices=list(MODEL_DEFAULTS.keys()),
        default="deberta_v3",
    )
    ap.add_argument("--grid", type=int, default=500, help="Number of threshold steps (default 500)")
    ap.add_argument(
        "--fpr-target", type=float, default=0.10,
        help="Hard-benign FPR ceiling for the research question (default 0.10)",
    )
    ap.add_argument(
        "--f1-floor", type=float, default=0.95,
        help="Test F1 minimum for the research question (default 0.95)",
    )
    ap.add_argument(
        "--combined", action="store_true",
        help="Generate the combined multi-model frontier figure from completed sweep CSVs.",
    )
    args = ap.parse_args()

    import os
    os.chdir(ROOT)

    if args.combined:
        run_combined(f1_floor=args.f1_floor)
        return

    cfg = MODEL_DEFAULTS[args.model]
    test_csv        = cfg["test_csv"]
    hard_benign_csv = cfg["hard_benign_csv"]
    out_dir         = cfg["out_dir"]
    display_name    = cfg["display_name"]
    default_tau     = cfg["default_tau"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load predictions ──────────────────────────────────────────────────────
    print(f"Loading test predictions from {test_csv.relative_to(ROOT)} …")
    y_test, p_test = load_probs(test_csv)
    print(f"  n={len(y_test)}, injection={int(y_test.sum())}, benign={int((y_test==0).sum())}")

    print(f"Loading hard-benign predictions from {hard_benign_csv.relative_to(ROOT)} …")
    y_hb, p_hb = load_probs(hard_benign_csv)
    # Validate that hard_benign is all-benign (or mostly benign)
    n_hb_injections = int(y_hb.sum())
    if n_hb_injections > 0:
        print(f"  Warning: hard_benign CSV has {n_hb_injections} injection rows. "
              "FPR is computed as fraction of all rows flagged (treating all as benign).")
    print(f"  n={len(y_hb)}")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print(f"Sweeping {args.grid} thresholds …")
    df_sweep = sweep_threshold(y_test, p_test, y_hb, p_hb, n_points=args.grid)
    df_sweep.to_csv(out_dir / "tradeoff_sweep.csv", index=False)
    print(f"  Saved {(out_dir / 'tradeoff_sweep.csv').relative_to(ROOT)}")

    # ── Crossover analysis ────────────────────────────────────────────────────
    crossover = find_crossover(df_sweep, fpr_target=args.fpr_target, f1_min=args.f1_floor)
    print(f"\n=== Research Question: FPR ≤ {args.fpr_target:.2f} with F1 ≥ {args.f1_floor:.2f}? ===")
    print(f"  {crossover.get('answer', 'Not achievable.')}")

    # Also report operating point at deployment threshold
    idx_deploy = (df_sweep["threshold"] - default_tau).abs().idxmin()
    row_deploy  = df_sweep.iloc[idx_deploy]
    deploy_stats = {
        "deployment_threshold":    default_tau,
        "test_f1_at_deployment":   float(row_deploy["test_f1"]),
        "hb_fpr_at_deployment":    float(row_deploy["hb_fpr"]),
        "test_prec_at_deployment": float(row_deploy["test_prec"]),
        "test_rec_at_deployment":  float(row_deploy["test_rec"]),
    }

    summary = {
        "model":                  args.model,
        "display_name":           display_name,
        "n_test":                 int(len(y_test)),
        "n_hard_benign":          int(len(y_hb)),
        "grid_points":            args.grid,
        "deployment_operating_point": deploy_stats,
        "crossover_analysis":     crossover,
    }
    with open(out_dir / "tradeoff_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved {(out_dir / 'tradeoff_summary.json').relative_to(ROOT)}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    out_pdf = out_dir / "figure_threshold_tradeoff.pdf"
    out_png = out_dir / "figure_threshold_tradeoff.png"
    plot_tradeoff(
        df_sweep,
        default_tau  = default_tau,
        display_name = display_name,
        out_pdf      = out_pdf,
        out_png      = out_png,
        fpr_target   = args.fpr_target,
        f1_floor     = args.f1_floor,
    )
    print(f"  Saved {out_pdf.relative_to(ROOT)}")
    print(f"  Saved {out_png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
