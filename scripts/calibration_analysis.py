#!/usr/bin/env python3
"""
Calibration analysis for the internal detectors (Reviewer 3 #4).

R3 asked us to separate calibration from discrimination: report Brier score,
expected calibration error (ECE), reliability curves, and the effect of
temperature scaling. The deeper question behind the request is whether the
over-defense is merely a calibration artifact -- i.e. whether recalibrating the
scores would remove the hard-benign over-blocking. This script answers that
directly.

DESIGN
  * Temperature scaling fits a single scalar T on the VALIDATION split (by
    minimising validation negative log-likelihood) and applies it to test and to
    the hard-benign set. Fitting on validation and evaluating elsewhere keeps the
    recalibration leakage-free.
  * We report, before and after temperature scaling:
      - Brier score and ECE on IID test (calibration of the scores)
      - reliability-curve data (confidence bin -> empirical accuracy)
      - the fitted temperature per seed
      - hard-benign FPR at tau=0.5 (does recalibration fix over-defense?)
  * ECE is binning-dependent; we use 15 equal-width bins and state it.

  Probabilities here are the model's stored probability_score. Temperature scaling
  operates on logits; we recover a logit as logit = ln(p/(1-p)) (clipped), scale
  by 1/T, and map back. This is the standard Platt-style temperature scaling for
  a binary score.

OUTPUT
  outputs/resub_analysis/calibration_summary.csv
  outputs/resub_analysis/calibration_reliability.csv   (for the reliability plot)

USAGE
  python scripts/calibration_analysis.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REPO = Path("/home/M2026206/khalid/Prompt-Injection-Detector-System")
OUT_DIR = REPO / "outputs" / "resub_analysis"
HB_FILE = REPO / "data" / "pids_bench_v3" / "eval_subsets" / "hard_benign_test.csv"
SEEDS = ["13", "42", "123", "2024", "7777"]
N_BINS = 15
EPS = 1e-6
TAU = 0.5

MODELS = {
    "DeBERTa-v3-FT": "outputs/multi_seed_runs/deberta_baseline_clean",
    "DistilBERT":    "outputs/multi_seed_runs/distilbert",
}


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def apply_temperature(p, T):
    return 1 / (1 + np.exp(-to_logit(p) / T))


def fit_temperature(p_val, y_val):
    """Fit T>0 minimising validation NLL of the recalibrated probabilities."""
    z = to_logit(p_val)
    def nll(T):
        if T <= 0:
            return 1e9
        q = 1 / (1 + np.exp(-z / T))
        q = np.clip(q, EPS, 1 - EPS)
        return -np.mean(y_val * np.log(q) + (1 - y_val) * np.log(1 - q))
    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def ece(p, y, n_bins=N_BINS):
    """Expected calibration error, equal-width bins on confidence."""
    # confidence = distance from 0.5 toward the predicted class
    conf = np.where(p >= 0.5, p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / len(p)) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability_rows(p, y, model, cond, n_bins=N_BINS):
    conf = np.where(p >= 0.5, p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        out.append({"model": model, "condition": cond,
                    "bin_lo": bins[i], "bin_hi": bins[i + 1],
                    "mean_conf": float(conf[m].mean()),
                    "empirical_acc": float(correct[m].mean()),
                    "n": int(m.sum())})
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hb = pd.read_csv(HB_FILE)
    summary, reliab = [], []

    for model, root in MODELS.items():
        print(f"\n{'='*72}\n{model}\n{'='*72}")
        Ts, briers0, briers1, eces0, eces1, hb0, hb1 = [], [], [], [], [], [], []
        # for the reliability plot, accumulate test scores across seeds (raw + scaled)
        rel_raw_p, rel_raw_y, rel_cal_p = [], [], []

        for s in SEEDS:
            sd = REPO / root / f"seed_{s}"
            val = pd.read_csv(sd / "val_predictions.csv")
            test = pd.read_csv(sd / "test_predictions.csv")
            hbp = pd.read_csv(sd / "hard_benign_predictions.csv")

            pv, yv = val["probability_score"].values, val["true_label"].values
            pt, yt = test["probability_score"].values, test["true_label"].values
            ph = hbp["probability_score"].values   # all benign (y=0)

            T = fit_temperature(pv, yv)
            Ts.append(T)

            # IID calibration before/after
            briers0.append(brier(pt, yt));     briers1.append(brier(apply_temperature(pt, T), yt))
            eces0.append(ece(pt, yt));         eces1.append(ece(apply_temperature(pt, T), yt))

            # hard-benign FPR at tau=0.5 before/after temperature scaling
            hb0.append((ph >= TAU).mean())
            hb1.append((apply_temperature(ph, T) >= TAU).mean())

            rel_raw_p.append(pt); rel_raw_y.append(yt)
            rel_cal_p.append(apply_temperature(pt, T))

        def ms(a): return np.mean(a), np.std(a, ddof=1)
        print(f"  fitted temperature T   : {ms(Ts)[0]:.3f} +/- {ms(Ts)[1]:.3f}  "
              f"(T!=1 indicates miscalibration)")
        print(f"  Brier  (IID)  raw->cal : {ms(briers0)[0]:.4f} -> {ms(briers1)[0]:.4f}")
        print(f"  ECE    (IID)  raw->cal : {ms(eces0)[0]:.4f} -> {ms(eces1)[0]:.4f}")
        print(f"  hard-benign FPR @0.5   : {ms(hb0)[0]:.4f} -> {ms(hb1)[0]:.4f}  "
              f"(does recalibration fix over-defense?)")

        summary.append({
            "model": model,
            "T_mean": ms(Ts)[0], "T_std": ms(Ts)[1],
            "brier_raw": ms(briers0)[0], "brier_cal": ms(briers1)[0],
            "ece_raw": ms(eces0)[0], "ece_cal": ms(eces1)[0],
            "hb_fpr_raw": ms(hb0)[0], "hb_fpr_cal": ms(hb1)[0],
        })

        # reliability rows from pooled test scores (raw and calibrated)
        reliab += reliability_rows(np.concatenate(rel_raw_p), np.concatenate(rel_raw_y),
                                   model, "raw")
        reliab += reliability_rows(np.concatenate(rel_cal_p), np.concatenate(rel_raw_y),
                                   model, "temperature_scaled")

    pd.DataFrame(summary).to_csv(OUT_DIR / "calibration_summary.csv", index=False)
    pd.DataFrame(reliab).to_csv(OUT_DIR / "calibration_reliability.csv", index=False)
    print(f"\nWrote:\n  {OUT_DIR/'calibration_summary.csv'}"
          f"\n  {OUT_DIR/'calibration_reliability.csv'}  (reliability-plot data)")
    print("\nHonest reading: temperature scaling is expected to improve IID ECE/Brier")
    print("(better calibrated scores) while leaving hard-benign FPR essentially")
    print("unchanged -- because scaling is monotonic and the over-blocked rows are")
    print("scored deep in the injection region. If so, over-defense is NOT a")
    print("calibration artifact. Report whatever the numbers actually show.")


if __name__ == "__main__":
    main()
