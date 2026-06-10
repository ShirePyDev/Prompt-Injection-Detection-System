# eval/calibration_analysis.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"true_label", "probability_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["true_label"] = df["true_label"].astype(int)
    df["probability_score"] = pd.to_numeric(df["probability_score"], errors="coerce")
    df = df.dropna(subset=["probability_score"]).copy()

    uniq = set(df["true_label"].unique())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"true_label must be 0/1; got {sorted(uniq)}")

    # clamp probs to avoid log(0) in log_loss
    eps = 1e-12
    df["probability_score"] = df["probability_score"].clip(eps, 1 - eps)

    return df


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """
    Returns per-bin: mean predicted prob (conf), empirical accuracy, and bin counts
    using equal-width bins over [0,1].
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        m = bin_ids == b
        if not np.any(m):
            conf.append(np.nan)
            acc.append(np.nan)
            cnt.append(0)
            continue
        conf.append(float(np.mean(y_prob[m])))
        acc.append(float(np.mean(y_true[m])))
        cnt.append(int(np.sum(m)))

    return np.array(conf), np.array(acc), np.array(cnt)


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    conf, acc, cnt = calibration_bins(y_true, y_prob, n_bins=n_bins)
    n = np.sum(cnt)
    ece = 0.0
    for c, a, k in zip(conf, acc, cnt):
        if k == 0:
            continue
        ece += (k / n) * abs(a - c)
    return float(ece)


def fit_platt(val_prob: np.ndarray, val_y: np.ndarray):
    """
    Platt scaling on probabilities using logit(p) as a single feature.
    This does not require logits from the model.
    """
    eps = 1e-12
    x = np.log(val_prob.clip(eps, 1 - eps) / (1 - val_prob.clip(eps, 1 - eps)))
    X = x.reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(X, val_y)

    def transform(p: np.ndarray) -> np.ndarray:
        x2 = np.log(p.clip(eps, 1 - eps) / (1 - p.clip(eps, 1 - eps)))
        P = lr.predict_proba(x2.reshape(-1, 1))[:, 1]
        return P.clip(eps, 1 - eps)

    return transform


def fit_isotonic(val_prob: np.ndarray, val_y: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_prob, val_y)

    def transform(p: np.ndarray) -> np.ndarray:
        eps = 1e-12
        P = iso.predict(p)
        return np.asarray(P).clip(eps, 1 - eps)

    return transform


def plot_reliability(y_true, y_prob, out_png: Path, n_bins: int, title: str):
    conf, acc, cnt = calibration_bins(y_true, y_prob, n_bins=n_bins)

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(conf, acc, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def compute_metrics(y_true, y_prob, n_bins: int):
    return {
        "ece": ece_score(y_true, y_prob, n_bins=n_bins),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", type=str, default="outputs/distilbert_baseline/val_predictions.csv")
    ap.add_argument("--test_csv", type=str, default="outputs/distilbert_baseline/test_predictions.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/distilbert_baseline/calibration_analysis")
    ap.add_argument("--method", choices=["platt", "isotonic"], default="platt")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_val = load_preds(Path(args.val_csv))
    df_test = load_preds(Path(args.test_csv))

    yv = df_val["true_label"].to_numpy()
    pv = df_val["probability_score"].to_numpy()

    yt = df_test["true_label"].to_numpy()
    pt = df_test["probability_score"].to_numpy()

    # Fit calibrator ONLY on validation
    if args.method == "platt":
        calibrate = fit_platt(pv, yv)
    else:
        calibrate = fit_isotonic(pv, yv)

    pt_cal = calibrate(pt)

    # Metrics on test: before vs after
    before = compute_metrics(yt, pt, n_bins=args.bins)
    after = compute_metrics(yt, pt_cal, n_bins=args.bins)

    # Save reliability plots (test)
    plot_reliability(yt, pt, out_dir / "reliability_before.png", args.bins,
                     title=f"Reliability (Test) - Before ({args.method})")
    plot_reliability(yt, pt_cal, out_dir / "reliability_after.png", args.bins,
                     title=f"Reliability (Test) - After ({args.method})")

    # Save calibrated test predictions
    df_out = df_test.copy()
    df_out["probability_calibrated"] = pt_cal
    df_out.to_csv(out_dir / "test_predictions_calibrated.csv", index=False)

    # Summary
    summary = {
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "method": args.method,
        "bins": int(args.bins),
        "test_ece_before": before["ece"],
        "test_ece_after": after["ece"],
        "test_brier_before": before["brier"],
        "test_brier_after": after["brier"],
        "test_logloss_before": before["log_loss"],
        "test_logloss_after": after["log_loss"],
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

    print("Saved:")
    print(" -", out_dir / "reliability_before.png")
    print(" -", out_dir / "reliability_after.png")
    print(" -", out_dir / "test_predictions_calibrated.csv")
    print(" -", out_dir / "summary.csv")
    print("\nSummary:", summary)


if __name__ == "__main__":
    main()