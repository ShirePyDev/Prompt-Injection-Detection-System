import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)

def load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"true_label", "probability_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    # Ensure numeric + clean
    df["true_label"] = df["true_label"].astype(int)
    df["probability_score"] = pd.to_numeric(df["probability_score"], errors="coerce")
    df = df.dropna(subset=["probability_score"]).copy()

    # sanity: labels must be 0/1
    uniq = set(df["true_label"].unique())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"true_label must be 0/1, got: {sorted(uniq)}")

    # Keep probabilities in a safe numeric range.
    df["probability_score"] = df["probability_score"].clip(0.0, 1.0)

    return df

def compute_threshold_table(y_true: np.ndarray, y_prob: np.ndarray, n=501) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, n)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        })
    return pd.DataFrame(rows)

def save_roc(y_true, y_prob, out_png: Path, label: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"{label} AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return float(auc)

def save_pr(y_true, y_prob, out_png: Path, label: str) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision, label=f"{label} AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return float(ap)

def save_threshold_curves(tbl: pd.DataFrame, out_png: Path, title: str) -> None:
    plt.figure()
    plt.plot(tbl["threshold"], tbl["precision"], label="Precision")
    plt.plot(tbl["threshold"], tbl["recall"], label="Recall")
    plt.plot(tbl["threshold"], tbl["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", type=str,
                    default="outputs/distilbert_baseline/val_predictions.csv")
    ap.add_argument("--test_csv", type=str,
                    default="outputs/distilbert_baseline/test_predictions.csv")
    ap.add_argument("--out_dir", type=str,
                    default="outputs/distilbert_baseline/threshold_analysis")
    ap.add_argument("--grid", type=int, default=501,
                    help="Number of threshold points from 0..1")
    args = ap.parse_args()

    val_csv = Path(args.val_csv)
    test_csv = Path(args.test_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_val = load_predictions(val_csv)
    df_test = load_predictions(test_csv)
    y_val = df_val["true_label"].to_numpy()
    p_val = df_val["probability_score"].to_numpy()
    y_test = df_test["true_label"].to_numpy()
    p_test = df_test["probability_score"].to_numpy()

    # Threshold-independent curves (reported on test; val also saved for completeness).
    auc_val = save_roc(y_val, p_val, out_dir / "roc_curve_val.png", label="Val")
    ap_val = save_pr(y_val, p_val, out_dir / "pr_curve_val.png", label="Val")
    auc_test = save_roc(y_test, p_test, out_dir / "roc_curve_test.png", label="Test")
    ap_test = save_pr(y_test, p_test, out_dir / "pr_curve_test.png", label="Test")

    # Threshold sweep is done on validation only.
    val_tbl = compute_threshold_table(y_val, p_val, n=args.grid)
    val_tbl.to_csv(out_dir / "validation_threshold_sweep.csv", index=False)
    save_threshold_curves(
        val_tbl,
        out_dir / "validation_threshold_curves.png",
        title="Validation threshold sensitivity",
    )

    # Select threshold from validation (best validation F1).
    best = val_tbl.iloc[val_tbl["f1"].idxmax()].to_dict() # type: ignore
    selected_t = float(best["threshold"])

    # Apply selected threshold once on test.
    yhat_test_selected = (p_test >= selected_t).astype(int)
    yhat_test_default = (p_test >= 0.5).astype(int)

    test_f1_selected = float(f1_score(y_test, yhat_test_selected, zero_division=0))
    test_precision_selected = float(precision_score(y_test, yhat_test_selected, zero_division=0))
    test_recall_selected = float(recall_score(y_test, yhat_test_selected, zero_division=0))

    test_f1_default = float(f1_score(y_test, yhat_test_default, zero_division=0))
    test_precision_default = float(precision_score(y_test, yhat_test_default, zero_division=0))
    test_recall_default = float(recall_score(y_test, yhat_test_default, zero_division=0))

    summary = {
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "roc_auc_val": float(auc_val),
        "pr_ap_val": float(ap_val),
        "roc_auc_test": float(auc_test),
        "pr_ap_test": float(ap_test),
        "selected_threshold_from_validation": selected_t,
        "validation_best_f1": float(best["f1"]),
        "validation_precision_at_selected_threshold": float(best["precision"]),
        "validation_recall_at_selected_threshold": float(best["recall"]),
        "test_f1_at_validation_selected_threshold": test_f1_selected,
        "test_precision_at_validation_selected_threshold": test_precision_selected,
        "test_recall_at_validation_selected_threshold": test_recall_selected,
        "test_f1_at_default_0_5": test_f1_default,
        "test_precision_at_default_0_5": test_precision_default,
        "test_recall_at_default_0_5": test_recall_default,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "validation_tuned_test_metrics.csv", index=False)

    # Save test predictions with the selected validation threshold.
    applied = df_test.copy()
    applied["predicted_label_at_validation_selected_threshold"] = yhat_test_selected
    applied["predicted_label_at_default_0_5"] = yhat_test_default
    applied.to_csv(out_dir / "test_predictions_applied_validation_threshold.csv", index=False)

    print("Saved:")
    print(" -", out_dir / "roc_curve_val.png")
    print(" -", out_dir / "pr_curve_val.png")
    print(" -", out_dir / "roc_curve_test.png")
    print(" -", out_dir / "pr_curve_test.png")
    print(" -", out_dir / "validation_threshold_curves.png")
    print(" -", out_dir / "validation_threshold_sweep.csv")
    print(" -", out_dir / "validation_tuned_test_metrics.csv")
    print(" -", out_dir / "test_predictions_applied_validation_threshold.csv")
    print("\nSummary:", summary)

if __name__ == "__main__":
    main()
