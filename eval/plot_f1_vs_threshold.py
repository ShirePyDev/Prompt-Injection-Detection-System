from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep_csv",
        type=str,
        default="outputs/distilbert_baseline/threshold_analysis/validation_threshold_sweep.csv",
        help="Validation sweep CSV with columns: threshold,f1.",
    )
    ap.add_argument(
        "--val_metrics_csv",
        type=str,
        default="outputs/distilbert_baseline/threshold_analysis/validation_tuned_test_metrics.csv",
        help="CSV containing the validation-selected threshold (single-row).",
    )
    ap.add_argument(
        "--out_png",
        type=str,
        default="outputs/distilbert_baseline/threshold_analysis/f1_vs_threshold.png",
    )
    ap.add_argument("--default_t", type=float, default=0.50)
    args = ap.parse_args()

    sweep_csv = Path(args.sweep_csv)
    val_metrics_csv = Path(args.val_metrics_csv)
    out_png = Path(args.out_png)

    if sweep_csv.name != "validation_threshold_sweep.csv":
        raise ValueError(
            f"Invalid sweep file: {sweep_csv}. "
            "Only validation_threshold_sweep.csv is allowed to avoid test-sweep mixing."
        )

    if not sweep_csv.exists():
        raise FileNotFoundError(f"Missing sweep file: {sweep_csv}")
    if not val_metrics_csv.exists():
        raise FileNotFoundError(f"Missing validation metrics file: {val_metrics_csv}")

    df = pd.read_csv(sweep_csv)
    if not {"threshold", "f1"}.issubset(df.columns):
        raise ValueError(f"{sweep_csv} must contain columns threshold,f1; found {list(df.columns)}")

    vm = pd.read_csv(val_metrics_csv)
    # Expect a single-row CSV; tolerate extra columns
    thresh_col_candidates = [
        "selected_threshold_from_validation",
        "threshold",
        "best_f1_threshold",
        "val_selected_threshold",
    ]
    tcol = next((c for c in thresh_col_candidates if c in vm.columns), None)
    if tcol is None:
        raise ValueError(
            f"{val_metrics_csv} must contain a threshold column (one of {thresh_col_candidates}); "
            f"found {list(vm.columns)}"
        )
    val_selected_t = float(vm[tcol].iloc[0])

    plt.figure()
    plt.plot(df["threshold"], df["f1"])
    plt.axvline(val_selected_t, linestyle="--")
    plt.axvline(float(args.default_t), linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"Saved: {out_png}")
    print(f"Curve source: {sweep_csv}")
    print(f"Validation-selected threshold: {val_selected_t}")
    print(f"Default threshold: {args.default_t}")

if __name__ == "__main__":
    main()
