"""Generate Figure 1 threshold_sweep.pdf from five-seed multi-run sweeps."""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "outputs" / "multi_seed_runs"
ALL_RUNS = OUT_ROOT / "all_runs.csv"
OUT_PDF = ROOT / "outputs" / "threshold_sweep.pdf"
REQUIRED_SEEDS = [13, 42, 123, 2024, 7777]

MEAN_REQUIRED_COLUMNS = ["tau", "mean_test_F1", "std_test_F1", "mean_hb_FPR", "std_hb_FPR"]
SEED_REQUIRED_COLUMNS = ["tau", "test_F1", "hb_FPR_agg"]
ALL_RUN_REQUIRED_COLUMNS = ["model_name", "seed", "tau", "IID_F1", "hb_FPR_agg"]

MODELS = [
    {
        "model_key": "tfidf",
        "label": "TF-IDF+LR",
        "mean_sweep_csv": OUT_ROOT / "mean_sweep_tfidf.csv",
        "color": "#2ca02c",
        "marker": "^",
    },
    {
        "model_key": "distilbert",
        "label": "DistilBERT",
        "mean_sweep_csv": OUT_ROOT / "mean_sweep_distilbert.csv",
        "color": "#ff7f0e",
        "marker": "s",
    },
    {
        "model_key": "deberta",
        "label": "DeBERTa-v3-FT",
        "mean_sweep_csv": OUT_ROOT / "mean_sweep_deberta.csv",
        "color": "#1f77b4",
        "marker": "o",
    },
]


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {path.relative_to(ROOT)}: {missing}")


def load_seed42_points() -> pd.DataFrame:
    if not ALL_RUNS.exists():
        raise SystemExit(f"Missing operating-point file: {ALL_RUNS.relative_to(ROOT)}")
    all_runs = pd.read_csv(ALL_RUNS)
    require_columns(all_runs, ALL_RUN_REQUIRED_COLUMNS, ALL_RUNS)
    return all_runs[all_runs["seed"] == 42].copy()


def verify_mean_sweep(model_key: str, mean_sweep: pd.DataFrame) -> list[Path]:
    merged = None
    seed_paths = []
    for seed in REQUIRED_SEEDS:
        seed_path = OUT_ROOT / f"sweep_{model_key}_seed{seed}.csv"
        if not seed_path.exists():
            raise SystemExit(f"Missing seed sweep: {seed_path.relative_to(ROOT)}")
        seed_df = pd.read_csv(seed_path)
        require_columns(seed_df, SEED_REQUIRED_COLUMNS, seed_path)
        seed_paths.append(seed_path)
        renamed = seed_df.rename(
            columns={"test_F1": f"f1_{seed}", "hb_FPR_agg": f"fpr_{seed}"}
        )[["tau", f"f1_{seed}", f"fpr_{seed}"]]
        merged = renamed if merged is None else merged.merge(renamed, on="tau", how="inner")

    if merged is None or len(merged) != len(mean_sweep):
        raise SystemExit(f"Mean sweep row count does not match seed sweeps for {model_key}")

    f1_cols = [f"f1_{seed}" for seed in REQUIRED_SEEDS]
    fpr_cols = [f"fpr_{seed}" for seed in REQUIRED_SEEDS]
    recomputed = pd.DataFrame({
        "tau": merged["tau"],
        "mean_test_F1": merged[f1_cols].mean(axis=1).round(6),
        "std_test_F1": merged[f1_cols].std(axis=1, ddof=1).round(6),
        "mean_hb_FPR": merged[fpr_cols].mean(axis=1).round(6),
        "std_hb_FPR": merged[fpr_cols].std(axis=1, ddof=1).round(6),
    })
    if not recomputed[MEAN_REQUIRED_COLUMNS].equals(mean_sweep[MEAN_REQUIRED_COLUMNS]):
        raise SystemExit(f"{model_key} mean sweep does not match recomputation from five seeds")
    return seed_paths


def main() -> None:
    seed42 = load_seed42_points()
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for cfg in MODELS:
        mean_path = cfg["mean_sweep_csv"]
        if not mean_path.exists():
            raise SystemExit(f"Missing mean sweep: {mean_path.relative_to(ROOT)}")
        sweep = pd.read_csv(mean_path)
        require_columns(sweep, MEAN_REQUIRED_COLUMNS, mean_path)
        seed_paths = verify_mean_sweep(cfg["model_key"], sweep)

        print(f"{cfg['label']} mean sweep: {mean_path.relative_to(ROOT)}")
        print(f"  columns: {list(sweep.columns)}")
        print("  seed sweeps:")
        for seed_path in seed_paths:
            print(f"    {seed_path.relative_to(ROOT)}")

        plot_df = sweep.sort_values("mean_hb_FPR")
        x = plot_df["mean_hb_FPR"]
        y = plot_df["mean_test_F1"]
        y_std = plot_df["std_test_F1"]

        ax.fill_between(
            x,
            (y - y_std).clip(lower=0.0),
            (y + y_std).clip(upper=1.0),
            color=cfg["color"],
            alpha=0.25,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            x,
            y,
            color=cfg["color"],
            label=cfg["label"],
            linewidth=1.8,
            alpha=0.95,
            zorder=3,
        )

        op_rows = seed42[seed42["model_name"] == cfg["model_key"]]
        if len(op_rows) != 1:
            raise SystemExit(f"Expected one seed-42 operating row for {cfg['model_key']}, found {len(op_rows)}")
        op = op_rows.iloc[0]
        ax.scatter(
            [op["hb_FPR_agg"]],
            [op["IID_F1"]],
            color=cfg["color"],
            marker=cfg["marker"],
            s=80,
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
        )
        print(
            f"  seed-42 operating point: "
            f"tau={op['tau']:.4f}, hb_FPR_agg={op['hb_FPR_agg']:.6f}, IID_F1={op['IID_F1']:.6f}"
        )

    ax.axvline(0.10, color="grey", linestyle="--", linewidth=1.0, label="FPR = 0.10")
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1.0, label="F1 = 0.95")

    ax.set_xlabel("Hard-benign FPR (false-positive rate)", fontsize=11)
    ax.set_ylabel("IID test F1", fontsize=11)
    ax.set_title("Threshold Frontier: IID F1 vs. Hard-benign FPR", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":", zorder=0)
    fig.tight_layout()

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
