"""
scripts/train_multi_seed.py

Orchestrates all 18 multi-seed training runs, collects every metric from
Section 5 of the experiment spec, writes per-run rows + threshold sweeps,
then computes mean ± std tables across seeds for each model.

Usage:
    python scripts/train_multi_seed.py [--models tfidf distilbert distilbert_hardneg deberta]

All outputs go to outputs/multi_seed_runs/.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path when running as `python scripts/train_multi_seed.py`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

RUNS: list[tuple[str, int]] = [
    # (model_key, seed)
    ("tfidf",              13),
    ("tfidf",              42),
    ("tfidf",             123),
    ("tfidf",            2024),
    ("tfidf",            7777),
    ("distilbert",         13),
    ("distilbert",         42),
    ("distilbert",        123),
    ("distilbert",       2024),
    ("distilbert",       7777),
    ("distilbert_hardneg", 13),
    ("distilbert_hardneg", 42),
    ("distilbert_hardneg",123),
    ("distilbert_hardneg",2024),
    ("distilbert_hardneg",7777),
    ("deberta",            13),
    ("deberta",            42),
    ("deberta",           123),
    ("deberta",          2024),
    ("deberta",          7777),
    # DeBERTa-v3-FT + hard-negative augmentation (5-seed cross-arch replication)
    # Seeds match the existing DistilBERT 5-seed runs exactly.
    # Baseline for each seed is read from outputs/multi_seed_runs/deberta/seed_{seed}/
    # (seeds 42, 123, 2024 have full per-seed dirs; 13/7777 are empty so
    #  baseline@τ_HN comparison is skipped for those two seeds).
    ("deberta_hardneg",    13),
    ("deberta_hardneg",    42),
    ("deberta_hardneg",   123),
    ("deberta_hardneg",  2024),
    ("deberta_hardneg",  7777),
]

METRIC_COLS = [
    "model_name", "seed", "tau",
    "IID_F1", "real_src_F1", "template_F1",
    "IID_recall", "IID_precision", "test_FPR", "ROC_AUC",
    "hb_FPR_agg", "hb_FPR_real", "hb_FPR_curated",
    "obf_recall",
    "domain_OOD_F1_agg",
    "medical_FPR",   "medical_recall",
    "legal_FPR",     "legal_recall",
    "finance_FPR",   "finance_recall",
    "code_FPR",      "code_recall",
    "struct_OOD_F1_agg", "struct_OOD_FPR_agg",
    "json_wrap_FPR",        "json_wrap_recall",
    "prompt_dilute_FPR",    "prompt_dilute_recall",
    "prefix_mimic_FPR",     "prefix_mimic_recall",
]

OUT_ROOT = Path("outputs/multi_seed_runs")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
MASTER_CSV = OUT_ROOT / "all_runs.csv"


# ── Per-model training dispatch ───────────────────────────────────────────────

def _run_tfidf(seed: int, out_dir: Path) -> dict:
    from src.baselines.tfidf_logreg import train_and_eval
    metrics = train_and_eval(grid_search=False, seed=seed, out_dir=out_dir)
    return metrics


def _run_distilbert(seed: int, out_dir: Path, hard_neg: bool) -> dict:
    from src.baselines.distilbert_baseline import run_train
    metrics = run_train(
        num_epochs=2, batch_size=16, lr=5e-5,
        seed=seed, hard_neg=hard_neg, out_dir=out_dir,
    )
    return metrics


def _run_deberta(seed: int, out_dir: Path) -> dict:
    import torch
    from src.baselines.deberta_v3 import run_train

    # Effective batch is always 16.  MPS needs smaller per-device batch to
    # stay under the ~20 GiB unified-memory limit (batch=4 OOMs in practice).
    if torch.cuda.is_available():
        batch, accum = 8, 2       # matches the 3 completed CUDA seeds
    elif torch.backends.mps.is_available():
        batch, accum = 2, 8       # MPS-safe; same effective batch = 16
    else:
        batch, accum = 4, 4       # CPU fallback

    metrics = run_train(
        num_epochs=3, batch_size=batch, lr=2e-5,
        seed=seed, gradient_accumulation_steps=accum, out_dir=out_dir,
    )
    return metrics


def _run_deberta_hardneg(seed: int, out_dir: Path, dry_run: bool = False) -> dict:
    """Train DeBERTa-v3-base + hard-negative augmentation for one seed.

    Hyperparameters are identical to the baseline deberta runs.  Outputs land
    in out_dir (outputs/multi_seed_runs/deberta_hardneg/seed_{seed}/).
    Baseline probability CSVs for the same seed are read automatically from
    outputs/multi_seed_runs/deberta/seed_{seed}/ when that directory is
    non-empty, enabling the three-way baseline / baseline@τ_HN / +hardneg
    comparison required for Table 11.
    """
    import torch
    from src.baselines.deberta_v3_hardneg import run_train

    if torch.cuda.is_available():
        batch, accum = 8, 2
    elif torch.backends.mps.is_available():
        batch, accum = 2, 8
    else:
        batch, accum = 4, 4

    metrics = run_train(
        seed=seed,
        out_dir=out_dir,
        repo_root=str(_ROOT),
        batch_size=batch,
        grad_accum=accum,
        dry_run=dry_run,
    )
    return metrics


def dispatch(model_key: str, seed: int, dry_run: bool = False) -> tuple[dict, Path]:
    run_dir = OUT_ROOT / model_key / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if model_key == "tfidf":
        metrics = _run_tfidf(seed, run_dir)
    elif model_key == "distilbert":
        metrics = _run_distilbert(seed, run_dir, hard_neg=False)
    elif model_key == "distilbert_hardneg":
        metrics = _run_distilbert(seed, run_dir, hard_neg=True)
    elif model_key == "deberta":
        metrics = _run_deberta(seed, run_dir)
    elif model_key == "deberta_hardneg":
        metrics = _run_deberta_hardneg(seed, run_dir, dry_run=dry_run)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    return metrics, run_dir


# ── Sweep file handling ───────────────────────────────────────────────────────

def _save_sweep(model_key: str, seed: int, run_dir: Path) -> None:
    src = run_dir / "threshold_sweep.csv"
    if not src.exists():
        print(f"  [warn] no threshold_sweep.csv in {run_dir}")
        return
    dest = OUT_ROOT / f"sweep_{model_key}_seed{seed}.csv"
    import shutil
    shutil.copy(src, dest)


# ── Master CSV ────────────────────────────────────────────────────────────────

def _append_row(row: dict) -> None:
    df = pd.DataFrame([row], columns=METRIC_COLS)
    if MASTER_CSV.exists():
        df.to_csv(MASTER_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(MASTER_CSV, index=False)


def _already_done(model_key: str, seed: int) -> bool:
    if not MASTER_CSV.exists():
        return False
    df = pd.read_csv(MASTER_CSV)
    return not df[(df["model_name"] == model_key) & (df["seed"] == seed)].empty


# ── Aggregation ───────────────────────────────────────────────────────────────

def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def aggregate_and_print() -> None:
    if not MASTER_CSV.exists():
        print("No results yet.")
        return

    df = pd.read_csv(MASTER_CSV)
    numeric_cols = [c for c in METRIC_COLS if c not in ("model_name", "seed")]

    print("\n" + "=" * 80)
    print("MULTI-SEED RESULTS — MEAN ± STD")
    print("=" * 80)

    summary_rows = []
    for model_key in df["model_name"].unique():
        sub = df[df["model_name"] == model_key]
        print(f"\nModel: {model_key}  (n_seeds={len(sub)})")
        print(f"  Seeds: {sorted(sub['seed'].tolist())}")
        row: dict = {"model_name": model_key, "n_seeds": len(sub)}
        for col in numeric_cols:
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            mean = vals.mean(); std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            row[f"{col}_mean"] = round(mean, 4)
            row[f"{col}_std"]  = round(std,  4)
            row[f"{col}_fmt"]  = _fmt(mean, std)
            print(f"  {col:30s}  {_fmt(mean, std)}")
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_ROOT / "summary_mean_std.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")

    # ── Mean threshold sweep curves ───────────────────────────────────────────
    print("\n── Mean threshold sweep curves ──")
    for model_key in df["model_name"].unique():
        sub = df[df["model_name"] == model_key]
        sweep_frames = []
        for _, row_r in sub.iterrows():
            p = OUT_ROOT / f"sweep_{model_key}_seed{int(row_r['seed'])}.csv"
            if p.exists():
                sweep_frames.append(pd.read_csv(p))
        if not sweep_frames:
            continue
        # align on tau
        merged = sweep_frames[0][["tau"]].copy()
        for i, sf in enumerate(sweep_frames):
            merged = merged.merge(sf.rename(columns={"test_F1": f"f1_{i}", "hb_FPR_agg": f"fpr_{i}"}),
                                  on="tau", how="inner")
        f1_cols  = [c for c in merged.columns if c.startswith("f1_")]
        fpr_cols = [c for c in merged.columns if c.startswith("fpr_")]
        out_sweep = OUT_ROOT / f"mean_sweep_{model_key}.csv"
        pd.DataFrame({
            "tau":           merged["tau"],
            "mean_test_F1":  merged[f1_cols].mean(axis=1).round(6),
            "std_test_F1":   merged[f1_cols].std(axis=1, ddof=1).round(6),
            "mean_hb_FPR":   merged[fpr_cols].mean(axis=1).round(6),
            "std_hb_FPR":    merged[fpr_cols].std(axis=1, ddof=1).round(6),
        }).to_csv(out_sweep, index=False)
        print(f"  {model_key}: mean sweep → {out_sweep}")


# ── Paper-ready numbers printer ───────────────────────────────────────────────

def print_paper_numbers() -> None:
    if not MASTER_CSV.exists():
        return
    df = pd.read_csv(MASTER_CSV)

    paper_metrics = [
        ("IID F1",                  "IID_F1"),
        ("Real-src F1",             "real_src_F1"),
        ("Template-only F1",        "template_F1"),
        ("IID Recall",              "IID_recall"),
        ("IID Ben-FPR",             "test_FPR"),
        ("Hard-benign FPR (agg)",   "hb_FPR_agg"),
        ("Hard-benign FPR (real)",  "hb_FPR_real"),
        ("Hard-benign FPR (cur.)",  "hb_FPR_curated"),
        ("Obfuscation Recall",      "obf_recall"),
        ("Domain OOD F1",           "domain_OOD_F1_agg"),
        ("Structural OOD FPR",      "struct_OOD_FPR_agg"),
        ("ROC-AUC",                 "ROC_AUC"),
    ]

    print("\n" + "=" * 80)
    print("PAPER-READY NUMBERS")
    print("=" * 80)
    for model_key in df["model_name"].unique():
        sub = df[df["model_name"] == model_key]
        print(f"\n### {model_key} ###")
        for label, col in paper_metrics:
            if col not in sub.columns:
                print(f"  {label:35s}  N/A")
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals) == 0:
                print(f"  {label:35s}  N/A")
                continue
            mean = vals.mean(); std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            print(f"  {label:35s}  {_fmt(mean, std)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=["tfidf", "distilbert", "distilbert_hardneg", "deberta", "deberta_hardneg"],
        help=(
            "Which models to train. "
            "Choices: tfidf distilbert distilbert_hardneg deberta deberta_hardneg"
        ),
    )
    parser.add_argument(
        "--skip-done", action="store_true", default=True,
        help="Skip runs already recorded in all_runs.csv (default: True)",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip training, just recompute aggregates from existing all_runs.csv",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "For deberta_hardneg runs: verify data, paths, and preflight checks "
            "without training.  Other model types print what would run and skip. "
            "Use this to smoke-test before an A100 session."
        ),
    )
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_and_print()
        print_paper_numbers()
        return

    runs_to_do = [(m, s) for m, s in RUNS if m in args.models]
    total = len(runs_to_do)
    print(f"Total runs planned: {total}")

    if args.dry_run:
        print("\n[DRY RUN MODE] Only deberta_hardneg runs perform preflight checks.")
        print("All other model types are listed but skipped.\n")

    for i, (model_key, seed) in enumerate(runs_to_do, 1):
        if not args.dry_run and args.skip_done and _already_done(model_key, seed):
            print(f"[{i}/{total}] SKIP  {model_key}  seed={seed}  (already in {MASTER_CSV})")
            continue

        print(f"\n{'='*70}")
        print(f"[{i}/{total}]  {model_key}  seed={seed}"
              + ("  [DRY RUN]" if args.dry_run else ""))
        print(f"{'='*70}")

        if args.dry_run and model_key != "deberta_hardneg":
            run_dir = OUT_ROOT / model_key / f"seed_{seed}"
            print(f"  [DRY RUN] Would run {model_key} seed={seed} → {run_dir}")
            continue

        try:
            metrics, run_dir = dispatch(model_key, seed, dry_run=args.dry_run)
            if args.dry_run:
                continue  # preflight only; nothing to record
            row = {"model_name": model_key, "seed": seed}
            row.update(metrics)
            _append_row({col: row.get(col, None) for col in METRIC_COLS})
            _save_sweep(model_key, seed, run_dir)
            print(f"  → Row saved  IID_F1={metrics.get('IID_F1')}  "
                  f"hb_FPR_agg={metrics.get('hb_FPR_agg')}")
        except Exception:
            print(f"  [ERROR] {model_key} seed={seed} failed:")
            traceback.print_exc()
            print("  Continuing to next run...")

    aggregate_and_print()
    print_paper_numbers()


if __name__ == "__main__":
    main()
