"""
scripts/run_deberta_mps_seeds.py

Trains the two remaining DeBERTa-v3 seeds (13 and 7777) on MPS.
Identical hyperparameters to the 3 completed CUDA seeds, except
batch_size=2 / grad_accum=8 (effective batch = 16) to stay under
the ~20 GiB MPS unified-memory limit.

Usage (from project root):
    python scripts/run_deberta_mps_seeds.py

Outputs land in:
    outputs/multi_seed_runs/deberta/seed_13/
    outputs/multi_seed_runs/deberta/seed_7777/

and both rows are appended to outputs/multi_seed_runs/all_runs.csv.

IMPORTANT — read before running
--------------------------------
DeBERTa-v3 has known numerical instability on MPS/CPU: training can
collapse to loss≈0 and grad_norm=NaN from the very first step.  This
script exits with a clear error after epoch 1 if that happens, so you
don't waste 8 hours on a dead run.  If it does fail, use Colab with
batch_size=8 / grad_accum=2 (exact match to the completed 3 seeds).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch

SEEDS = [13, 7777]
OUT_ROOT  = Path("outputs/multi_seed_runs")
MASTER_CSV = OUT_ROOT / "all_runs.csv"

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


def _already_done(seed: int) -> bool:
    if not MASTER_CSV.exists():
        return False
    df = pd.read_csv(MASTER_CSV)
    return not df[(df["model_name"] == "deberta") & (df["seed"] == seed)].empty


def _append_row(row: dict) -> None:
    df = pd.DataFrame([row], columns=METRIC_COLS)
    if MASTER_CSV.exists():
        df.to_csv(MASTER_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(MASTER_CSV, index=False)


def _save_sweep(seed: int, run_dir: Path) -> None:
    import shutil
    src  = run_dir / "threshold_sweep.csv"
    dest = OUT_ROOT / f"sweep_deberta_seed{seed}.csv"
    if src.exists():
        shutil.copy(src, dest)
    else:
        print(f"  [warn] no threshold_sweep.csv for seed {seed}")


def _recompute_aggregates() -> None:
    """Print updated mean±std for DeBERTa after new seeds are added."""
    if not MASTER_CSV.exists():
        return
    df = pd.read_csv(MASTER_CSV)
    sub = df[df["model_name"] == "deberta"]
    print(f"\n{'='*70}")
    print(f"DeBERTa-v3  n_seeds={len(sub)}  seeds={sorted(sub['seed'].tolist())}")
    print(f"{'='*70}")
    for col in ["IID_F1", "hb_FPR_agg", "hb_FPR_real", "hb_FPR_curated",
                "real_src_F1", "template_F1", "obf_recall",
                "domain_OOD_F1_agg", "struct_OOD_FPR_agg", "ROC_AUC"]:
        if col not in sub.columns:
            continue
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        mean = vals.mean()
        std  = vals.std(ddof=1) if len(vals) > 1 else 0.0
        print(f"  {col:30s}  {mean:.4f} ± {std:.4f}")


def _check_device() -> torch.device:
    if torch.cuda.is_available():
        print("[info] CUDA detected — using batch_size=8, grad_accum=2 (exact match to completed seeds)")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("[info] MPS detected — using batch_size=2, grad_accum=8 (effective batch=16)")
        print("[warn] DeBERTa-v3 may produce NaN loss on MPS.  Will verify after epoch 1.")
        return torch.device("mps")
    print("[warn] CPU only — using batch_size=4, grad_accum=4. NaN risk is high on CPU.")
    return torch.device("cpu")


def main() -> None:
    device = _check_device()

    if device.type == "cuda":
        batch, accum = 8, 2
    elif device.type == "mps":
        batch, accum = 2, 8
    else:
        batch, accum = 4, 4

    print(f"\nConfig: batch_size={batch}  grad_accum={accum}  effective_batch={batch*accum}")
    print(f"Seeds to run: {SEEDS}\n")

    for seed in SEEDS:
        if _already_done(seed):
            print(f"[skip] seed={seed} already in {MASTER_CSV}")
            continue

        run_dir = OUT_ROOT / "deberta" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  DeBERTa-v3  seed={seed}")
        print(f"{'='*70}")

        try:
            from src.baselines.deberta_v3 import run_train
            metrics = run_train(
                num_epochs=3,
                batch_size=batch,
                lr=2e-5,
                seed=seed,
                gradient_accumulation_steps=accum,
                out_dir=run_dir,
            )
            row = {"model_name": "deberta", "seed": seed}
            row.update(metrics)
            _append_row({col: row.get(col) for col in METRIC_COLS})
            _save_sweep(seed, run_dir)
            print(f"\n  seed={seed} done  IID_F1={metrics.get('IID_F1')}  "
                  f"hb_FPR_real={metrics.get('hb_FPR_real')}")

        except ValueError as e:
            # run_train raises ValueError when val logits contain NaN/Inf
            print(f"\n[FATAL] seed={seed} — NaN divergence detected after epoch 1:")
            print(f"  {e}")
            print("\nDo NOT continue on MPS/CPU for this seed.")
            print("Use Google Colab (free T4 GPU) with batch_size=8, grad_accum=2.")
            print("The Colab notebook is at: colab_deberta_training.ipynb")
            sys.exit(1)

        except Exception:
            print(f"\n[ERROR] seed={seed} failed:")
            traceback.print_exc()
            print("Continuing to next seed...")

    _recompute_aggregates()


if __name__ == "__main__":
    main()
