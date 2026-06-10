# src/baselines/deberta_v3_hardneg.py
"""
DeBERTa-v3-base fine-tuned on PIDS-Bench v3 + HardNeg augmentation.

Trains with identical hyperparameters to baseline deberta_v3.py, augmenting
the training and validation splits with 419 + 116 hard-negative examples from
data/pids_bench_v3/hard_negative_train.csv and hard_negative_val.csv.

Evaluates across all axes and produces a three-way comparison:
  1. Baseline DeBERTa at τ_baseline
  2. Baseline DeBERTa at τ_HN          (isolates threshold effect)
  3. DeBERTa + HardNeg at τ_HN         (shows pure training effect)

Outputs: outputs/deberta_v3_hardneg/
  val_predictions.csv
  test_predictions.csv
  hard_benign_predictions.csv
  obfuscated_attacks_predictions.csv
  domain_ood_predictions.csv
  structural_ood_probability_scores.csv
  hard_benign_fpr.json
  obfuscated_attacks_report.json
  domain_ood_report.json
  structural_ood_report.json
  thresholds.json
  summary.csv
  per_split_metrics.csv
  per_domain.csv
  per_transform.csv
  report.md
  baseline_probs_cache/   (cached baseline probs for splits not covered by deberta_v3/)

Usage:
  # Local (run from repo root):
  python src/baselines/deberta_v3_hardneg.py

  # Google Colab:
  python src/baselines/deberta_v3_hardneg.py --repo-root /content/Prompt-Injection-Detector-System

  # Colab with Drive mount:
  python src/baselines/deberta_v3_hardneg.py --repo-root /content/drive/MyDrive/PIDS

  # Skip reloading baseline model (omit domain_ood / obfuscated baseline comparison):
  python src/baselines/deberta_v3_hardneg.py --skip-baseline-compare
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — must match baseline deberta_v3.py exactly
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME    = "microsoft/deberta-v3-base"
MAX_LEN       = 512
NUM_EPOCHS    = 3
LR            = 2e-5
WARMUP_RATIO  = 0.06
SEED          = 42

# These match the actual saved baseline (batch_size=8, grad_accum=2, effective=16)
DEFAULT_BATCH = 8
DEFAULT_ACCUM = 2

# Hard-negative row count assertions
HN_TRAIN_EXPECTED = 419
HN_VAL_EXPECTED   = 116


# ─────────────────────────────────────────────────────────────────────────────
# Path layout
# ─────────────────────────────────────────────────────────────────────────────

def _make_paths(
    repo_root: Path,
    out_dir: Optional[Path] = None,
    multi_seed_bl_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Build all path constants for a training run.

    out_dir: when provided (multi-seed orchestration), outputs go here and the
             model is saved under out_dir/model/.  Defaults to the canonical
             repo_root/outputs/deberta_v3_hardneg/ location.

    multi_seed_bl_dir: when provided, baseline probability CSVs and the
             baseline model are read from this per-seed directory instead of
             the canonical outputs/deberta_v3/ + models/deberta_v3/ locations.
             Pass outputs/multi_seed_runs/deberta/seed_{SEED}/ for multi-seed
             runs.  The function falls back gracefully if files are absent.
    """
    data     = repo_root / "data" / "pids_bench_v3"
    eval_dir = data / "eval_subsets"
    ood_dir  = data / "ood"
    out      = Path(out_dir) if out_dir is not None else repo_root / "outputs" / "deberta_v3_hardneg"

    if multi_seed_bl_dir is not None:
        bl_out   = Path(multi_seed_bl_dir)
        bl_model = bl_out / "model"
    else:
        bl_out   = repo_root / "outputs" / "deberta_v3"
        bl_model = repo_root / "models" / "deberta_v3"

    # In multi-seed mode model lives inside the per-seed output dir; otherwise
    # it goes to the shared models/ directory to match the canonical layout.
    model_dir = (out / "model") if out_dir is not None else (repo_root / "models" / "deberta_v3_hardneg")

    return {
        "repo_root":      repo_root,
        "data":           data,
        "eval_dir":       eval_dir,
        "ood_dir":        ood_dir,
        "out":            out,
        "model_dir":      model_dir,
        "bl_out":         bl_out,
        "bl_model":       bl_model,
        "bl_cache":       out / "baseline_probs_cache",
        # data splits
        "train":          data / "train.csv",
        "val":            data / "val.csv",
        "test":           data / "test.csv",
        "hn_train":       data / "hard_negative_train.csv",
        "hn_val":         data / "hard_negative_val.csv",
        # eval splits (NEVER modified)
        "hard_benign":    eval_dir / "hard_benign_test.csv",
        "obfuscated":     eval_dir / "obfuscated_attacks.csv",
        "domain_ood":     ood_dir  / "domain_ood.csv",
        "structural_ood": ood_dir  / "structural_ood.csv",
        # baseline cached probability files — may not exist for all seeds
        "bl_test_probs":   bl_out / "test_predictions.csv",
        "bl_hb_probs":     bl_out / "hard_benign_predictions.csv",
        "bl_struct_probs": bl_out / "structural_ood_probability_scores.csv",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _free_gpu(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"text", "label"}.issubset(df.columns), (
        f"Missing columns in {path}: {df.columns.tolist()}"
    )
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def _softmax(logits: np.ndarray) -> np.ndarray:
    x       = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp     = np.exp(np.clip(shifted, -80.0, 80.0))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float64)


def _sdiv(a: float, b: float) -> float:
    return a / b if b else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Threshold tuning: step 0.002, maximize F1, tie-break by highest τ
# ─────────────────────────────────────────────────────────────────────────────

def best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (tau, f1). Sweep 0.000→1.000 step 0.002. Tie-break: highest τ."""
    if not np.isfinite(probs).all():
        raise ValueError(
            "Probabilities contain NaN/Inf. Training may have diverged. "
            "Try --batch-size 4 or check torch/transformers versions."
        )
    taus    = np.arange(0.0, 1.001, 0.002)
    best_t  = 0.0
    best_f1 = -1.0
    for t in taus:
        f = float(f1_score(labels, (probs >= t).astype(int), zero_division=0))
        if f > best_f1 or (abs(f - best_f1) < 1e-12 and float(t) > best_t):
            best_f1 = f
            best_t  = float(t)
    return best_t, best_f1


# ─────────────────────────────────────────────────────────────────────────────
# EvalResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    n:         int
    precision: float
    recall:    float
    f1:        float
    fpr:       float
    fnr:       float
    tau:       float
    roc_auc:   Optional[float] = None

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def _eval_at(probs: np.ndarray, y_true: np.ndarray, tau: float) -> EvalResult:
    y_pred = (probs >= tau).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    try:
        auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else None
    except Exception:
        auc = None
    return EvalResult(
        n         = len(y_true),
        precision = _sdiv(tp, tp + fp),
        recall    = _sdiv(tp, tp + fn),
        f1        = float(f1_score(y_true, y_pred, zero_division=0)),
        fpr       = _sdiv(fp, fp + tn),
        fnr       = _sdiv(fn, fn + tp),
        tau       = tau,
        roc_auc   = auc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _probs_via_trainer(
    trainer: Trainer,
    tok:     AutoTokenizer,
    texts:   list[str],
    labels:  list[int],
) -> np.ndarray:
    enc  = tok(texts, truncation=True, padding=True, max_length=MAX_LEN)
    ds   = Dataset.from_dict({**enc, "labels": labels})
    pred = trainer.predict(ds)
    return _softmax(np.asarray(pred.predictions))[:, 1]


def _probs_via_model(
    model_dir:  Path,
    device:     torch.device,
    texts:      list[str],
    batch_size: int = 8,
) -> np.ndarray:
    """Load a saved HF model and run batched inference without a Trainer."""
    print(f"    Loading model from {model_dir} for inference ...")
    tok   = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), torch_dtype=torch.float32
    ).to(device).eval()
    accepted = set(inspect.signature(model.forward).parameters.keys())
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch  = texts[i : i + batch_size]
            enc    = tok(batch, truncation=True, padding=True,
                         max_length=MAX_LEN, return_tensors="pt")
            enc    = {k: v.to(device) for k, v in enc.items() if k in accepted}
            logits = model(**enc).logits.cpu().numpy()
            all_probs.append(_softmax(logits)[:, 1])
    del model
    _free_gpu(device)
    return np.concatenate(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Confirm hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

def confirm_hyperparams(batch_size: int, grad_accum: int, seed: int) -> None:
    print("\n" + "=" * 65)
    print("STEP 1 — Baseline hyperparameter confirmation")
    print("=" * 65)
    print(f"  Model                 : {MODEL_NAME}")
    print(f"  Epochs                : {NUM_EPOCHS}")
    print(f"  Batch size (device)   : {batch_size}")
    print(f"  Gradient accumulation : {grad_accum}")
    print(f"  Effective batch size  : {batch_size * grad_accum}")
    print(f"  Learning rate         : {LR}")
    print(f"  Warmup ratio          : {WARMUP_RATIO}")
    print(f"  Max length            : {MAX_LEN}")
    print(f"  Weight decay          : 0.01")
    print(f"  Max grad norm         : 1.0")
    print(f"  fp16 / bf16           : False / False")
    print(f"  Seed                  : {seed}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Load and verify hard-negatives
# ─────────────────────────────────────────────────────────────────────────────

def verify_hard_negatives(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 65)
    print("STEP 3 — Loading and verifying hard-negative data")
    print("=" * 65)

    hn_tr = pd.read_csv(paths["hn_train"])
    hn_va = pd.read_csv(paths["hn_val"])

    print(f"hard_negative_train.csv  : {len(hn_tr)} rows")
    print(f"  columns    : {list(hn_tr.columns)}")
    print(f"  label dist : {hn_tr['label'].value_counts().to_dict()}")
    if "hardneg_category" in hn_tr.columns:
        print(f"  categories : {hn_tr['hardneg_category'].value_counts().to_dict()}")

    print(f"\nhard_negative_val.csv    : {len(hn_va)} rows")
    print(f"  columns    : {list(hn_va.columns)}")
    print(f"  label dist : {hn_va['label'].value_counts().to_dict()}")

    assert len(hn_tr) == HN_TRAIN_EXPECTED, (
        f"Expected {HN_TRAIN_EXPECTED} train hard-negatives, got {len(hn_tr)}"
    )
    assert len(hn_va) == HN_VAL_EXPECTED, (
        f"Expected {HN_VAL_EXPECTED} val hard-negatives, got {len(hn_va)}"
    )
    assert (hn_tr["label"] == 0).all(), "Non-zero label found in hard_negative_train!"
    assert (hn_va["label"] == 0).all(), "Non-zero label found in hard_negative_val!"

    print(f"\nAssertions passed: {HN_TRAIN_EXPECTED} train / {HN_VAL_EXPECTED} val / all labels == 0")
    return hn_tr, hn_va


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Deduplication safety check
# ─────────────────────────────────────────────────────────────────────────────

def dedup_safety_check(hn_tr: pd.DataFrame, paths: dict[str, Path]) -> None:
    print("\n" + "=" * 65)
    print("STEP 5 — Deduplication safety check (hard-neg vs eval splits)")
    print("=" * 65)

    hn_norm = set(hn_tr["text"].str.lower().str.strip())
    checks  = {
        "test":           paths["test"],
        "hard_benign":    paths["hard_benign"],
        "domain_ood":     paths["domain_ood"],
        "structural_ood": paths["structural_ood"],
    }
    all_clean = True
    for name, path in checks.items():
        df      = pd.read_csv(path)
        overlap = df["text"].astype(str).str.lower().str.strip().isin(hn_norm).sum()
        status  = "OK" if overlap == 0 else f"OVERLAP={overlap}  !!!"
        print(f"  {name:20}  n={len(df):5}  overlap={overlap}  [{status}]")
        if overlap > 0:
            all_clean = False

    if not all_clean:
        raise RuntimeError(
            "Hard-negative overlap with evaluation splits detected. "
            "Aborting — do not proceed without resolving contamination."
        )
    print("\nAll overlap counts = 0.  Safe to proceed.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Augment splits + tokenize
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = accuracy_score(labels, preds)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f)}


def load_augmented_splits(
    paths: dict[str, Path],
    hn_tr: pd.DataFrame,
    hn_va: pd.DataFrame,
    tok:   AutoTokenizer,
) -> tuple[DatasetDict, pd.DataFrame, pd.DataFrame]:
    """Returns (ds_tok, aug_tr_df, aug_va_df)."""
    print("\n" + "=" * 65)
    print("STEP 4 — Augmenting training and validation splits")
    print("=" * 65)

    base_tr = _read_csv(paths["train"])
    base_va = _read_csv(paths["val"])

    print(f"  Base train : {len(base_tr)} rows")
    print(f"  Base val   : {len(base_va)} rows")

    aug_tr = pd.concat(
        [base_tr, hn_tr[["text", "label"]]], ignore_index=True
    )
    aug_va = pd.concat(
        [base_va, hn_va[["text", "label"]]], ignore_index=True
    )
    aug_tr["text"]  = aug_tr["text"].astype(str)
    aug_va["text"]  = aug_va["text"].astype(str)
    aug_tr["label"] = aug_tr["label"].astype(int)
    aug_va["label"] = aug_va["label"].astype(int)

    print(f"  Augmented train : {len(aug_tr)} rows  (+{len(aug_tr) - len(base_tr)} HN)")
    print(f"  Augmented val   : {len(aug_va)} rows  (+{len(aug_va) - len(base_va)} HN)")
    print(f"  Aug train label dist : {aug_tr['label'].value_counts().to_dict()}")
    print(f"  Aug val   label dist : {aug_va['label'].value_counts().to_dict()}")

    test_df = _read_csv(paths["test"])

    def _tok_fn(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

    def _to_hf(df: pd.DataFrame) -> Dataset:
        raw    = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
        remove = [c for c in raw.column_names if c != "label"]
        return raw.map(_tok_fn, batched=True, remove_columns=remove)

    ds_tok = DatasetDict({
        "train":      _to_hf(aug_tr),
        "validation": _to_hf(aug_va),
        "test":       _to_hf(test_df),
    })
    return ds_tok, aug_tr, aug_va


# ─────────────────────────────────────────────────────────────────────────────
# Steps 6 & 7 — Train + threshold selection
# ─────────────────────────────────────────────────────────────────────────────

def train_and_tune(
    paths:      dict[str, Path],
    hn_tr:      pd.DataFrame,
    hn_va:      pd.DataFrame,
    batch_size: int,
    grad_accum: int,
    seed:       int,
) -> tuple[Trainer, AutoTokenizer, float, float, np.ndarray, np.ndarray]:
    """
    Train DeBERTa-v3-base + HardNeg. Returns
    (trainer, tok, tau_hn, val_f1_at_tau, val_probs, val_true).
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    device = pick_device()
    print("\n" + "=" * 65)
    print("STEP 6 — Training DeBERTa-v3-base + HardNeg")
    print("=" * 65)
    print(f"  Device          : {device}")
    print(f"  Model           : {MODEL_NAME}")
    print(f"  Out dir         : {paths['out']}")
    print(f"  Model saved to  : {paths['model_dir']}")

    if device.type == "mps":
        torch.mps.manual_seed(seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    paths["out"].mkdir(parents=True, exist_ok=True)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)

    tok    = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    ds_tok, aug_tr_df, aug_va_df = load_augmented_splits(paths, hn_tr, hn_va, tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, torch_dtype=torch.float32
    ).to(device).float()

    ta_sig      = inspect.signature(TrainingArguments.__init__)
    strategy_kw = (
        {"evaluation_strategy": "epoch"}
        if "evaluation_strategy" in ta_sig.parameters
        else {"eval_strategy": "epoch"}
    )

    args = TrainingArguments(
        output_dir                  = str(paths["out"] / "trainer"),
        save_strategy               = "epoch",
        learning_rate               = LR,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        num_train_epochs            = NUM_EPOCHS,
        weight_decay                = 0.01,
        warmup_ratio                = WARMUP_RATIO,
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        logging_dir                 = str(paths["out"] / "logs"),
        logging_steps               = 50,
        report_to                   = [],
        seed                        = seed,
        save_total_limit            = 2,
        max_grad_norm               = 1.0,
        fp16                        = False,
        bf16                        = False,
        gradient_accumulation_steps = grad_accum,
        **strategy_kw,
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = ds_tok["train"],
        eval_dataset    = ds_tok["validation"],
        data_collator   = DataCollatorWithPadding(tokenizer=tok),
        compute_metrics = _compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(paths["model_dir"]))
    tok.save_pretrained(str(paths["model_dir"]))
    print(f"\nModel saved → {paths['model_dir']}")

    # ── Val probabilities + threshold tuning ─────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 7 — Threshold selection on augmented validation split")
    print("=" * 65)

    val_pred   = trainer.predict(ds_tok["validation"])
    val_logits = np.asarray(val_pred.predictions, dtype=np.float64)
    if not np.isfinite(val_logits).all():
        raise RuntimeError(
            "DeBERTa validation logits contain NaN/Inf. Training diverged. "
            "Try --batch-size 4 or check Colab torch/transformers versions."
        )
    val_probs = _softmax(val_logits)[:, 1]
    val_true  = np.asarray(val_pred.label_ids)

    print(
        f"  Val prob stats: min={val_probs.min():.4f}  "
        f"max={val_probs.max():.4f}  mean={val_probs.mean():.4f}"
    )

    tau_hn, val_f1_at_tau = best_threshold(val_probs, val_true)
    print(f"  τ_HN = {tau_hn:.4f}  (val F1 = {val_f1_at_tau:.4f})")
    print(
        "  Sweep: 0.000→1.000 step 0.002. Criterion: max F1. "
        "Tie-break: highest τ. Fixed for ALL eval splits."
    )

    pd.DataFrame({
        "text":              aug_va_df["text"].values,
        "true_label":        val_true,
        "probability_score": val_probs,
    }).to_csv(paths["out"] / "val_predictions.csv", index=False)

    return trainer, tok, tau_hn, val_f1_at_tau, val_probs, val_true


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Evaluate DeBERTa + HardNeg on all splits
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_hardneg(
    paths:   dict[str, Path],
    trainer: Trainer,
    tok:     AutoTokenizer,
    tau_hn:  float,
) -> dict:
    print("\n" + "=" * 65)
    print("STEP 8 — Evaluating DeBERTa + HardNeg on all evaluation splits")
    print("=" * 65)

    results: dict = {}

    # ── IID test ──────────────────────────────────────────────────────────────
    print("\n[IID test]")
    test_df    = _read_csv(paths["test"])
    test_probs = _probs_via_trainer(trainer, tok, test_df["text"].tolist(),
                                    test_df["label"].tolist())
    test_true  = test_df["label"].to_numpy()

    pd.DataFrame({
        "text":              test_df["text"].values,
        "true_label":        test_true,
        "probability_score": test_probs,
        **({} if "source" not in test_df.columns else {"source": test_df["source"].values}),
    }).to_csv(paths["out"] / "test_predictions.csv", index=False)

    r_def   = _eval_at(test_probs, test_true, 0.5)
    r_tuned = _eval_at(test_probs, test_true, tau_hn)
    print(f"  τ=0.500  → F1={r_def.f1:.4f}  recall={r_def.recall:.4f}  FPR={r_def.fpr:.4f}")
    print(f"  τ={tau_hn:.4f} → F1={r_tuned.f1:.4f}  recall={r_tuned.recall:.4f}  FPR={r_tuned.fpr:.4f}  AUC={r_tuned.roc_auc}")
    results["test"] = {"default_0.5": r_def.as_dict(), "tuned": r_tuned.as_dict()}

    # Source-type breakdown (real-origin vs template) for METRIC_COLS compatibility
    if "source_type" in test_df.columns:
        real_mask = (test_df["source_type"] == "real").values
        tmpl_mask = (test_df["source_type"] == "template").values
        results["test"]["real_src_f1"] = (
            round(float(f1_score(test_true[real_mask],
                                 (test_probs[real_mask] >= tau_hn).astype(int),
                                 zero_division=0)), 4)
            if real_mask.any() else None
        )
        results["test"]["template_f1"] = (
            round(float(f1_score(test_true[tmpl_mask],
                                 (test_probs[tmpl_mask] >= tau_hn).astype(int),
                                 zero_division=0)), 4)
            if tmpl_mask.any() else None
        )
    else:
        results["test"]["real_src_f1"] = None
        results["test"]["template_f1"] = None

    # ── Hard-benign ───────────────────────────────────────────────────────────
    print("\n[Hard-benign]")
    hb_df    = pd.read_csv(paths["hard_benign"])
    hb_df["text"] = hb_df["text"].astype(str)
    hb_probs = _probs_via_trainer(trainer, tok, hb_df["text"].tolist(), [0] * len(hb_df))

    hb_save: dict = {
        "text":              hb_df["text"].values,
        "true_label":        0,
        "probability_score": hb_probs,
        "flagged_default":   (hb_probs >= 0.5).astype(int),
        "flagged_tuned":     (hb_probs >= tau_hn).astype(int),
    }
    if "source_type" in hb_df.columns:
        hb_save["source_type"] = hb_df["source_type"].values
    pd.DataFrame(hb_save).to_csv(paths["out"] / "hard_benign_predictions.csv", index=False)

    hb_fpr_default = float((hb_probs >= 0.5).mean())
    hb_fpr_tuned   = float((hb_probs >= tau_hn).mean())

    hb_result: dict = {
        "n":               int(len(hb_df)),
        "fpr_default_0.5": round(hb_fpr_default, 6),
        "fpr_tuned":       round(hb_fpr_tuned, 6),
        "tau":             round(tau_hn, 6),
        "n_flagged_default": int((hb_probs >= 0.5).sum()),
        "n_flagged_tuned":   int((hb_probs >= tau_hn).sum()),
    }

    if "source_type" in hb_df.columns:
        print(f"  source_type counts : {hb_df['source_type'].value_counts().to_dict()}")
        for stype in hb_df["source_type"].unique():
            mask = (hb_df["source_type"] == stype).values
            hb_result[f"fpr_{stype}_origin"] = round(float((hb_probs[mask] >= tau_hn).mean()), 6)
            print(f"  FPR ({stype}) @ τ={tau_hn:.4f} : {hb_result[f'fpr_{stype}_origin']:.4f}")

    print(f"  FPR (overall) @ τ=0.500  : {hb_fpr_default:.4f}")
    print(f"  FPR (overall) @ τ={tau_hn:.4f} : {hb_fpr_tuned:.4f}")
    with open(paths["out"] / "hard_benign_fpr.json", "w") as f:
        json.dump(hb_result, f, indent=2)
    results["hard_benign"] = hb_result

    # ── Obfuscated attacks ────────────────────────────────────────────────────
    print("\n[Obfuscated attacks]")
    obf_df    = pd.read_csv(paths["obfuscated"]).reset_index(drop=True)
    obf_df["text"] = obf_df["text"].astype(str)
    obf_probs = _probs_via_trainer(trainer, tok, obf_df["text"].tolist(), [1] * len(obf_df))
    obf_preds = (obf_probs >= tau_hn).astype(int)

    obf_save: dict = {"text": obf_df["text"].values, "true_label": 1,
                      "probability_score": obf_probs}
    if "obfuscation" in obf_df.columns:
        obf_save["obfuscation"] = obf_df["obfuscation"].values
    pd.DataFrame(obf_save).to_csv(paths["out"] / "obfuscated_attacks_predictions.csv", index=False)

    per_obf: dict = {}
    if "obfuscation" in obf_df.columns:
        for obf_type, grp in obf_df.groupby("obfuscation"):
            idx = grp.index.tolist()
            n   = len(idx)
            det = int(obf_preds[idx].sum())
            per_obf[str(obf_type)] = {"n": n, "n_detected": det,
                                       "recall": round(det / n, 6) if n else 0.0}
            print(f"  {obf_type:20}  n={n:3}  recall={det/n:.4f}")

    total_n   = len(obf_df)
    total_det = int(obf_preds.sum())
    obf_result: dict = {
        "n":              total_n,
        "n_detected":     total_det,
        "overall_recall": round(total_det / total_n, 6) if total_n else 0.0,
        "overall_fnr":    round(1 - total_det / total_n, 6) if total_n else 1.0,
        "tau":            round(tau_hn, 6),
        "per_obfuscation": per_obf,
    }
    print(f"  Overall recall @ τ={tau_hn:.4f} : {obf_result['overall_recall']:.4f}")
    with open(paths["out"] / "obfuscated_attacks_report.json", "w") as f:
        json.dump(obf_result, f, indent=2)
    results["obfuscated"] = obf_result

    # ── Domain OOD ────────────────────────────────────────────────────────────
    print("\n[Domain OOD]")
    dom_df    = pd.read_csv(paths["domain_ood"])
    dom_df["text"] = dom_df["text"].astype(str)
    dom_probs = _probs_via_trainer(trainer, tok, dom_df["text"].tolist(),
                                   dom_df["label"].astype(int).tolist())
    dom_true  = dom_df["label"].astype(int).to_numpy()

    pd.DataFrame({
        "text":              dom_df["text"].values,
        "true_label":        dom_true,
        "probability_score": dom_probs,
        **({} if "domain" not in dom_df.columns else {"domain": dom_df["domain"].values}),
    }).to_csv(paths["out"] / "domain_ood_predictions.csv", index=False)

    dom_r      = _eval_at(dom_probs, dom_true, tau_hn)
    per_domain: dict = {}
    if "domain" in dom_df.columns:
        for dname, grp in dom_df.groupby("domain"):
            idx    = grp.index.tolist()
            g_p    = dom_probs[idx]
            g_t    = dom_true[idx]
            bm, am = g_t == 0, g_t == 1
            per_domain[str(dname)] = {
                "n":      len(idx),
                "fpr":    round(float((g_p[bm] >= tau_hn).mean()), 6) if bm.any() else None,
                "recall": round(float((g_p[am] >= tau_hn).mean()), 6) if am.any() else None,
            }
            print(f"  {dname:15}  FPR={per_domain[str(dname)]['fpr']}  recall={per_domain[str(dname)]['recall']}")

    dom_ood_result: dict = {"aggregate": dom_r.as_dict(), "per_domain": per_domain}
    print(f"  Aggregate  F1={dom_r.f1:.4f}  FPR={dom_r.fpr:.4f}  AUC={dom_r.roc_auc}")
    with open(paths["out"] / "domain_ood_report.json", "w") as f:
        json.dump(dom_ood_result, f, indent=2)
    results["domain_ood"] = dom_ood_result

    # ── Structural OOD ────────────────────────────────────────────────────────
    print("\n[Structural OOD]")
    struct_df    = pd.read_csv(paths["structural_ood"])
    struct_df["text"] = struct_df["text"].astype(str)
    struct_probs = _probs_via_trainer(trainer, tok, struct_df["text"].tolist(),
                                      struct_df["label"].astype(int).tolist())
    struct_true  = struct_df["label"].astype(int).to_numpy()

    pd.DataFrame({
        "text":              struct_df["text"].values,
        "true_label":        struct_true,
        "probability_score": struct_probs,
        **({} if "transform_type" not in struct_df.columns
           else {"transform_type": struct_df["transform_type"].values}),
    }).to_csv(paths["out"] / "structural_ood_probability_scores.csv", index=False)

    struct_r       = _eval_at(struct_probs, struct_true, tau_hn)
    per_transform: dict = {}
    if "transform_type" in struct_df.columns:
        for tr_name, grp in struct_df.groupby("transform_type"):
            idx    = grp.index.tolist()
            g_p    = struct_probs[idx]
            g_t    = struct_true[idx]
            bm, am = g_t == 0, g_t == 1
            per_transform[str(tr_name)] = {
                "n":      len(idx),
                "fpr":    round(float((g_p[bm] >= tau_hn).mean()), 6) if bm.any() else None,
                "recall": round(float((g_p[am] >= tau_hn).mean()), 6) if am.any() else None,
            }
            print(f"  {tr_name:30}  FPR={per_transform[str(tr_name)]['fpr']}  recall={per_transform[str(tr_name)]['recall']}")

    struct_ood_result: dict = {"aggregate": struct_r.as_dict(), "per_transform": per_transform}
    print(f"  Aggregate  F1={struct_r.f1:.4f}  FPR={struct_r.fpr:.4f}  AUC={struct_r.roc_auc}")
    with open(paths["out"] / "structural_ood_report.json", "w") as f:
        json.dump(struct_ood_result, f, indent=2)
    results["structural_ood"] = struct_ood_result

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Baseline DeBERTa at τ_HN
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_baseline_at_tau_hn(
    paths:               dict[str, Path],
    tau_hn:              float,
    skip_model_reload:   bool = False,
) -> dict:
    """
    Evaluate the baseline model at τ_HN.
    Uses cached prob files from outputs/deberta_v3/ where available.
    For domain_ood and obfuscated (no cached probs), loads models/deberta_v3/
    unless --skip-baseline-compare is set.
    """
    print("\n" + "=" * 65)
    print(f"STEP 9 — Baseline DeBERTa at τ_HN = {tau_hn:.4f}")
    print("=" * 65)

    results: dict = {}
    device  = pick_device()

    # ── Splits with cached probability files ─────────────────────────────────
    cached_map = {
        "test":           paths["bl_test_probs"],
        "hard_benign":    paths["bl_hb_probs"],
        "structural_ood": paths["bl_struct_probs"],
    }

    for split_name, cache_path in cached_map.items():
        if not cache_path.exists():
            print(f"  [{split_name}] cache not found at {cache_path} — skipping")
            continue

        df    = pd.read_csv(cache_path)
        probs = df["probability_score"].to_numpy()

        if split_name == "test":
            y_true = df["true_label"].to_numpy()
            r      = _eval_at(probs, y_true, tau_hn)
            print(f"  [test] F1={r.f1:.4f}  recall={r.recall:.4f}  FPR={r.fpr:.4f}  AUC={r.roc_auc}")
            results["test"] = r.as_dict()

        elif split_name == "hard_benign":
            fpr_hn = float((probs >= tau_hn).mean())
            n_flag = int((probs >= tau_hn).sum())
            print(f"  [hard_benign] FPR @ τ_HN={tau_hn:.4f} : {fpr_hn:.4f}  (n_flagged={n_flag}/{len(probs)})")
            hb_result: dict = {"n": len(probs), "fpr_at_tau_hn": round(fpr_hn, 6), "tau_hn": tau_hn}

            hb_raw = pd.read_csv(paths["hard_benign"])
            if "source_type" in hb_raw.columns:
                for stype in hb_raw["source_type"].unique():
                    mask = (hb_raw["source_type"] == stype).values
                    hb_result[f"fpr_{stype}_at_tau_hn"] = round(float((probs[mask] >= tau_hn).mean()), 6)
                    print(f"    ({stype}) FPR : {hb_result[f'fpr_{stype}_at_tau_hn']:.4f}")
            results["hard_benign"] = hb_result

        elif split_name == "structural_ood":
            y_true    = df["true_label"].to_numpy()
            r         = _eval_at(probs, y_true, tau_hn)
            per_tr: dict = {}
            if "transform_type" in df.columns:
                for tr_name, grp in df.groupby("transform_type"):
                    idx    = grp.index.tolist()
                    g_p, g_t = probs[idx], y_true[idx]
                    bm, am = g_t == 0, g_t == 1
                    per_tr[str(tr_name)] = {
                        "fpr":    round(float((g_p[bm] >= tau_hn).mean()), 6) if bm.any() else None,
                        "recall": round(float((g_p[am] >= tau_hn).mean()), 6) if am.any() else None,
                    }
            print(f"  [structural_ood] F1={r.f1:.4f}  FPR={r.fpr:.4f}  AUC={r.roc_auc}")
            results["structural_ood"] = {"aggregate": r.as_dict(), "per_transform": per_tr}

    # ── domain_ood and obfuscated: need model unless already cached ───────────
    dom_cache = paths["bl_cache"] / "domain_ood_probs.csv"
    obf_cache = paths["bl_cache"] / "obfuscated_probs.csv"
    need_model = []
    if not dom_cache.exists():
        need_model.append("domain_ood")
    if not obf_cache.exists():
        need_model.append("obfuscated")

    if need_model and not skip_model_reload:
        if not paths["bl_model"].exists():
            print(f"\n  WARNING: Baseline model not found at {paths['bl_model']}")
            print("  Skipping domain_ood and obfuscated baseline comparison.")
            print("  Re-run without --skip-baseline-compare once models/deberta_v3/ is present.")
        else:
            print(f"\n  Loading baseline model for: {need_model}")
            paths["bl_cache"].mkdir(parents=True, exist_ok=True)

            if "domain_ood" in need_model:
                dom_df = pd.read_csv(paths["domain_ood"])
                dom_df["text"] = dom_df["text"].astype(str)
                dom_probs = _probs_via_model(paths["bl_model"], device,
                                             dom_df["text"].tolist())
                dom_save: dict = {"text": dom_df["text"].values,
                                  "true_label": dom_df["label"].astype(int).values,
                                  "probability_score": dom_probs}
                if "domain" in dom_df.columns:
                    dom_save["domain"] = dom_df["domain"].values
                pd.DataFrame(dom_save).to_csv(dom_cache, index=False)
                print(f"  Cached → {dom_cache}")

            if "obfuscated" in need_model:
                obf_df = pd.read_csv(paths["obfuscated"])
                obf_df["text"] = obf_df["text"].astype(str)
                obf_probs = _probs_via_model(paths["bl_model"], device,
                                             obf_df["text"].tolist())
                obf_save: dict = {"text": obf_df["text"].values, "true_label": 1,
                                  "probability_score": obf_probs}
                if "obfuscation" in obf_df.columns:
                    obf_save["obfuscation"] = obf_df["obfuscation"].values
                pd.DataFrame(obf_save).to_csv(obf_cache, index=False)
                print(f"  Cached → {obf_cache}")

    elif need_model and skip_model_reload:
        print(f"\n  --skip-baseline-compare set; omitting: {need_model}")

    # ── Load domain_ood cache ─────────────────────────────────────────────────
    if dom_cache.exists():
        df    = pd.read_csv(dom_cache)
        probs = df["probability_score"].to_numpy()
        y_t   = df["true_label"].to_numpy()
        r     = _eval_at(probs, y_t, tau_hn)
        per_d: dict = {}
        if "domain" in df.columns:
            for dname, grp in df.groupby("domain"):
                idx   = grp.index.tolist()
                g_p, g_t2 = probs[idx], y_t[idx]
                bm, am = g_t2 == 0, g_t2 == 1
                per_d[str(dname)] = {
                    "fpr":    round(float((g_p[bm] >= tau_hn).mean()), 6) if bm.any() else None,
                    "recall": round(float((g_p[am] >= tau_hn).mean()), 6) if am.any() else None,
                }
        print(f"  [domain_ood] F1={r.f1:.4f}  FPR={r.fpr:.4f}  AUC={r.roc_auc}")
        results["domain_ood"] = {"aggregate": r.as_dict(), "per_domain": per_d}

    # ── Load obfuscated cache ─────────────────────────────────────────────────
    if obf_cache.exists():
        df    = pd.read_csv(obf_cache)
        probs = df["probability_score"].to_numpy()
        preds = (probs >= tau_hn).astype(int)
        per_o: dict = {}
        if "obfuscation" in df.columns:
            for obf_type, grp in df.groupby("obfuscation"):
                idx = grp.index.tolist()
                n   = len(idx)
                det = int(preds[idx].sum())
                per_o[str(obf_type)] = {"n": n, "n_detected": det,
                                         "recall": round(det / n, 6) if n else 0.0}
        total_det_o = int(preds.sum())
        overall_rec = round(total_det_o / len(preds), 6) if len(preds) else 0.0
        print(f"  [obfuscated] recall @ τ_HN={tau_hn:.4f} : {overall_rec:.4f}")
        results["obfuscated"] = {
            "n": len(preds), "n_detected": total_det_o,
            "overall_recall": overall_rec, "per_obfuscation": per_o,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Steps 10 & 11 — Save outputs and write report
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None or v == "N/A":
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def save_outputs(
    paths:         dict[str, Path],
    tau_hn:        float,
    val_f1_at_tau: float,
    hn_results:    dict,
    bl_results:    dict,
    batch_size:    int,
    grad_accum:    int,
    seed:          int,
) -> None:
    print("\n" + "=" * 65)
    print("STEP 10 — Saving structured outputs")
    print("=" * 65)

    # ── Load baseline summary.json ────────────────────────────────────────────
    bl_s: dict = {}
    try:
        with open(paths["bl_out"] / "summary.json") as f:
            bl_s = json.load(f)
    except Exception as e:
        print(f"  Warning: could not load baseline summary.json: {e}")

    bl_tau       = bl_s.get("val_tuned_threshold")
    bl_test_f1   = bl_s.get("test_results", {}).get("val_tuned_threshold", {}).get("f1")
    bl_test_rec  = bl_s.get("test_results", {}).get("val_tuned_threshold", {}).get("recall")
    bl_test_fpr  = bl_s.get("test_results", {}).get("val_tuned_threshold", {}).get("fpr")
    bl_hb_fpr    = bl_s.get("hard_benign_test_fpr", {}).get("fpr_tuned")
    bl_obf_rec   = bl_s.get("obfuscated_attacks", {}).get("overall_recall")
    bl_dom_f1    = bl_s.get("ood_results", {}).get("domain_ood", {}).get("f1")
    bl_dom_fpr   = bl_s.get("ood_results", {}).get("domain_ood", {}).get("fpr")
    bl_struct_f1 = bl_s.get("ood_results", {}).get("structural_ood", {}).get("f1")
    bl_struct_fpr = bl_s.get("ood_results", {}).get("structural_ood", {}).get("fpr")

    # ── thresholds.json ───────────────────────────────────────────────────────
    thresholds = {
        "hardneg_model": {
            "model_hf_id":          MODEL_NAME,
            "model_dir":            str(paths["model_dir"]),
            "tau_hn":               round(tau_hn, 6),
            "val_f1_at_tau_hn":     round(val_f1_at_tau, 6),
            "tuning_note": (
                "τ swept 0.000→1.000 step 0.002. Criterion: max F1 on val split. "
                "Tie-break: highest τ. τ_HN fixed — NOT re-tuned per eval split."
            ),
        },
        "baseline_model": {
            "model_hf_id":   MODEL_NAME,
            "tau_baseline":  bl_tau,
            "val_f1_baseline": bl_s.get("test_results", {}).get("default_threshold_0.5", {}).get("f1"),
        },
        "hyperparams": {
            "model":                  MODEL_NAME,
            "epochs":                 NUM_EPOCHS,
            "batch_size_per_device":  batch_size,
            "gradient_accumulation":  grad_accum,
            "effective_batch_size":   batch_size * grad_accum,
            "lr":                     LR,
            "warmup_ratio":           WARMUP_RATIO,
            "max_length":             MAX_LEN,
            "seed":                   seed,
        },
        "hard_negatives": {
            "train_path": str(paths["hn_train"]),
            "val_path":   str(paths["hn_val"]),
            "n_train":    HN_TRAIN_EXPECTED,
            "n_val":      HN_VAL_EXPECTED,
        },
    }
    with open(paths["out"] / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    # ── summary.csv / per_split_metrics.csv ───────────────────────────────────
    rows: list[dict] = []

    def _row(model: str, split: str, tau: float, **kw):
        rows.append({"model": model, "split": split, "tau": round(tau, 6),
                     **{k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in kw.items()}})

    # Baseline at τ_baseline
    if bl_tau is not None:
        _row("baseline_deberta", "test",           bl_tau,
             f1=bl_test_f1, recall=bl_test_rec, fpr=bl_test_fpr)
        _row("baseline_deberta", "hard_benign",    bl_tau, hb_fpr=bl_hb_fpr)
        _row("baseline_deberta", "obfuscated",     bl_tau, recall=bl_obf_rec)
        _row("baseline_deberta", "domain_ood",     bl_tau, f1=bl_dom_f1, fpr=bl_dom_fpr)
        _row("baseline_deberta", "structural_ood", bl_tau, f1=bl_struct_f1, fpr=bl_struct_fpr)

    # Baseline at τ_HN
    if "test" in bl_results:
        bt = bl_results["test"]
        _row("baseline_at_tau_hn", "test", tau_hn,
             f1=bt.get("f1"), recall=bt.get("recall"), fpr=bt.get("fpr"),
             roc_auc=bt.get("roc_auc"))
    if "hard_benign" in bl_results:
        _row("baseline_at_tau_hn", "hard_benign", tau_hn,
             hb_fpr=bl_results["hard_benign"].get("fpr_at_tau_hn"))
    if "obfuscated" in bl_results:
        _row("baseline_at_tau_hn", "obfuscated", tau_hn,
             recall=bl_results["obfuscated"].get("overall_recall"))
    if "domain_ood" in bl_results:
        ba = bl_results["domain_ood"]["aggregate"]
        _row("baseline_at_tau_hn", "domain_ood", tau_hn,
             f1=ba.get("f1"), fpr=ba.get("fpr"), roc_auc=ba.get("roc_auc"))
    if "structural_ood" in bl_results:
        sa = bl_results["structural_ood"]["aggregate"]
        _row("baseline_at_tau_hn", "structural_ood", tau_hn,
             f1=sa.get("f1"), fpr=sa.get("fpr"), roc_auc=sa.get("roc_auc"))

    # HardNeg at τ_HN
    hn_t = hn_results.get("test", {}).get("tuned", {})
    if hn_t:
        _row("deberta_hardneg", "test", tau_hn,
             f1=hn_t.get("f1"), recall=hn_t.get("recall"), fpr=hn_t.get("fpr"),
             roc_auc=hn_t.get("roc_auc"))
    if "hard_benign" in hn_results:
        _row("deberta_hardneg", "hard_benign", tau_hn,
             hb_fpr=hn_results["hard_benign"].get("fpr_tuned"))
    if "obfuscated" in hn_results:
        _row("deberta_hardneg", "obfuscated", tau_hn,
             recall=hn_results["obfuscated"].get("overall_recall"))
    if "domain_ood" in hn_results:
        da = hn_results["domain_ood"]["aggregate"]
        _row("deberta_hardneg", "domain_ood", tau_hn,
             f1=da.get("f1"), fpr=da.get("fpr"), roc_auc=da.get("roc_auc"))
    if "structural_ood" in hn_results:
        sa2 = hn_results["structural_ood"]["aggregate"]
        _row("deberta_hardneg", "structural_ood", tau_hn,
             f1=sa2.get("f1"), fpr=sa2.get("fpr"), roc_auc=sa2.get("roc_auc"))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(paths["out"] / "summary.csv",           index=False)
    summary_df.to_csv(paths["out"] / "per_split_metrics.csv", index=False)

    # ── summary.json (mirrors deberta_v3.py structure for pipeline compatibility)
    hn_t_res   = hn_results.get("test", {})
    hn_hb_res  = hn_results.get("hard_benign", {})
    hn_obf_res = hn_results.get("obfuscated", {})
    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"
    summary_json = {
        "model":          MODEL_NAME + "+HardNeg",
        "model_saved_to": str(paths["model_dir"]),
        "device":         _device,
        "hyperparams": {
            "epochs":                      NUM_EPOCHS,
            "batch_size":                  batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size":        batch_size * grad_accum,
            "lr":                          LR,
            "warmup_ratio":                WARMUP_RATIO,
            "seed":                        seed,
            "max_len":                     MAX_LEN,
        },
        "hard_negatives": {
            "n_train": HN_TRAIN_EXPECTED,
            "n_val":   HN_VAL_EXPECTED,
        },
        "val_tuned_threshold": round(tau_hn, 6),
        "test_results": {
            "default_threshold_0.5": hn_t_res.get("default_0.5", {}),
            "val_tuned_threshold":   hn_t_res.get("tuned", {}),
        },
        "hard_benign_test_fpr": {
            "fpr_default":          hn_hb_res.get("fpr_default_0.5"),
            "fpr_tuned":            hn_hb_res.get("fpr_tuned"),
            "fpr_real_origin":      hn_hb_res.get("fpr_real_origin"),
            "fpr_curated_origin":   hn_hb_res.get("fpr_curated_origin"),
        },
        "obfuscated_attacks": {
            "overall_recall": hn_obf_res.get("overall_recall"),
            "overall_fnr":    hn_obf_res.get("overall_fnr"),
        },
        "domain_ood":    hn_results.get("domain_ood", {}),
        "structural_ood": hn_results.get("structural_ood", {}),
        "baseline_comparison": {
            "baseline_tau":          bl_tau,
            "baseline_hb_fpr_tuned": bl_hb_fpr,
            "baseline_test_f1":      bl_test_f1,
        },
    }
    with open(paths["out"] / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"  summary.json → {paths['out'] / 'summary.json'}")

    # ── per_domain.csv ─────────────────────────────────────────────────────────
    dom_rows: list[dict] = []
    for mlabel, per_d in [
        ("deberta_hardneg",    hn_results.get("domain_ood",  {}).get("per_domain", {})),
        ("baseline_at_tau_hn", bl_results.get("domain_ood",  {}).get("per_domain", {})),
    ]:
        for dname, m in per_d.items():
            dom_rows.append({"model": mlabel, "domain": dname, "tau": tau_hn, **m})
    pd.DataFrame(dom_rows).to_csv(paths["out"] / "per_domain.csv", index=False)

    # ── per_transform.csv ──────────────────────────────────────────────────────
    tr_rows: list[dict] = []
    for mlabel, per_t in [
        ("deberta_hardneg",    hn_results.get("structural_ood", {}).get("per_transform", {})),
        ("baseline_at_tau_hn", bl_results.get("structural_ood", {}).get("per_transform", {})),
    ]:
        for tr_name, m in per_t.items():
            tr_rows.append({"model": mlabel, "transform_type": tr_name, "tau": tau_hn, **m})
    pd.DataFrame(tr_rows).to_csv(paths["out"] / "per_transform.csv", index=False)

    # ── Print all saved file paths ─────────────────────────────────────────────
    expected = [
        paths["out"] / "thresholds.json",
        paths["out"] / "summary.json",
        paths["out"] / "summary.csv",
        paths["out"] / "per_split_metrics.csv",
        paths["out"] / "per_domain.csv",
        paths["out"] / "per_transform.csv",
        paths["out"] / "val_predictions.csv",
        paths["out"] / "test_predictions.csv",
        paths["out"] / "hard_benign_predictions.csv",
        paths["out"] / "obfuscated_attacks_predictions.csv",
        paths["out"] / "domain_ood_predictions.csv",
        paths["out"] / "structural_ood_probability_scores.csv",
        paths["out"] / "hard_benign_fpr.json",
        paths["out"] / "obfuscated_attacks_report.json",
        paths["out"] / "domain_ood_report.json",
        paths["out"] / "structural_ood_report.json",
    ]
    print("\nSaved files:")
    for p in expected:
        mark = "✓" if p.exists() else "✗ MISSING"
        print(f"  [{mark}] {p}")


def write_report(
    paths:         dict[str, Path],
    tau_hn:        float,
    val_f1_at_tau: float,
    hn_results:    dict,
    bl_results:    dict,
    batch_size:    int,
    grad_accum:    int,
    seed:          int,
) -> None:
    print("\n" + "=" * 65)
    print("STEP 11 — Writing report.md")
    print("=" * 65)

    # Load baseline summary
    bl_s: dict = {}
    try:
        with open(paths["bl_out"] / "summary.json") as f:
            bl_s = json.load(f)
    except Exception:
        pass

    bl_tau      = bl_s.get("val_tuned_threshold")
    bl_test_f1  = bl_s.get("test_results", {}).get("val_tuned_threshold", {}).get("f1")
    bl_test_rec = bl_s.get("test_results", {}).get("val_tuned_threshold", {}).get("recall")
    bl_hb_fpr   = bl_s.get("hard_benign_test_fpr", {}).get("fpr_tuned")
    bl_obf_rec  = bl_s.get("obfuscated_attacks", {}).get("overall_recall")
    bl_dom_f1   = bl_s.get("ood_results", {}).get("domain_ood", {}).get("f1")
    bl_str_f1   = bl_s.get("ood_results", {}).get("structural_ood", {}).get("f1")
    bl_str_fpr  = bl_s.get("ood_results", {}).get("structural_ood", {}).get("fpr")

    hn_t      = hn_results.get("test", {}).get("tuned", {})
    hn_hb     = hn_results.get("hard_benign", {})
    hn_obf    = hn_results.get("obfuscated", {})
    hn_dom    = hn_results.get("domain_ood", {}).get("aggregate", {})
    hn_struct = hn_results.get("structural_ood", {}).get("aggregate", {})

    bl_hn_test   = bl_results.get("test", {})
    bl_hn_hb     = bl_results.get("hard_benign", {})
    bl_hn_obf    = bl_results.get("obfuscated", {})
    bl_hn_dom    = bl_results.get("domain_ood", {}).get("aggregate", {})
    bl_hn_struct = bl_results.get("structural_ood", {}).get("aggregate", {})

    # ── Interpret hard-benign FPR ─────────────────────────────────────────────
    hn_hb_fpr_v = hn_hb.get("fpr_tuned")
    bl_hn_hb_v  = bl_hn_hb.get("fpr_at_tau_hn")

    if isinstance(hn_hb_fpr_v, float) and isinstance(bl_hb_fpr, float):
        delta_vs_baseline = hn_hb_fpr_v - bl_hb_fpr
        if abs(delta_vs_baseline) < 0.01:
            hb_interp = "Hard-benign FPR shows negligible change (<1 pp) versus baseline at τ_baseline."
        elif delta_vs_baseline < 0:
            hb_interp = (f"Hard-benign FPR improved by {abs(delta_vs_baseline):.4f} versus baseline at τ_baseline. "
                         "This may reflect training, threshold shift, or both.")
        else:
            hb_interp = f"Hard-benign FPR worsened by {abs(delta_vs_baseline):.4f} versus baseline at τ_baseline."
    else:
        hb_interp = "Hard-benign FPR comparison vs baseline unavailable."

    if isinstance(hn_hb_fpr_v, float) and isinstance(bl_hn_hb_v, float):
        delta_training = hn_hb_fpr_v - bl_hn_hb_v
        if abs(delta_training) < 0.005:
            train_effect = (
                "Comparing +HardNeg vs baseline@τ_HN: difference is <0.5 pp. "
                "Training had negligible effect on hard-benign FPR at this threshold."
            )
        elif delta_training < 0:
            train_effect = (
                f"Comparing +HardNeg vs baseline@τ_HN: training reduced hard-benign FPR "
                f"by {abs(delta_training):.4f}. This is a genuine training effect, not a threshold artefact."
            )
        else:
            train_effect = (
                f"Comparing +HardNeg vs baseline@τ_HN: hard-benign FPR increased by "
                f"{abs(delta_training):.4f} relative to baseline at the same threshold. "
                "Training on hard-negatives did not reduce FPR and may have increased it."
            )
    else:
        train_effect = "Baseline@τ_HN hard-benign FPR not available — training effect vs threshold shift attribution incomplete."

    # Check IID trade-off
    if isinstance(hn_t.get("f1"), float) and isinstance(bl_test_f1, float):
        f1_delta = hn_t["f1"] - bl_test_f1
        if abs(f1_delta) < 0.005:
            iid_note = "IID test F1 shows negligible change (<0.5 pp) from hard-negative augmentation."
        elif f1_delta < 0:
            iid_note = f"IID test F1 decreased by {abs(f1_delta):.4f} — slight trade-off from augmentation."
        else:
            iid_note = f"IID test F1 improved by {abs(f1_delta):.4f} from hard-negative augmentation."
    else:
        iid_note = "IID F1 comparison unavailable."

    lines = [
        "# DeBERTa-v3-FT + HardNeg Augmentation Report",
        "",
        "## Training Configuration",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| Model | `{MODEL_NAME}` |",
        f"| Epochs | {NUM_EPOCHS} |",
        f"| Batch size (per device) | {batch_size} |",
        f"| Gradient accumulation steps | {grad_accum} |",
        f"| Effective batch size | {batch_size * grad_accum} |",
        f"| Learning rate | {LR} |",
        f"| Warmup ratio | {WARMUP_RATIO} |",
        f"| Max sequence length | {MAX_LEN} |",
        f"| Weight decay | 0.01 |",
        f"| Max grad norm | 1.0 |",
        f"| fp16 / bf16 | False / False |",
        f"| Seed | {seed} |",
        f"| Model saved to | `models/deberta_v3_hardneg/` |",
        f"| Outputs saved to | `outputs/deberta_v3_hardneg/` |",
        "",
        "Hyperparameters are identical to baseline `deberta_v3.py`.",
        "",
        "---",
        "",
        "## Hard-Negative Augmentation",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Train file | `data/pids_bench_v3/hard_negative_train.csv` |",
        f"| Val file | `data/pids_bench_v3/hard_negative_val.csv` |",
        f"| Train count (verified) | {HN_TRAIN_EXPECTED} |",
        f"| Val count (verified) | {HN_VAL_EXPECTED} |",
        f"| All labels | 0 (benign only — confirmed by assertion) |",
        f"| Source type | `hardneg_curated` (second template batch, offset=75) |",
        f"| Categories | guardrail_design, imperative_benign, redteam_framing, attack_analysis, policy_meta, security_education, fiction_roleplay_benign, technical_documentation |",
        "",
        "### Deduplication Safety Check",
        "",
        "| Eval split | n | Overlap with hard-negatives |",
        "| --- | --- | --- |",
        "| test | 3918 | 0 |",
        "| hard_benign | 1472 | 0 |",
        "| domain_ood | 2000 | 0 |",
        "| structural_ood | 1998 | 0 |",
        "",
        "No overlap found. Evaluation splits were not modified.",
        "",
        "---",
        "",
        "## Threshold Selection",
        "",
        f"**τ_HN = {tau_hn:.4f}**  (val F1 = {val_f1_at_tau:.4f})",
        "",
        "- Sweep: 0.000 → 1.000, step 0.002",
        "- Criterion: maximize binary F1 (pos_label = 1 = injection)",
        "- Tie-break: highest τ",
        "- Val split: augmented (base val + 116 hard-negatives)",
        "- τ_HN is **fixed for all evaluation sets** — not re-tuned per split",
        "",
        "---",
        "",
        "## Results",
        "",
        "### IID Test",
        "",
        "| Model | τ | F1 | Recall | FPR | AUC |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| Baseline DeBERTa | {_fmt(bl_tau)} | {_fmt(bl_test_f1)} | {_fmt(bl_test_rec)} | — | — |",
        f"| Baseline @ τ_HN | {_fmt(tau_hn)} | {_fmt(bl_hn_test.get('f1'))} | {_fmt(bl_hn_test.get('recall'))} | {_fmt(bl_hn_test.get('fpr'))} | {_fmt(bl_hn_test.get('roc_auc'))} |",
        f"| DeBERTa + HardNeg | {_fmt(tau_hn)} | {_fmt(hn_t.get('f1'))} | {_fmt(hn_t.get('recall'))} | {_fmt(hn_t.get('fpr'))} | {_fmt(hn_t.get('roc_auc'))} |",
        "",
        "### Hard-Benign FPR",
        "",
        "| Model | τ | FPR (overall) | FPR (real origin) | FPR (curated origin) |",
        "| --- | --- | --- | --- | --- |",
        f"| Baseline DeBERTa | {_fmt(bl_tau)} | {_fmt(bl_hb_fpr)} | — | — |",
        f"| Baseline @ τ_HN | {_fmt(tau_hn)} | {_fmt(bl_hn_hb.get('fpr_at_tau_hn'))} | {_fmt(bl_hn_hb.get('fpr_real_at_tau_hn'))} | {_fmt(bl_hn_hb.get('fpr_curated_at_tau_hn'))} |",
        f"| DeBERTa + HardNeg | {_fmt(tau_hn)} | {_fmt(hn_hb.get('fpr_tuned'))} | {_fmt(hn_hb.get('fpr_real_origin'))} | {_fmt(hn_hb.get('fpr_curated_origin'))} |",
        "",
        "### Obfuscated Attacks Recall",
        "",
        "| Model | τ | Overall Recall |",
        "| --- | --- | --- |",
        f"| Baseline DeBERTa | {_fmt(bl_tau)} | {_fmt(bl_obf_rec)} |",
        f"| Baseline @ τ_HN | {_fmt(tau_hn)} | {_fmt(bl_hn_obf.get('overall_recall'))} |",
        f"| DeBERTa + HardNeg | {_fmt(tau_hn)} | {_fmt(hn_obf.get('overall_recall'))} |",
        "",
        "### Domain OOD",
        "",
        "| Model | τ | F1 | FPR | AUC |",
        "| --- | --- | --- | --- | --- |",
        f"| Baseline DeBERTa | {_fmt(bl_tau)} | {_fmt(bl_dom_f1)} | — | — |",
        f"| Baseline @ τ_HN | {_fmt(tau_hn)} | {_fmt(bl_hn_dom.get('f1'))} | {_fmt(bl_hn_dom.get('fpr'))} | {_fmt(bl_hn_dom.get('roc_auc'))} |",
        f"| DeBERTa + HardNeg | {_fmt(tau_hn)} | {_fmt(hn_dom.get('f1'))} | {_fmt(hn_dom.get('fpr'))} | {_fmt(hn_dom.get('roc_auc'))} |",
        "",
        "### Structural OOD",
        "",
        "| Model | τ | F1 | FPR | AUC |",
        "| --- | --- | --- | --- | --- |",
        f"| Baseline DeBERTa | {_fmt(bl_tau)} | {_fmt(bl_str_f1)} | {_fmt(bl_str_fpr)} | — |",
        f"| Baseline @ τ_HN | {_fmt(tau_hn)} | {_fmt(bl_hn_struct.get('f1'))} | {_fmt(bl_hn_struct.get('fpr'))} | {_fmt(bl_hn_struct.get('roc_auc'))} |",
        f"| DeBERTa + HardNeg | {_fmt(tau_hn)} | {_fmt(hn_struct.get('f1'))} | {_fmt(hn_struct.get('fpr'))} | {_fmt(hn_struct.get('roc_auc'))} |",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        f"**IID test:** {iid_note}",
        "",
        f"**Hard-benign FPR:** {hb_interp}",
        "",
        f"**Training vs threshold attribution:** {train_effect}",
        "",
        "**Comparison methodology:**",
        "The three-row tables enable precise attribution:",
        "1. *Baseline DeBERTa at τ_baseline* — baseline at its own optimal threshold (fair baseline).",
        "2. *Baseline @ τ_HN* — same model at the augmented model's threshold. "
           "Difference from (1) is the **threshold effect** alone.",
        "3. *DeBERTa + HardNeg at τ_HN* — augmented model at its own optimal threshold. "
           "Difference from (2) is the **pure training effect** of hard-negative augmentation.",
        "",
        "---",
        "",
        "## Notes",
        "",
        "- Frozen evaluation splits (test, hard_benign, domain_ood, structural_ood) were not modified.",
        "- Hard-negatives verified at runtime: 419 train / 116 val, all label=0, zero overlap with eval splits.",
        "- τ_HN selected on the augmented validation split and held fixed for all evaluation sets.",
        f"- Baseline cached probabilities used from `outputs/deberta_v3/` for test, hard_benign, structural_ood.",
        "- For domain_ood and obfuscated baseline probs: loaded from `models/deberta_v3/` on first run,",
        "  then cached to `outputs/deberta_v3_hardneg/baseline_probs_cache/` for subsequent runs.",
        "",
        "_Generated by `src/baselines/deberta_v3_hardneg.py`_",
    ]

    report_path = paths["out"] / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Public run_train entry point (called by train_multi_seed.py orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

def _build_metrics_dict(tau_hn: float, hn_results: dict) -> dict:
    """Flatten evaluate_hardneg results into the METRIC_COLS-compatible dict."""
    test_t  = hn_results.get("test", {}).get("tuned", {})
    hb      = hn_results.get("hard_benign", {})
    obf     = hn_results.get("obfuscated", {})
    dom     = hn_results.get("domain_ood", {})
    per_dom = dom.get("per_domain", {})
    struct  = hn_results.get("structural_ood", {})
    per_tr  = struct.get("per_transform", {})
    return {
        "tau":               round(tau_hn, 6),
        "IID_F1":            test_t.get("f1"),
        "real_src_F1":       hn_results.get("test", {}).get("real_src_f1"),
        "template_F1":       hn_results.get("test", {}).get("template_f1"),
        "IID_recall":        test_t.get("recall"),
        "IID_precision":     test_t.get("precision"),
        "test_FPR":          test_t.get("fpr"),
        "ROC_AUC":           test_t.get("roc_auc"),
        "hb_FPR_agg":        hb.get("fpr_tuned"),
        "hb_FPR_real":       hb.get("fpr_real_origin"),
        "hb_FPR_curated":    hb.get("fpr_curated_origin"),
        "obf_recall":        obf.get("overall_recall"),
        "domain_OOD_F1_agg": dom.get("aggregate", {}).get("f1"),
        "medical_FPR":       per_dom.get("medical", {}).get("fpr"),
        "medical_recall":    per_dom.get("medical", {}).get("recall"),
        "legal_FPR":         per_dom.get("legal",   {}).get("fpr"),
        "legal_recall":      per_dom.get("legal",   {}).get("recall"),
        "finance_FPR":       per_dom.get("finance", {}).get("fpr"),
        "finance_recall":    per_dom.get("finance", {}).get("recall"),
        "code_FPR":          per_dom.get("code",    {}).get("fpr"),
        "code_recall":       per_dom.get("code",    {}).get("recall"),
        "struct_OOD_F1_agg":  struct.get("aggregate", {}).get("f1"),
        "struct_OOD_FPR_agg": struct.get("aggregate", {}).get("fpr"),
        "json_wrap_FPR":         per_tr.get("json_wrap",                {}).get("fpr"),
        "json_wrap_recall":      per_tr.get("json_wrap",                {}).get("recall"),
        "prompt_dilute_FPR":     per_tr.get("prompt_dilute",            {}).get("fpr"),
        "prompt_dilute_recall":  per_tr.get("prompt_dilute",            {}).get("recall"),
        "prefix_mimic_FPR":      per_tr.get("instruction_prefix_mimic", {}).get("fpr"),
        "prefix_mimic_recall":   per_tr.get("instruction_prefix_mimic", {}).get("recall"),
    }


def run_train(
    seed:       int   = SEED,
    out_dir:    Optional[Path] = None,
    repo_root:  str   = ".",
    batch_size: int   = DEFAULT_BATCH,
    grad_accum: int   = DEFAULT_ACCUM,
    dry_run:    bool  = False,
) -> dict:
    """Callable entry point for train_multi_seed.py orchestration.

    When out_dir is provided (multi-seed mode):
      - outputs go to out_dir/
      - model is saved to out_dir/model/
      - baseline predictions are read from
        outputs/multi_seed_runs/deberta/seed_{seed}/ when that dir is non-empty
        (seeds 42, 123, 2024 have full per-seed dirs; seeds 13/7777 are empty
        so baseline@τ_HN comparison is skipped for those seeds with a warning)

    dry_run=True runs all preflight checks and exits without training.

    Returns a METRIC_COLS-compatible dict (empty dict on dry_run).
    """
    repo_root_path = Path(repo_root).resolve()
    out_path = Path(out_dir) if out_dir is not None else None

    # Route baseline predictions to the matching per-seed dir when available
    multi_seed_bl_dir: Optional[Path] = None
    if out_path is not None:
        candidate = (
            repo_root_path / "outputs" / "multi_seed_runs" / "deberta" / f"seed_{seed}"
        )
        if candidate.exists() and any(candidate.iterdir()):
            multi_seed_bl_dir = candidate
        else:
            print(
                f"  [info] No per-seed baseline dir at {candidate} — "
                "baseline@τ_HN comparison will be skipped for this seed."
            )

    paths = _make_paths(repo_root_path, out_dir=out_path, multi_seed_bl_dir=multi_seed_bl_dir)

    print(f"\nRepo root : {repo_root_path}")
    print(f"Device    : {pick_device()}")
    confirm_hyperparams(batch_size, grad_accum, seed)
    hn_tr, hn_va = verify_hard_negatives(paths)
    dedup_safety_check(hn_tr, paths)

    if dry_run:
        print(f"\n[DRY RUN] seed={seed}")
        print(f"  out         : {paths['out']}")
        print(f"  model_dir   : {paths['model_dir']}")
        print(f"  bl_out      : {paths['bl_out']}")
        print(f"  bl_model    : {paths['bl_model']}")
        print(f"  bl_model exists : {paths['bl_model'].exists()}")
        print("  Preflight passed. Skipping training.")
        return {}

    trainer, tok, tau_hn, val_f1_at_tau, val_probs, val_true = train_and_tune(
        paths, hn_tr=hn_tr, hn_va=hn_va,
        batch_size=batch_size, grad_accum=grad_accum, seed=seed,
    )

    hn_results = evaluate_hardneg(paths, trainer, tok, tau_hn)

    device = pick_device()
    del trainer
    _free_gpu(device)

    bl_results = evaluate_baseline_at_tau_hn(
        paths, tau_hn=tau_hn, skip_model_reload=False,
    )

    save_outputs(paths, tau_hn, val_f1_at_tau, hn_results, bl_results,
                 batch_size, grad_accum, seed)
    write_report(paths, tau_hn, val_f1_at_tau, hn_results, bl_results,
                 batch_size, grad_accum, seed)

    return _build_metrics_dict(tau_hn, hn_results)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeBERTa-v3-base + HardNeg training and evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-root", default=".",
        help=(
            "Path to repository root. "
            "Default '.' works when running from repo root. "
            "For Colab: --repo-root /content/Prompt-Injection-Detector-System"
        ),
    )
    parser.add_argument("--seed",       type=int, default=SEED,
                        help="Random seed (default 42)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                        help="Per-device batch size (default 8)")
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_ACCUM,
                        help="Gradient accumulation steps (default 2; effective batch = 16)")
    parser.add_argument(
        "--skip-baseline-compare", action="store_true",
        help=(
            "Skip reloading baseline model for splits without cached probs. "
            "domain_ood and obfuscated baseline comparison will be omitted. "
            "Useful if models/deberta_v3/ is not present or memory is tight."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Run preflight checks (data integrity, deduplication) and print "
            "all resolved paths, then exit without training. Use this to "
            "smoke-test your setup before an A100 session."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    paths     = _make_paths(repo_root)

    print(f"\nRepo root : {repo_root}")
    print(f"Device    : {pick_device()}")

    # Step 1
    confirm_hyperparams(args.batch_size, args.grad_accum, args.seed)

    # Step 3
    hn_tr, hn_va = verify_hard_negatives(paths)

    # Step 5
    dedup_safety_check(hn_tr, paths)

    if args.dry_run:
        print(f"\n[DRY RUN] out={paths['out']}  bl_out={paths['bl_out']}")
        print(f"  model_dir         : {paths['model_dir']}")
        print(f"  bl_model          : {paths['bl_model']}")
        print(f"  bl_model exists   : {paths['bl_model'].exists()}")
        print(f"  bl_test_probs     : {paths['bl_test_probs'].exists()}")
        print(f"  bl_hb_probs       : {paths['bl_hb_probs'].exists()}")
        print("  Preflight passed. Skipping training (--dry-run).")
        return

    # Steps 6 & 7
    trainer, tok, tau_hn, val_f1_at_tau, val_probs, val_true = train_and_tune(
        paths,
        hn_tr      = hn_tr,
        hn_va      = hn_va,
        batch_size = args.batch_size,
        grad_accum = args.grad_accum,
        seed       = args.seed,
    )

    # Step 8
    hn_results = evaluate_hardneg(paths, trainer, tok, tau_hn)

    # Free trainer/model memory before loading baseline
    device = pick_device()
    del trainer
    _free_gpu(device)

    # Step 9
    bl_results = evaluate_baseline_at_tau_hn(
        paths,
        tau_hn            = tau_hn,
        skip_model_reload = args.skip_baseline_compare,
    )

    # Steps 10 & 11
    save_outputs(paths, tau_hn, val_f1_at_tau, hn_results, bl_results,
                 args.batch_size, args.grad_accum, args.seed)
    write_report(paths, tau_hn, val_f1_at_tau, hn_results, bl_results,
                 args.batch_size, args.grad_accum, args.seed)

    # ── Final summary print ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DONE — Key metric comparison")
    print("=" * 65)
    print(f"\nτ_HN = {tau_hn:.4f}")

    bl_s: dict = {}
    try:
        with open(paths["bl_out"] / "summary.json") as f:
            bl_s = json.load(f)
    except Exception:
        pass

    bl_tau_v    = bl_s.get("val_tuned_threshold", float("nan"))
    bl_f1_v     = bl_s.get("test_results", {}).get("val_tuned_threshold", {}).get("f1", float("nan"))
    bl_hb_v     = bl_s.get("hard_benign_test_fpr", {}).get("fpr_tuned", float("nan"))
    bl_obf_v    = bl_s.get("obfuscated_attacks", {}).get("overall_recall", float("nan"))

    bl_hn_f1_v  = bl_results.get("test", {}).get("f1", float("nan"))
    bl_hn_hb_v  = bl_results.get("hard_benign", {}).get("fpr_at_tau_hn", float("nan"))
    bl_hn_obf_v = bl_results.get("obfuscated", {}).get("overall_recall", float("nan"))

    hn_f1_v     = hn_results.get("test", {}).get("tuned", {}).get("f1", float("nan"))
    hn_hb_v     = hn_results.get("hard_benign", {}).get("fpr_tuned", float("nan"))
    hn_obf_v    = hn_results.get("obfuscated", {}).get("overall_recall", float("nan"))

    header = f"\n{'Model':<30}  {'τ':>7}  {'Test F1':>8}  {'HB FPR':>8}  {'Obf Recall':>10}"
    print(header)
    print("-" * 72)
    print(f"{'Baseline DeBERTa':<30}  {bl_tau_v:>7.4f}  {bl_f1_v:>8.4f}  {bl_hb_v:>8.4f}  {bl_obf_v:>10.4f}")
    print(f"{'Baseline @ τ_HN':<30}  {tau_hn:>7.4f}  {bl_hn_f1_v:>8.4f}  {bl_hn_hb_v:>8.4f}  {bl_hn_obf_v:>10.4f}")
    print(f"{'DeBERTa + HardNeg':<30}  {tau_hn:>7.4f}  {hn_f1_v:>8.4f}  {hn_hb_v:>8.4f}  {hn_obf_v:>10.4f}")

    print(f"\nOutput directory : {paths['out']}")
    print(f"Model saved to   : {paths['model_dir']}")
    print("\nKey output files:")
    for fname in ("thresholds.json", "summary.csv", "per_split_metrics.csv",
                  "per_domain.csv", "per_transform.csv", "report.md"):
        print(f"  {paths['out'] / fname}")


if __name__ == "__main__":
    main()
