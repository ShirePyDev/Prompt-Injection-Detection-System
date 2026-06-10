#!/usr/bin/env python3
"""
Generate per-split probability score CSVs for models where only binary predictions
were saved, and for splits where inference was not previously run.

Covered models
--------------
tfidf_logreg  (always):  loads saved joblib pipeline, runs predict_proba across all splits.
distilbert    (--transformers flag): loads saved HF model, infers structural_ood probs.
deberta_v3    (--transformers flag): loads saved HF model, infers structural_ood probs.

Output layout
-------------
outputs/tfidf_baseline/val_probability_scores.csv        (true_label, probability_score)
outputs/tfidf_baseline/test_probability_scores.csv
outputs/tfidf_baseline/hard_benign_probability_scores.csv
outputs/tfidf_baseline/structural_ood_probability_scores.csv
outputs/distilbert_baseline/structural_ood_probability_scores.csv   (--transformers only)
outputs/deberta_v3/structural_ood_probability_scores.csv            (--transformers only)

All output CSVs follow the schema expected by eval/threshold_analysis.py and
eval/plot_roc_pr_curves.py:
    text (str), true_label (int 0/1), probability_score (float 0..1)

Usage
-----
  python eval/generate_probability_scores.py               # TF-IDF only
  python eval/generate_probability_scores.py --transformers # TF-IDF + transformer models
  python eval/generate_probability_scores.py --model tfidf_logreg
  python eval/generate_probability_scores.py --model deberta_v3 --transformers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR  = ROOT / "data" / "pids_bench_v3"
EVAL_DIR  = DATA_DIR / "eval_subsets"
OOD_DIR   = DATA_DIR / "ood"

# Splits with label availability
TFIDF_SPLITS: list[tuple[str, Path, str | None]] = [
    # (name,  data_path,                              label_col)
    ("val",            DATA_DIR / "val.csv",                          "label"),
    ("test",           DATA_DIR / "test.csv",                         "label"),
    ("hard_benign",    EVAL_DIR / "hard_benign_test.csv",              "label"),
    ("structural_ood", OOD_DIR  / "structural_ood.csv",               "label"),
    ("domain_ood",     OOD_DIR  / "domain_ood.csv",                   "label"),
    ("obfuscated",     EVAL_DIR / "obfuscated_attacks.csv",            "label"),
]


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF / Logistic Regression
# ──────────────────────────────────────────────────────────────────────────────

def _patch_logreg_version(pipe) -> None:
    """
    Apply forward-compatibility patch when a Pipeline was saved with a newer
    sklearn version than the one currently installed.

    Specifically: sklearn ≥ 1.8 removed the `multi_class` attribute from
    LogisticRegression, but sklearn 1.7.x's predict_proba still reads it.
    Setting it to 'auto' restores the original behaviour without changing
    any model weights or decision boundaries.
    """
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from packaging.version import Version

    clf = pipe.steps[-1][1]
    if not isinstance(clf, LogisticRegression):
        return

    installed = Version(sklearn.__version__)
    if installed < Version("1.8.0") and not hasattr(clf, "multi_class"):
        clf.multi_class = "auto"
        print("  [COMPAT] Applied multi_class='auto' patch for sklearn version mismatch.")


def run_tfidf(force: bool = False) -> None:
    """Load the saved TF-IDF + LR pipeline and generate probability CSVs."""
    import joblib

    model_path = ROOT / "models" / "tfidf_logreg" / "tfidf_logreg.joblib"
    if not model_path.exists():
        print(f"[SKIP] TF-IDF model not found at {model_path}. "
              "Run src/baselines/tfidf_logreg.py first.")
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = joblib.load(model_path)
    _patch_logreg_version(pipe)
    print(f"Loaded TF-IDF pipeline from {model_path.relative_to(ROOT)}")

    out_dir = ROOT / "outputs" / "tfidf_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data_path, label_col in TFIDF_SPLITS:
        out_csv = out_dir / f"{split_name}_probability_scores.csv"
        if out_csv.exists() and not force:
            print(f"  [EXISTS] {out_csv.relative_to(ROOT)} — skipping (pass --force to overwrite)")
            continue
        if not data_path.exists():
            print(f"  [SKIP] {data_path.relative_to(ROOT)} not found")
            continue

        df = pd.read_csv(data_path)
        lc = label_col or "label"
        if lc not in df.columns:
            print(f"  [SKIP] {split_name}: label column '{lc}' not in {data_path.name}")
            continue
        if "text" not in df.columns:
            print(f"  [SKIP] {split_name}: 'text' column not in {data_path.name}")
            continue

        texts  = df["text"].astype(str)
        labels = df[lc].astype(int).values
        probs  = pipe.predict_proba(texts)[:, 1]  # P(injection)

        result = pd.DataFrame({
            "text":              df["text"].astype(str).values,
            "true_label":        labels,
            "probability_score": probs,
        })
        # Carry over useful metadata columns if present
        for extra_col in ("source", "source_type", "attack_type", "obfuscation",
                          "transform_type", "domain"):
            if extra_col in df.columns:
                result[extra_col] = df[extra_col].values

        result.to_csv(out_csv, index=False)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        print(f"  Wrote {out_csv.relative_to(ROOT)}  "
              f"(n={len(result)}, benign={n_neg}, injection={n_pos})")


# ──────────────────────────────────────────────────────────────────────────────
# Transformer models (DistilBERT, DeBERTa-v3)
# ──────────────────────────────────────────────────────────────────────────────

def _softmax2(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 2-class logits."""
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float64)


def _run_transformer_split(
    model,
    tokenizer,
    device,
    df: pd.DataFrame,
    max_len: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Run batched inference, return P(injection) for each row."""
    import torch

    all_probs: list[float] = []
    texts = df["text"].astype(str).tolist()

    # Determine which input keys the model's forward() actually accepts,
    # so we don't pass token_type_ids to DistilBERT (which doesn't use them).
    import inspect as _inspect
    model_forward_params = set(_inspect.signature(model.forward).parameters.keys())

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            # Filter to only keys the model's forward() accepts
            enc = {k: v.to(device) for k, v in enc.items()
                   if k in model_forward_params}
            logits = model(**enc).logits.cpu().numpy()
            probs  = _softmax2(logits)[:, 1]
            all_probs.extend(probs.tolist())
            if hasattr(device, "type") and device.type == "mps":
                import torch as _torch
                _torch.mps.empty_cache()

    return np.array(all_probs, dtype=np.float64)


def run_transformer(
    model_key: str,
    splits: list[str] | None = None,
    force: bool = False,
) -> None:
    """
    Load a saved fine-tuned transformer model and generate probability CSVs
    for the requested splits (defaults to structural_ood only, since other
    splits already have saved predictions from training).

    Parameters
    ----------
    model_key : "distilbert" or "deberta_v3"
    splits    : list of split names from TFIDF_SPLITS; defaults to ["structural_ood"]
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if splits is None:
        splits = ["structural_ood"]

    model_dir_map = {
        "distilbert": (ROOT / "models" / "distilbert",  256),
        "deberta_v3": (ROOT / "models" / "deberta_v3",  512),
    }
    if model_key not in model_dir_map:
        raise ValueError(f"Unknown model_key '{model_key}'. Expected: distilbert, deberta_v3")

    model_dir, max_len = model_dir_map[model_key]
    out_dir_map = {
        "distilbert": ROOT / "outputs" / "distilbert_baseline",
        "deberta_v3": ROOT / "outputs" / "deberta_v3",
    }
    out_dir = out_dir_map[model_key]

    if not model_dir.exists():
        print(f"[SKIP] {model_key}: model directory not found at {model_dir.relative_to(ROOT)}. "
              "Re-run training first.")
        return

    # Pick device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Loading {model_key} from {model_dir.relative_to(ROOT)} on {device}…")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model     = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), torch_dtype=torch.float32
    )
    model.to(device)
    model.eval()

    split_map = {s[0]: s for s in TFIDF_SPLITS}
    for split_name in splits:
        if split_name not in split_map:
            print(f"  [SKIP] Unknown split '{split_name}'")
            continue
        _, data_path, label_col = split_map[split_name]
        out_csv = out_dir / f"{split_name}_probability_scores.csv"

        if out_csv.exists() and not force:
            print(f"  [EXISTS] {out_csv.relative_to(ROOT)} — skipping")
            continue
        if not data_path.exists():
            print(f"  [SKIP] {data_path.relative_to(ROOT)} not found")
            continue

        df = pd.read_csv(data_path)
        lc = label_col or "label"
        probs  = _run_transformer_split(model, tokenizer, device, df, max_len)
        labels = df[lc].astype(int).values

        result = pd.DataFrame({
            "text":              df["text"].astype(str).values,
            "true_label":        labels,
            "probability_score": probs,
        })
        for extra_col in ("source", "source_type", "attack_type", "obfuscation",
                          "transform_type", "domain"):
            if extra_col in df.columns:
                result[extra_col] = df[extra_col].values

        result.to_csv(out_csv, index=False)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        print(f"  Wrote {out_csv.relative_to(ROOT)}  "
              f"(n={len(result)}, benign={n_neg}, injection={n_pos})")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate probability score CSVs for paper analysis")
    ap.add_argument(
        "--model",
        choices=["tfidf_logreg", "distilbert", "deberta_v3", "all"],
        default="all",
        help="Which model to process (default: all).",
    )
    ap.add_argument(
        "--transformers",
        action="store_true",
        help="Include transformer models (requires saved models under models/).",
    )
    ap.add_argument(
        "--splits",
        default="structural_ood",
        help="Comma-separated split names for transformer inference (default: structural_ood).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = ap.parse_args()

    import os
    os.chdir(ROOT)

    transformer_splits = [s.strip() for s in args.splits.split(",")]

    if args.model in ("tfidf_logreg", "all"):
        print("=== TF-IDF + Logistic Regression ===")
        run_tfidf(force=args.force)

    if args.transformers or args.model in ("distilbert", "deberta_v3"):
        for key in ("distilbert", "deberta_v3"):
            if args.model == "all" or args.model == key:
                print(f"\n=== {key} ===")
                run_transformer(key, splits=transformer_splits, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
