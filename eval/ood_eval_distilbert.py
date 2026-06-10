#!/usr/bin/env python3
"""
OOD Generalization Evaluation (NO retraining)
- Loads your trained DistilBERT checkpoint from models/distilbert
- Evaluates on an external labeled prompt-injection dataset
- Uses Trainer.predict(...) (same style as your current eval path)
- Writes predictions + metrics report

Default external dataset:
  qualifire/prompt-injections-benchmark  (label: jailbreak/benign)
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# -----------------------------
# Metrics helpers (no sklearn)
# -----------------------------
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true, y_pred: {0,1}
    Positive class = 1 (jailbreak)
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)

    # Reviewer-friendly:
    fpr = _safe_div(fp, fp + tn)  # false positive rate on benign
    fnr = _safe_div(fn, fn + tp)

    return {
        "n": int(len(y_true)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr_benign": fpr,
        "fnr_jailbreak": fnr,
    }

def threshold_sweep(probs_pos: np.ndarray, y_true: np.ndarray, thresholds=None) -> dict:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    best = None
    rows = []
    for t in thresholds:
        y_pred = (probs_pos >= t).astype(int)
        m = compute_binary_metrics(y_true, y_pred)
        m["threshold"] = float(t)
        rows.append(m)
        if best is None or m["f1"] > best["f1"]:
            best = m

    return {
        "best_by_f1": best,
        "curve": rows,  # can be large; you can disable saving if you want
    }

# -----------------------------
# Dataset loading (Qualifire)
# -----------------------------
def load_qualifire_dataset() -> tuple[list[str], np.ndarray]:
    """
    Returns (texts, labels)
    labels: jailbreak=1, benign=0
    Dataset page: qualifire/prompt-injections-benchmark
    """
    ds = load_dataset("qualifire/prompt-injections-benchmark")

    # Many HF datasets use 'train' only
    split = "train" if "train" in ds else list(ds.keys())[0]
    d = ds[split]

    # expected columns: text, label (label is string: 'jailbreak'/'benign')
    if "text" not in d.column_names or "label" not in d.column_names:
        raise ValueError(f"Unexpected columns: {d.column_names}. Need at least ['text','label'].")

    texts = [str(x) for x in d["text"]]
    raw_labels = d["label"]

    # Map labels to {0,1}
    # Accept either string labels or int labels depending on dataset version.
    mapped = []
    for lab in raw_labels:
        if isinstance(lab, str):
            lab_l = lab.strip().lower()
            if lab_l == "jailbreak":
                mapped.append(1)
            elif lab_l == "benign":
                mapped.append(0)
            else:
                raise ValueError(f"Unknown label string: {lab}")
        else:
            # If it's already numeric, try common conventions:
            # We'll assume 1=jailbreak, 0=benign. If wrong, you'll see weird metrics.
            mapped.append(int(lab))

    return texts, np.array(mapped, dtype=int)

# -----------------------------
# Main OOD eval
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/distilbert", help="Trained checkpoint dir")
    ap.add_argument("--outdir", default="outputs/ood", help="Where to write outputs")
    ap.add_argument("--max_len", type=int, default=256, help="Tokenizer max_length")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on P(jailbreak)")
    ap.add_argument("--sweep", action="store_true", help="Run threshold sweep and save curve")
    ap.add_argument("--dataset", default="qualifire", choices=["qualifire"], help="Which OOD dataset loader to use")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load external dataset
    if args.dataset == "qualifire":
        texts, y_true = load_qualifire_dataset()
        dataset_name = "qualifire_prompt_injections_benchmark"
    else:
        raise ValueError("Unsupported dataset option")

    # 2) Load model/tokenizer from your trained checkpoint (NO retraining)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)

    # 3) Tokenize
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=args.max_len,
    )

    # Create a minimal HF Dataset-like object for Trainer.predict
    from datasets import Dataset
    ds_tok = Dataset.from_dict({**enc, "labels": y_true.tolist()})

    # 4) Build a Trainer for inference-only
    # Note: TrainingArguments still required by Trainer
    tmp_out = os.path.join(args.outdir, "_tmp_trainer")
    targs = TrainingArguments(
        output_dir=tmp_out,
        per_device_eval_batch_size=32,
        dataloader_drop_last=False,
        report_to=[],  # no wandb
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
    )

    # 5) Predict
    pred = trainer.predict(ds_tok) 
    logits = pred.predictions
    # Handle shape: (N,2) for binary classifier
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(f"Unexpected logits shape: {logits.shape}. Expected (N,2).")   

    # Softmax for P(jailbreak)=class1
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    p_jb = probs[:, 1]

    y_pred = (p_jb >= args.threshold).astype(int)

    # 6) Metrics
    metrics = compute_binary_metrics(y_true, y_pred)
    metrics.update({
        "dataset": dataset_name,
        "ckpt": args.ckpt,
        "max_len": args.max_len,
        "threshold": float(args.threshold),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    sweep_obj = None
    if args.sweep:
        sweep_obj = threshold_sweep(p_jb, y_true)

    # 7) Save outputs
    preds_path = os.path.join(args.outdir, f"{dataset_name}_preds.csv")
    rep_path = os.path.join(args.outdir, f"{dataset_name}_report.json")
    sweep_path = os.path.join(args.outdir, f"{dataset_name}_threshold_sweep.json")

    df = pd.DataFrame({
        "text": texts,
        "y_true": y_true,
        "p_jailbreak": p_jb,
        "y_pred": y_pred,
    })
    df.to_csv(preds_path, index=False)

    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if sweep_obj is not None:
        with open(sweep_path, "w", encoding="utf-8") as f:
            json.dump(sweep_obj, f, indent=2)

    print("=== OOD EVAL DONE ===")
    print(f"Dataset: {dataset_name}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Saved predictions: {preds_path}")
    print(f"Saved report: {rep_path}")
    if sweep_obj is not None:
        print(f"Saved threshold sweep: {sweep_path}")

    print("\n--- Metrics ---")
    for k in ["n", "accuracy", "precision", "recall", "f1", "fpr_benign", "fnr_jailbreak", "tp", "tn", "fp", "fn"]:
        print(f"{k:>12}: {metrics[k]}")

if __name__ == "__main__":
    main()