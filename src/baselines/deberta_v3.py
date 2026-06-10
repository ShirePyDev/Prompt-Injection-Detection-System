# src/baselines/deberta_v3.py
"""
DeBERTa-v3-base fine-tuned on PIDS-Bench v3.

Roles in the paper:
  - Strong reference model (upper bound for fine-tuned transformers)
  - Evaluated on: main test, hard_benign_test (FPR), domain_ood, structural_ood

Threshold tuning:
  - Optimal threshold found on val.csv (F1-maximising)
  - Applied once to test.csv and all eval subsets
  - Never tuned on test data

Output layout (all under outputs/deberta_v3/):
  trainer/                   HF Trainer checkpoints
  logs/                      training logs
  val_report.txt             val classification report
  test_report.txt            test classification report
  val_predictions.csv        row-level val probs  (compatible with threshold_analysis.py)
  test_predictions.csv       row-level test probs (compatible with threshold_analysis.py)
  hard_benign_fpr.json       FPR on hard_benign_test at default + tuned threshold
  domain_ood_report.json     metrics on domain_ood.csv
  structural_ood_report.json metrics on structural_ood.csv
  summary.json               compact run summary
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/pids_bench_v3")
EVAL_DIR   = DATA_DIR / "eval_subsets"
OOD_DIR    = DATA_DIR / "ood"
OUT_DIR    = Path("outputs/deberta_v3")
MODEL_DIR  = Path("models/deberta_v3")

for d in (OUT_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Model ───────────────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN    = 512


# ── Device ──────────────────────────────────────────────────────────────────
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Data loading ─────────────────────────────────────────────────────────────
def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"text", "label"}.issubset(df.columns), (
        f"Missing columns in {path}: {df.columns.tolist()}"
    )
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def load_splits() -> DatasetDict:
    tr = _read_csv(DATA_DIR / "train.csv")
    va = _read_csv(DATA_DIR / "val.csv")
    te = _read_csv(DATA_DIR / "test.csv")
    return DatasetDict({
        "train":      Dataset.from_pandas(tr, preserve_index=False),
        "validation": Dataset.from_pandas(va, preserve_index=False),
        "test":       Dataset.from_pandas(te, preserve_index=False),
    })


def tokenize_splits(ds: DatasetDict, tok: AutoTokenizer) -> DatasetDict:
    def _tok(batch):
        return tok(batch["text"], truncation=True,
                   padding="max_length", max_length=MAX_LEN)  # type: ignore

    out = {}
    for split in ds:
        remove = [c for c in ds[split].column_names if c != "label"]
        out[split] = ds[split].map(_tok, batched=True, remove_columns=remove)
    return DatasetDict(out)


# ── Metrics (training loop) ──────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


# ── Softmax helper ───────────────────────────────────────────────────────────
def _softmax(logits: np.ndarray) -> np.ndarray:
    """Float64 softmax for numerical stability (avoids NaN from bf16 logits)."""
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float64)


# ── Threshold helpers ────────────────────────────────────────────────────────
def best_threshold_on_val(probs: np.ndarray, labels: np.ndarray,
                           grid: int = 501) -> float:
    """Return F1-maximising threshold from validation set."""
    if not np.isfinite(probs).all():
        raise ValueError(
            "Validation probabilities contain NaN/Inf — model outputs are invalid. "
            "Do not use these metrics; re-train with full FP32 (see TrainingArguments)."
        )
    thresholds = np.linspace(0.0, 1.0, grid)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = float(t)
    return best_t


# ── Single-file evaluation ───────────────────────────────────────────────────
@dataclass
class EvalResult:
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float     # FP / (FP + TN) — key for over-defense analysis
    fnr: float     # FN / (FN + TP)
    threshold: float


def _eval_at_threshold(probs: np.ndarray, y_true: np.ndarray,
                        threshold: float) -> EvalResult:
    y_pred = (probs >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n  = len(y_true)

    def sdiv(a, b): return a / b if b else 0.0

    return EvalResult(
        n         = n,
        accuracy  = sdiv(tp + tn, n),
        precision = sdiv(tp, tp + fp),
        recall    = sdiv(tp, tp + fn),
        f1        = f1_score(y_true, y_pred, zero_division=0),                  #type: ignore
        fpr       = sdiv(fp, fp + tn),
        fnr       = sdiv(fn, fn + tp),
        threshold = threshold,
    )


def eval_csv(csv_path: Path, trainer: Trainer, tok: AutoTokenizer,
             threshold: float = 0.5) -> tuple[EvalResult, np.ndarray, np.ndarray]:
    """
    Load a CSV (must have text + label), run inference, return EvalResult
    plus (y_true, probs) for downstream use.
    """
    df   = _read_csv(csv_path)
    enc  = tok(df["text"].tolist(), truncation=True,                # type: ignore
               padding=True, max_length=MAX_LEN)
    ds   = Dataset.from_dict({**enc, "labels": df["label"].tolist()})
    pred = trainer.predict(ds)  # type: ignore
    probs   = _softmax(np.asarray(pred.predictions))[:, 1]
    y_true  = df["label"].to_numpy()
    result  = _eval_at_threshold(probs, y_true, threshold)
    return result, y_true, probs


# ── Training + full evaluation ───────────────────────────────────────────────
def run_train(
    num_epochs:  int   = 3,
    batch_size:  int   = 16,
    lr:          float = 2e-5,
    seed:        int   = 42,
    gradient_accumulation_steps: int = 1,
    out_dir: Path | None = None,
):
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)

    global OUT_DIR, MODEL_DIR
    if out_dir is not None:
        OUT_DIR   = Path(out_dir)
        MODEL_DIR = OUT_DIR / "model"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = pick_device()
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    print(f"Device: {device}")
    print(
        f"Train batch={batch_size}  grad_accum={gradient_accumulation_steps}  "
        f"(effective batch={batch_size * gradient_accumulation_steps})"
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # ── Load & tokenise ──────────────────────────────────────────────────────
    ds    = load_splits()
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    ds_tok = tokenize_splits(ds, tok)

    # ── Model ────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, torch_dtype=torch.float32
    )
    model.to(device)
    model.float()

    # ── TrainingArguments (handles old/new HF API) ───────────────────────────
    ta_sig = inspect.signature(TrainingArguments.__init__)
    strategy_kw = (
        {"evaluation_strategy": "epoch"}
        if "evaluation_strategy" in ta_sig.parameters
        else {"eval_strategy": "epoch"}
    )

    args = TrainingArguments(
        output_dir                  = str(OUT_DIR / "trainer"),
        save_strategy               = "epoch",
        learning_rate               = lr,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        num_train_epochs            = num_epochs,
        weight_decay                = 0.01,
        warmup_ratio                = 0.06,
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        logging_dir                 = str(OUT_DIR / "logs"),
        logging_steps               = 50,
        report_to                   = [],
        seed                        = seed,
        save_total_limit            = 2,
        max_grad_norm               = 1.0,
        fp16                        = False,
        bf16                        = False,
        gradient_accumulation_steps = gradient_accumulation_steps,
        **strategy_kw,  # type: ignore
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = ds_tok["train"],
        eval_dataset    = ds_tok["validation"],
        data_collator   = DataCollatorWithPadding(tokenizer=tok),
        compute_metrics = compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(MODEL_DIR))
    tok.save_pretrained(str(MODEL_DIR))
    print(f"Model saved → {MODEL_DIR}")

    # ── Collect val predictions (for threshold tuning) ───────────────────────
    val_pred   = trainer.predict(ds_tok["validation"])  # type: ignore
    val_logits = np.asarray(val_pred.predictions, dtype=np.float64)
    if not np.isfinite(val_logits).all():
        raise RuntimeError(
            "DeBERTa validation logits contain NaN/Inf. Training diverged — "
            "discard this run. Try batch_size=8 on GPU or check Colab torch/transformers versions."
        )
    val_probs  = _softmax(val_logits)[:, 1]
    val_true   = np.asarray(val_pred.label_ids)
    print(
        f"Val prob stats: min={val_probs.min():.4f} max={val_probs.max():.4f} "
        f"mean={val_probs.mean():.4f}"
    )

    val_df_raw = pd.read_csv(DATA_DIR / "val.csv")
    pd.DataFrame({
        "text":              val_df_raw["text"].astype(str).values,
        "true_label":        val_true,
        "probability_score": val_probs,
    }).to_csv(OUT_DIR / "val_predictions.csv", index=False)

    # ── Find optimal threshold on val ────────────────────────────────────────
    tuned_threshold = best_threshold_on_val(val_probs, val_true)
    print(f"Val-tuned threshold: {tuned_threshold:.4f}")

    # ── Evaluate on val (classification report) ──────────────────────────────
    val_preds_bin = np.argmax(val_pred.predictions, axis=-1)
    val_report    = classification_report(
        val_true, val_preds_bin,
        target_names=["benign", "injection"], digits=4
    )
    val_cm = confusion_matrix(val_true, val_preds_bin, labels=[0, 1])
    with open(OUT_DIR / "val_report.txt", "w") as f:
        f.write(val_report)             #type: ignore
        f.write("\nConfusion Matrix [[TN FP][FN TP]]:\n")
        f.write(str(val_cm))

    # ── Evaluate on test — default + tuned threshold ─────────────────────────
    test_pred   = trainer.predict(ds_tok["test"])  # type: ignore
    test_probs  = _softmax(np.asarray(test_pred.predictions))[:, 1]
    test_true   = np.asarray(test_pred.label_ids)

    test_df_raw = pd.read_csv(DATA_DIR / "test.csv")
    source_col  = test_df_raw["source"].astype(str).values if "source" in test_df_raw.columns else [""] * len(test_true)

    pd.DataFrame({
        "text":              test_df_raw["text"].astype(str).values,
        "true_label":        test_true,
        "probability_score": test_probs,
        "source":            source_col,
    }).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    test_report = classification_report(
        test_true, np.argmax(test_pred.predictions, axis=-1),
        target_names=["benign", "injection"], digits=4
    )
    test_cm = confusion_matrix(test_true, np.argmax(test_pred.predictions, axis=-1), labels=[0, 1])
    with open(OUT_DIR / "test_report.txt", "w") as f:
        f.write(test_report)                            #type: ignore
        f.write("\nConfusion Matrix [[TN FP][FN TP]]:\n")
        f.write(str(test_cm))

    res_default = _eval_at_threshold(test_probs, test_true, 0.5)
    res_tuned   = _eval_at_threshold(test_probs, test_true, tuned_threshold)

    # ── FPR on hard_benign_test ───────────────────────────────────────────────
    # hard_benign_test is all benign (label=0); FPR = fraction flagged as injection
    hb_path = EVAL_DIR / "hard_benign_test.csv"
    hb_df   = pd.read_csv(hb_path)
    hb_df["text"] = hb_df["text"].astype(str)
    hb_enc  = tok(hb_df["text"].tolist(), truncation=True,
                  padding=True, max_length=MAX_LEN)
    hb_ds   = Dataset.from_dict({**hb_enc, "labels": [0] * len(hb_df)})
    hb_pred = trainer.predict(hb_ds)  # type: ignore
    hb_probs = _softmax(np.asarray(hb_pred.predictions))[:, 1]

    hb_fpr_default = float((hb_probs >= 0.5).mean())
    hb_fpr_tuned   = float((hb_probs >= tuned_threshold).mean())

    hard_benign_result = {
        "n":             int(len(hb_df)),
        "fpr_default_0.5":     round(hb_fpr_default, 6),
        "fpr_tuned":           round(hb_fpr_tuned, 6),
        "tuned_threshold":     round(tuned_threshold, 6),
        "n_flagged_default":   int((hb_probs >= 0.5).sum()),
        "n_flagged_tuned":     int((hb_probs >= tuned_threshold).sum()),
    }
    with open(OUT_DIR / "hard_benign_fpr.json", "w") as f:
        json.dump(hard_benign_result, f, indent=2)

    # Save row-level hard_benign predictions
    pd.DataFrame({
        "text":              hb_df["text"].values,
        "true_label":        0,
        "probability_score": hb_probs,
        "flagged_default":   (hb_probs >= 0.5).astype(int),
        "flagged_tuned":     (hb_probs >= tuned_threshold).astype(int),
    }).to_csv(OUT_DIR / "hard_benign_predictions.csv", index=False)

    # ── Source-type breakdown on test ─────────────────────────────────────────
    test_raw_df = pd.read_csv(DATA_DIR / "test.csv")
    sdiv = lambda a, b: a / b if b else 0.0
    real_mask = (test_raw_df["source_type"] == "real").values if "source_type" in test_raw_df.columns else np.ones(len(test_true), dtype=bool)
    tmpl_mask = (test_raw_df["source_type"] == "template").values if "source_type" in test_raw_df.columns else np.zeros(len(test_true), dtype=bool)
    real_src_f1 = round(float(f1_score(test_true[real_mask], (test_probs[real_mask] >= tuned_threshold).astype(int), zero_division=0)), 4)
    template_f1 = round(float(f1_score(test_true[tmpl_mask], (test_probs[tmpl_mask] >= tuned_threshold).astype(int), zero_division=0)), 4)

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    try:
        roc_auc = round(float(roc_auc_score(test_true, test_probs)), 4)
    except Exception:
        roc_auc = None

    # ── Hard-benign origin breakdown ─────────────────────────────────────────
    hb_fpr_agg    = round(hb_fpr_tuned, 4)
    hb_real_mask    = (hb_df["source_type"] == "real").values if "source_type" in hb_df.columns else np.ones(len(hb_df), dtype=bool)
    hb_curated_mask = (hb_df["source_type"] == "curated").values if "source_type" in hb_df.columns else np.zeros(len(hb_df), dtype=bool)
    hb_fpr_real    = round(float((hb_probs[hb_real_mask] >= tuned_threshold).mean()), 4) if hb_real_mask.any() else 0.0
    hb_fpr_curated = round(float((hb_probs[hb_curated_mask] >= tuned_threshold).mean()), 4) if hb_curated_mask.any() else 0.0
    hard_benign_result["fpr_real_origin"]    = hb_fpr_real
    hard_benign_result["fpr_curated_origin"] = hb_fpr_curated

    # ── Domain OOD (aggregate + per-domain) ───────────────────────────────────
    dom_df = pd.read_csv(OOD_DIR / "domain_ood.csv")
    dom_result, dom_true_ood, dom_probs_ood = eval_csv(
        OOD_DIR / "domain_ood.csv", trainer, tok, threshold=tuned_threshold
    )
    domain_ood_result: dict = {
        "aggregate": {
            "f1": round(dom_result.f1, 4), "fpr": round(dom_result.fpr, 4),
            "recall": round(dom_result.recall, 4), "precision": round(dom_result.precision, 4),
        },
        "per_domain": {},
    }
    for dom_name, grp in dom_df.groupby("domain"):
        idx = grp.index.tolist()
        g_probs = dom_probs_ood[idx]; g_true = dom_true_ood[idx]
        benign_mask = g_true == 0; attack_mask = g_true == 1
        d_fpr    = round(float((g_probs[benign_mask] >= tuned_threshold).mean()), 4) if benign_mask.any() else 0.0
        d_recall = round(float((g_probs[attack_mask] >= tuned_threshold).mean()), 4) if attack_mask.any() else 0.0
        domain_ood_result["per_domain"][str(dom_name)] = {"fpr": d_fpr, "recall": d_recall}
    with open(OUT_DIR / "domain_ood_report.json", "w") as f:
        json.dump(domain_ood_result, f, indent=2)
    print(f"domain_ood: F1={dom_result.f1:.4f}  FPR={dom_result.fpr:.4f}")

    # ── Structural OOD (aggregate + per-transform) ────────────────────────────
    struct_df = pd.read_csv(OOD_DIR / "structural_ood.csv")
    struct_result, struct_true_ood, struct_probs_ood = eval_csv(
        OOD_DIR / "structural_ood.csv", trainer, tok, threshold=tuned_threshold
    )
    struct_ood_result: dict = {
        "aggregate": {
            "f1": round(struct_result.f1, 4), "fpr": round(struct_result.fpr, 4),
            "recall": round(struct_result.recall, 4), "precision": round(struct_result.precision, 4),
        },
        "per_transform": {},
    }
    for tr_name, grp in struct_df.groupby("transform_type"):
        idx = grp.index.tolist()
        g_probs = struct_probs_ood[idx]; g_true = struct_true_ood[idx]
        benign_mask = g_true == 0; attack_mask = g_true == 1
        t_fpr    = round(float((g_probs[benign_mask] >= tuned_threshold).mean()), 4) if benign_mask.any() else 0.0
        t_recall = round(float((g_probs[attack_mask] >= tuned_threshold).mean()), 4) if attack_mask.any() else 0.0
        struct_ood_result["per_transform"][str(tr_name)] = {"fpr": t_fpr, "recall": t_recall}
    with open(OUT_DIR / "structural_ood_report.json", "w") as f:
        json.dump(struct_ood_result, f, indent=2)
    print(f"structural_ood: F1={struct_result.f1:.4f}  FPR={struct_result.fpr:.4f}")

    # ── Obfuscated attacks ────────────────────────────────────────────────────
    obf_df = pd.read_csv(EVAL_DIR / "obfuscated_attacks.csv").reset_index(drop=True)
    obf_df["text"] = obf_df["text"].astype(str)
    obf_enc  = tok(obf_df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    obf_ds   = Dataset.from_dict({**obf_enc, "labels": [1] * len(obf_df)})
    obf_pred = trainer.predict(obf_ds)  # type: ignore
    obf_probs = _softmax(np.asarray(obf_pred.predictions))[:, 1]
    obf_preds = (obf_probs >= tuned_threshold).astype(int)
    obf_n = len(obf_df); obf_n_detected = int(obf_preds.sum())
    per_obf: dict = {}
    for obf_type, group in obf_df.groupby("obfuscation"):
        idx = group.index.tolist(); grp_n = len(idx); grp_det = int(obf_preds[idx].sum())
        per_obf[str(obf_type)] = {"n": grp_n, "n_detected": grp_det,
                                   "recall": round(grp_det / grp_n, 6) if grp_n else 0.0}
    obfuscated_attacks_result = {
        "n": obf_n, "n_detected": obf_n_detected,
        "overall_recall": round(obf_n_detected / obf_n, 6) if obf_n else 0.0,
        "overall_fnr":    round(1 - obf_n_detected / obf_n, 6) if obf_n else 1.0,
        "threshold_used": round(tuned_threshold, 6), "per_obfuscation": per_obf,
    }
    with open(OUT_DIR / "obfuscated_attacks_report.json", "w") as f:
        json.dump(obfuscated_attacks_result, f, indent=2)
    print(f"obfuscated_attacks: recall={obfuscated_attacks_result['overall_recall']:.4f}")

    # ── Balanced subtype test ─────────────────────────────────────────────────
    sub_result, sub_true, sub_probs = eval_csv(
        EVAL_DIR / "balanced_subtype_test.csv", trainer, tok, threshold=tuned_threshold
    )
    sub_preds = (sub_probs >= tuned_threshold).astype(int)
    sub_df    = pd.read_csv(EVAL_DIR / "balanced_subtype_test.csv").reset_index(drop=True)
    per_subtype: dict = {}
    if "attack_type" in sub_df.columns:
        for atype, group in sub_df.groupby("attack_type"):
            idx = group.index.tolist(); g_true = sub_true[idx]; g_pred = sub_preds[idx]
            g_tp = int(((g_true==1)&(g_pred==1)).sum()); g_tn = int(((g_true==0)&(g_pred==0)).sum())
            g_fp = int(((g_true==0)&(g_pred==1)).sum()); g_fn = int(((g_true==1)&(g_pred==0)).sum())
            per_subtype[str(atype)] = {
                "n": len(g_true), "f1": round(sdiv(2*g_tp, 2*g_tp+g_fp+g_fn), 4),
                "recall": round(sdiv(g_tp, g_tp+g_fn), 4), "fpr": round(sdiv(g_fp, g_fp+g_tn), 4),
            }
    balanced_subtype_result = {
        "n": sub_result.n, "f1": round(sub_result.f1, 4),
        "precision": round(sub_result.precision, 4), "recall": round(sub_result.recall, 4),
        "fpr": round(sub_result.fpr, 4), "threshold_used": round(tuned_threshold, 6),
        "per_subtype": per_subtype,
    }
    with open(OUT_DIR / "balanced_subtype_report.json", "w") as f:
        json.dump(balanced_subtype_result, f, indent=2)
    print(f"balanced_subtype: F1={sub_result.f1:.4f}  FPR={sub_result.fpr:.4f}")

    # # ── Threshold sweep for Figure 1 ─────────────────────────────────────────
    # sweep_taus = np.arange(0.0, 1.001, 0.002)
    # sweep_rows = []
    # for tau in sweep_taus:
    #     sw_f1  = round(float(f1_score(test_true, (test_probs >= tau).astype(int), zero_division=0)), 6)
    #     sw_fpr = round(float((hb_probs >= tau).mean()), 6)
    #     sweep_rows.append({"tau": round(float(tau), 4), "test_F1": sw_f1, "hb_FPR_agg": sw_fpr})
    # pd.DataFrame(sweep_rows).to_csv(OUT_DIR / "threshold_sweep.csv", index=False)

    # ── Threshold sweep for Figure 1 ─────────────────────────────────────────
    sweep_taus = np.arange(0.0, 1.001, 0.002)
    sweep_rows = []
    for tau in sweep_taus:
        sw_f1   = round(float(f1_score(test_true, (test_probs >= tau).astype(int), zero_division=0)), 6)
        sw_fpr  = round(float((hb_probs >= tau).mean()), 6)
        sw_real = round(float((hb_probs[hb_real_mask] >= tau).mean()), 6) if hb_real_mask.any() else 0.0
        sw_cur  = round(float((hb_probs[hb_curated_mask] >= tau).mean()), 6) if hb_curated_mask.any() else 0.0
        sweep_rows.append({
            "tau": round(float(tau), 4),
            "test_F1": sw_f1,
            "hb_FPR_agg": sw_fpr,
            "hb_FPR_real": sw_real,
            "hb_FPR_curated": sw_cur,
        })
    pd.DataFrame(sweep_rows).to_csv(OUT_DIR / "threshold_sweep.csv", index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    per_dom = domain_ood_result["per_domain"]
    per_tr  = struct_ood_result["per_transform"]
    summary = {
        "model":          MODEL_NAME,
        "model_saved_to": str(MODEL_DIR),
        "device":         str(device),
        "hyperparams": {
            "epochs": num_epochs, "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lr": lr, "seed": seed, "max_len": MAX_LEN,
        },
        "val_tuned_threshold": round(tuned_threshold, 6),
        "test_results": {
            "default_threshold_0.5": {
                "f1": round(res_default.f1, 4), "precision": round(res_default.precision, 4),
                "recall": round(res_default.recall, 4), "fpr": round(res_default.fpr, 4),
            },
            "val_tuned_threshold": {
                "threshold": round(res_tuned.threshold, 6), "f1": round(res_tuned.f1, 4),
                "precision": round(res_tuned.precision, 4), "recall": round(res_tuned.recall, 4),
                "fpr": round(res_tuned.fpr, 4),
            },
        },
        "real_src_f1": real_src_f1, "template_f1": template_f1, "roc_auc": roc_auc,
        "hard_benign_test_fpr": hard_benign_result,
        "obfuscated_attacks": {
            "overall_recall": obfuscated_attacks_result["overall_recall"],
            "overall_fnr":    obfuscated_attacks_result["overall_fnr"],
        },
        "balanced_subtype": {"f1": balanced_subtype_result["f1"],
                              "recall": balanced_subtype_result["recall"],
                              "fpr": balanced_subtype_result["fpr"]},
        "domain_ood":    domain_ood_result,
        "structural_ood": struct_ood_result,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DeBERTa-v3 training complete ===")
    print(f"Test F1  (tuned {tuned_threshold:.3f}):   {res_tuned.f1:.4f}")
    print(f"Hard-benign FPR (tuned):   {hb_fpr_agg:.4f}")
    print(f"Obfuscated recall:         {obfuscated_attacks_result['overall_recall']:.4f}")
    print(f"Summary → {OUT_DIR / 'summary.json'}")

    return {
        "IID_F1":           round(res_tuned.f1,        4),
        "real_src_F1":      real_src_f1,
        "template_F1":      template_f1,
        "IID_recall":       round(res_tuned.recall,    4),
        "IID_precision":    round(res_tuned.precision, 4),
        "test_FPR":         round(res_tuned.fpr,       4),
        "ROC_AUC":          roc_auc,
        "tau":              round(tuned_threshold,      6),
        "hb_FPR_agg":       hb_fpr_agg,
        "hb_FPR_real":      hb_fpr_real,
        "hb_FPR_curated":   hb_fpr_curated,
        "obf_recall":       obfuscated_attacks_result["overall_recall"],
        "domain_OOD_F1_agg": round(dom_result.f1, 4),
        "medical_FPR":      per_dom.get("medical", {}).get("fpr", None),
        "medical_recall":   per_dom.get("medical", {}).get("recall", None),
        "legal_FPR":        per_dom.get("legal",   {}).get("fpr", None),
        "legal_recall":     per_dom.get("legal",   {}).get("recall", None),
        "finance_FPR":      per_dom.get("finance", {}).get("fpr", None),
        "finance_recall":   per_dom.get("finance", {}).get("recall", None),
        "code_FPR":         per_dom.get("code",    {}).get("fpr", None),
        "code_recall":      per_dom.get("code",    {}).get("recall", None),
        "struct_OOD_F1_agg":  round(struct_result.f1,  4),
        "struct_OOD_FPR_agg": round(struct_result.fpr, 4),
        "json_wrap_FPR":         per_tr.get("json_wrap",                {}).get("fpr", None),
        "json_wrap_recall":      per_tr.get("json_wrap",                {}).get("recall", None),
        "prompt_dilute_FPR":     per_tr.get("prompt_dilute",            {}).get("fpr", None),
        "prompt_dilute_recall":  per_tr.get("prompt_dilute",            {}).get("recall", None),
        "prefix_mimic_FPR":      per_tr.get("instruction_prefix_mimic", {}).get("fpr", None),
        "prefix_mimic_recall":   per_tr.get("instruction_prefix_mimic", {}).get("recall", None),
    }


def run():
    run_train(num_epochs=3, batch_size=16, lr=2e-5, seed=42)


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--seed", type=int, default=42)
    _a = _p.parse_args()
    run_train(num_epochs=3, batch_size=16, lr=2e-5, seed=_a.seed)
