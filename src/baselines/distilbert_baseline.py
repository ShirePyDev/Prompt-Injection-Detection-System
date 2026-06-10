# src/baselines/distilbert_baseline.py
from __future__ import annotations
from pathlib import Path
import os, json, inspect

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, set_seed,
)
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score,
    classification_report, confusion_matrix, f1_score, roc_auc_score,
)

DATA_DIR  = Path("data/pids_bench_v3")
EVAL_DIR  = DATA_DIR / "eval_subsets"
OOD_DIR   = DATA_DIR / "ood"
OUT_DIR   = Path("outputs/distilbert_baseline")
MODEL_DIR = Path("models/distilbert")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 256


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_splits(hard_neg: bool = False) -> DatasetDict:
    tr = pd.read_csv(DATA_DIR / "train.csv")
    va = pd.read_csv(DATA_DIR / "val.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    if hard_neg:
        hn_train_path = DATA_DIR / "hard_negative_train.csv"
        hn_val_path   = DATA_DIR / "hard_negative_val.csv"
        if not hn_train_path.exists() or not hn_val_path.exists():
            raise FileNotFoundError(
                "Hard-negative CSVs not found. Run:\n"
                "  python -m data_builder.build_hard_negative"
            )
        hn_tr = pd.read_csv(hn_train_path)[["text", "label"]]
        hn_va = pd.read_csv(hn_val_path)[["text", "label"]]
        tr = pd.concat([tr, hn_tr], ignore_index=True)
        va = pd.concat([va, hn_va], ignore_index=True)
        print(
            f"[hard-neg] appended {len(hn_tr)} train rows and "
            f"{len(hn_va)} val rows"
        )
    for df in (tr, va, te):
        assert {"text", "label"}.issubset(df.columns)
        df["text"]  = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
    return DatasetDict({
        "train":      Dataset.from_pandas(tr, preserve_index=False),
        "validation": Dataset.from_pandas(va, preserve_index=False),
        "test":       Dataset.from_pandas(te, preserve_index=False),
    })


def _tokenize_splits(ds: DatasetDict, tok: AutoTokenizer) -> DatasetDict:
    def _tok(batch):
        return tok(batch["text"], truncation=True,
                   padding="max_length", max_length=MAX_LEN)  # type: ignore
    out = {}
    for split in ds.keys():
        cols_to_remove = [c for c in ds[split].column_names if c != "label"]
        out[split] = ds[split].map(_tok, batched=True, remove_columns=cols_to_remove)
    return DatasetDict(out)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp     = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _best_threshold_on_val(probs: np.ndarray, labels: np.ndarray,
                            grid: int = 501) -> float:
    """
    F1-maximising threshold found on val data ONLY.
    Never call this with test or OOD labels — that would be data leakage.
    """
    thresholds      = np.linspace(0.0, 1.0, grid)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = float(t)
    return best_t


def _metrics_at_threshold(probs: np.ndarray, y_true: np.ndarray,
                           threshold: float) -> dict:
    y_pred = (probs >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n  = len(y_true)
    sdiv = lambda a, b: a / b if b else 0.0
    return {
        "n":         n,
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(sdiv(tp, tp + fp), 4),
        "recall":    round(sdiv(tp, tp + fn), 4),
        "fpr":       round(sdiv(fp, fp + tn), 4),
        "fnr":       round(sdiv(fn, fn + tp), 4),
        "threshold": round(threshold, 6),
    }


def _save_report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    report  = classification_report(
        y_true, y_pred, target_names=["benign", "injection"], digits=4
    )
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out_txt = OUT_DIR / f"{name}_report.txt"
    with open(out_txt, "w") as f:
        f.write(report)  # type: ignore
        f.write("\nConfusion Matrix [[TN, FP],[FN, TP]]:\n")
        f.write(str(cm))
    return out_txt


def _infer_csv(csv_path: Path, trainer: Trainer, tok: AutoTokenizer,
               threshold: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Tokenise + run inference on any labelled CSV.
    Threshold passed in from val tuning — never derived from this data.
    Returns (y_true, probs, metrics_dict).
    """
    df = pd.read_csv(csv_path)
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    enc    = tok(df["text"].tolist(), truncation=True,
                 padding=True, max_length=MAX_LEN)
    ds     = Dataset.from_dict({**enc, "labels": df["label"].tolist()})
    pred   = trainer.predict(ds)  # type: ignore
    probs  = _softmax(np.asarray(pred.predictions))[:, 1]
    y_true = df["label"].to_numpy()
    return y_true, probs, _metrics_at_threshold(probs, y_true, threshold)


def _eval_obfuscated_attacks(csv_path: Path, trainer: Trainer,
                              tok: AutoTokenizer, threshold: float) -> dict:
    """
    obfuscated_attacks.csv is 100% injection (label=1).
    Metric: recall per obfuscation type.
    Threshold passed in from val — never tuned on this data.
    """
    df = pd.read_csv(csv_path).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    enc   = tok(df["text"].tolist(), truncation=True,
                padding=True, max_length=MAX_LEN)
    ds    = Dataset.from_dict({**enc, "labels": [1] * len(df)})
    pred  = trainer.predict(ds)  # type: ignore
    probs = _softmax(np.asarray(pred.predictions))[:, 1]
    preds = (probs >= threshold).astype(int)

    n          = len(df)
    n_detected = int(preds.sum())

    per_obf: dict = {}
    for obf, group in df.groupby("obfuscation"):
        idx       = group.index.tolist()
        grp_preds = preds[idx]
        grp_n     = len(idx)
        grp_det   = int(grp_preds.sum())
        per_obf[str(obf)] = {
            "n":          grp_n,
            "n_detected": grp_det,
            "recall":     round(grp_det / grp_n, 6) if grp_n else 0.0,
        }

    result = {
        "n":               n,
        "n_detected":      n_detected,
        "overall_recall":  round(n_detected / n, 6) if n else 0.0,
        "overall_fnr":     round(1 - n_detected / n, 6) if n else 1.0,
        "threshold_used":  round(threshold, 6),
        "per_obfuscation": per_obf,
    }
    with open(OUT_DIR / "obfuscated_attacks_report.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def _eval_balanced_subtype(csv_path: Path, trainer: Trainer,
                            tok: AutoTokenizer, threshold: float) -> dict:
    """
    balanced_subtype_test.csv has benign + injection with attack_type labels.
    Reports overall metrics + per-subtype breakdown.
    Threshold passed in from val — never tuned on this data.
    """
    df     = pd.read_csv(csv_path).reset_index(drop=True)
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    y_true, probs, overall = _infer_csv(csv_path, trainer, tok, threshold)
    preds  = (probs >= threshold).astype(int)

    _save_report("balanced_subtype", y_true, preds)

    sdiv = lambda a, b: a / b if b else 0.0
    per_subtype: dict = {}
    if "attack_type" in df.columns:
        for atype, group in df.groupby("attack_type"):
            idx    = group.index.tolist()
            g_true = y_true[idx]
            g_pred = preds[idx]
            g_tp   = int(((g_true == 1) & (g_pred == 1)).sum())
            g_tn   = int(((g_true == 0) & (g_pred == 0)).sum())
            g_fp   = int(((g_true == 0) & (g_pred == 1)).sum())
            g_fn   = int(((g_true == 1) & (g_pred == 0)).sum())
            per_subtype[str(atype)] = {
                "n":      len(g_true),
                "f1":     round(sdiv(2 * g_tp, 2 * g_tp + g_fp + g_fn), 4),
                "recall": round(sdiv(g_tp, g_tp + g_fn), 4),
                "fpr":    round(sdiv(g_fp, g_fp + g_tn), 4),
            }

    result = {**overall, "per_subtype": per_subtype}
    with open(OUT_DIR / "balanced_subtype_report.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_train(num_epochs: int = 2, batch_size: int = 16,
              lr: float = 5e-5, seed: int = 42,
              hard_neg: bool = False,
              out_dir: Path | None = None) -> dict:
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)

    global OUT_DIR, MODEL_DIR
    if out_dir is not None:
        OUT_DIR   = Path(out_dir)
        MODEL_DIR = OUT_DIR / "model"
    elif hard_neg:
        OUT_DIR   = Path("outputs/distilbert_hardneg")
        MODEL_DIR = Path("models/distilbert_hardneg")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = pick_device()
    if device.type == "mps":
        import torch as _t
        _t.mps.manual_seed(seed)

    # ── Load & tokenise ───────────────────────────────────────────────────────
    ds     = _load_splits(hard_neg=hard_neg)
    tok    = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    ds_tok = _tokenize_splits(ds, tok)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    ta_sig      = inspect.signature(TrainingArguments.__init__)
    strategy_kw = (
        {"evaluation_strategy": "epoch"}
        if "evaluation_strategy" in ta_sig.parameters
        else {"eval_strategy": "epoch"}
    )
    args = TrainingArguments(
        output_dir=str(OUT_DIR / "trainer"),
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,   # picks best checkpoint via val loss — not leakage
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(OUT_DIR / "logs"),
        logging_steps=50,
        report_to=[],
        seed=seed,
        save_total_limit=2,
        max_grad_norm=1.0,
        **strategy_kw,  # type: ignore
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(MODEL_DIR))
    tok.save_pretrained(str(MODEL_DIR))

    # ── Val predictions → threshold tuning ───────────────────────────────────
    # Threshold derived from val ONLY. Test and eval sets never seen here.
    val_pred   = trainer.predict(ds_tok["validation"])  # type: ignore
    val_logits = np.asarray(val_pred.predictions)
    val_probs  = _softmax(val_logits)[:, 1]
    val_true   = np.asarray(val_pred.label_ids)

    tuned_threshold = _best_threshold_on_val(val_probs, val_true)
    print(f"Val-tuned threshold: {tuned_threshold:.4f}")

    pd.DataFrame({
        "true_label":        val_true,
        "probability_score": val_probs,
    }).to_csv(OUT_DIR / "val_predictions.csv", index=False)
    _save_report("val", val_true, np.argmax(val_logits, axis=-1))

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_pred   = trainer.predict(ds_tok["test"])  # type: ignore
    test_logits = np.asarray(test_pred.predictions)
    test_probs  = _softmax(test_logits)[:, 1]
    test_true   = np.asarray(test_pred.label_ids)

    test_res_default = _metrics_at_threshold(test_probs, test_true, 0.5)
    test_res_tuned   = _metrics_at_threshold(test_probs, test_true, tuned_threshold)
    _save_report("test", test_true, (test_probs >= tuned_threshold).astype(int))

    test_raw   = pd.read_csv(DATA_DIR / "test.csv")
    source_col = (test_raw["source"].astype(str).values
                  if "source" in test_raw.columns else [""] * len(test_true))
    test_preds_df = pd.DataFrame({
        "text":              test_raw["text"].astype(str).values,
        "true_label":        test_true,
        "predicted_label":   (test_probs >= tuned_threshold).astype(int),
        "probability_score": test_probs,
        "source":            source_col,
    })
    test_preds_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    test_preds_df[
        (test_preds_df["true_label"] == 0) & (test_preds_df["predicted_label"] == 1)
    ].to_csv(OUT_DIR / "false_positives.csv", index=False)
    test_preds_df[
        (test_preds_df["true_label"] == 1) & (test_preds_df["predicted_label"] == 0)
    ].to_csv(OUT_DIR / "false_negatives.csv", index=False)

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

    # ── hard_benign FPR (aggregate + by origin) ──────────────────────────────
    hb_df   = pd.read_csv(EVAL_DIR / "hard_benign_test.csv")
    hb_enc  = tok(hb_df["text"].astype(str).tolist(), truncation=True,
                  padding=True, max_length=MAX_LEN)
    hb_ds   = Dataset.from_dict({**hb_enc, "labels": [0] * len(hb_df)})
    hb_pred = trainer.predict(hb_ds)  # type: ignore
    hb_probs = _softmax(np.asarray(hb_pred.predictions))[:, 1]

    hb_fpr_agg = round(float((hb_probs >= tuned_threshold).mean()), 4)
    hb_real_mask    = (hb_df["source_type"] == "real").values if "source_type" in hb_df.columns else np.ones(len(hb_df), dtype=bool)
    hb_curated_mask = (hb_df["source_type"] == "curated").values if "source_type" in hb_df.columns else np.zeros(len(hb_df), dtype=bool)
    hb_fpr_real    = round(float((hb_probs[hb_real_mask] >= tuned_threshold).mean()), 4) if hb_real_mask.any() else 0.0
    hb_fpr_curated = round(float((hb_probs[hb_curated_mask] >= tuned_threshold).mean()), 4) if hb_curated_mask.any() else 0.0

    hard_benign_result = {
        "n":                 int(len(hb_df)),
        "fpr_default_0.5":   round(float((hb_probs >= 0.5).mean()),            6),
        "fpr_tuned":         hb_fpr_agg,
        "fpr_real_origin":   hb_fpr_real,
        "fpr_curated_origin": hb_fpr_curated,
        "tuned_threshold":   round(tuned_threshold, 6),
        "n_flagged_default": int((hb_probs >= 0.5).sum()),
        "n_flagged_tuned":   int((hb_probs >= tuned_threshold).sum()),
    }
    with open(OUT_DIR / "hard_benign_fpr.json", "w") as f:
        json.dump(hard_benign_result, f, indent=2)
    pd.DataFrame({
        "text":              hb_df["text"].values,
        "true_label":        0,
        "probability_score": hb_probs,
        "source_type":       hb_df["source_type"].values if "source_type" in hb_df.columns else "",
        "flagged_default":   (hb_probs >= 0.5).astype(int),
        "flagged_tuned":     (hb_probs >= tuned_threshold).astype(int),
    }).to_csv(OUT_DIR / "hard_benign_predictions.csv", index=False)

    # ── Obfuscated attacks ────────────────────────────────────────────────────
    obf_res = _eval_obfuscated_attacks(
        EVAL_DIR / "obfuscated_attacks.csv", trainer, tok, tuned_threshold
    )

    # ── Balanced subtype test ─────────────────────────────────────────────────
    sub_res = _eval_balanced_subtype(
        EVAL_DIR / "balanced_subtype_test.csv", trainer, tok, tuned_threshold
    )

    # ── Domain OOD (aggregate + per-domain) ───────────────────────────────────
    dom_df = pd.read_csv(OOD_DIR / "domain_ood.csv")
    dom_true_ood, dom_probs_ood, dom_agg = _infer_csv(
        OOD_DIR / "domain_ood.csv", trainer, tok, threshold=tuned_threshold
    )
    domain_ood_result: dict = {"aggregate": dom_agg, "per_domain": {}}
    for dom_name, grp in dom_df.groupby("domain"):
        idx = grp.index.tolist()
        g_probs = dom_probs_ood[idx]; g_true = dom_true_ood[idx]
        benign_mask = g_true == 0; attack_mask = g_true == 1
        d_fpr    = round(float((g_probs[benign_mask] >= tuned_threshold).mean()), 4) if benign_mask.any() else 0.0
        d_recall = round(float((g_probs[attack_mask] >= tuned_threshold).mean()), 4) if attack_mask.any() else 0.0
        domain_ood_result["per_domain"][str(dom_name)] = {"fpr": d_fpr, "recall": d_recall}
    with open(OUT_DIR / "domain_ood_report.json", "w") as f:
        json.dump(domain_ood_result, f, indent=2)
    print(f"  domain_ood: F1={dom_agg['f1']:.4f}  FPR={dom_agg['fpr']:.4f}")

    # ── Structural OOD (aggregate + per-transform) ────────────────────────────
    struct_df = pd.read_csv(OOD_DIR / "structural_ood.csv")
    struct_true_ood, struct_probs_ood, struct_agg = _infer_csv(
        OOD_DIR / "structural_ood.csv", trainer, tok, threshold=tuned_threshold
    )
    struct_ood_result: dict = {"aggregate": struct_agg, "per_transform": {}}
    for tr_name, grp in struct_df.groupby("transform_type"):
        idx = grp.index.tolist()
        g_probs = struct_probs_ood[idx]; g_true = struct_true_ood[idx]
        benign_mask = g_true == 0; attack_mask = g_true == 1
        t_fpr    = round(float((g_probs[benign_mask] >= tuned_threshold).mean()), 4) if benign_mask.any() else 0.0
        t_recall = round(float((g_probs[attack_mask] >= tuned_threshold).mean()), 4) if attack_mask.any() else 0.0
        struct_ood_result["per_transform"][str(tr_name)] = {"fpr": t_fpr, "recall": t_recall}
    with open(OUT_DIR / "structural_ood_report.json", "w") as f:
        json.dump(struct_ood_result, f, indent=2)
    print(f"  structural_ood: F1={struct_agg['f1']:.4f}  FPR={struct_agg['fpr']:.4f}")

    # ── Threshold sweep for Figure 1 ─────────────────────────────────────────
    sweep_taus = np.arange(0.0, 1.001, 0.002)
    sweep_rows = []
    for tau in sweep_taus:
        sw_f1  = round(float(f1_score(test_true, (test_probs >= tau).astype(int), zero_division=0)), 6)
        sw_fpr = round(float((hb_probs >= tau).mean()), 6)
        sweep_rows.append({"tau": round(float(tau), 4), "test_F1": sw_f1, "hb_FPR_agg": sw_fpr})
    pd.DataFrame(sweep_rows).to_csv(OUT_DIR / "threshold_sweep.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    per_dom = domain_ood_result["per_domain"]
    per_tr  = struct_ood_result["per_transform"]
    summary = {
        "model":          MODEL_NAME,
        "model_saved_to": str(MODEL_DIR),
        "device":         str(device),
        "hyperparams": {
            "epochs": num_epochs, "batch_size": batch_size,
            "lr": lr, "seed": seed, "max_len": MAX_LEN,
        },
        "val_tuned_threshold": round(tuned_threshold, 6),
        "test_results": {
            "default_threshold_0.5": test_res_default,
            "val_tuned_threshold":   test_res_tuned,
        },
        "real_src_f1":   real_src_f1,
        "template_f1":   template_f1,
        "roc_auc":       roc_auc,
        "hard_benign_test_fpr": hard_benign_result,
        "obfuscated_attacks": {
            "overall_recall": obf_res["overall_recall"],
            "overall_fnr":    obf_res["overall_fnr"],
        },
        "balanced_subtype": {"f1": sub_res["f1"], "recall": sub_res["recall"], "fpr": sub_res["fpr"]},
        "domain_ood":    domain_ood_result,
        "structural_ood": struct_ood_result,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("DistilBERT training complete.")
    print(f"  test F1 (tuned {tuned_threshold:.3f}):    {test_res_tuned['f1']:.4f}")
    print(f"  hard_benign FPR (tuned):   {hb_fpr_agg:.4f}")
    print(f"  obfuscated recall:         {obf_res['overall_recall']:.4f}")
    print(f"  summary → {OUT_DIR / 'summary.json'}")

    return {
        "IID_F1":           test_res_tuned["f1"],
        "real_src_F1":      real_src_f1,
        "template_F1":      template_f1,
        "IID_recall":       test_res_tuned["recall"],
        "IID_precision":    test_res_tuned["precision"],
        "test_FPR":         test_res_tuned["fpr"],
        "ROC_AUC":          roc_auc,
        "tau":              round(tuned_threshold, 6),
        "hb_FPR_agg":       hb_fpr_agg,
        "hb_FPR_real":      hb_fpr_real,
        "hb_FPR_curated":   hb_fpr_curated,
        "obf_recall":       obf_res["overall_recall"],
        "domain_OOD_F1_agg": dom_agg["f1"],
        "medical_FPR":      per_dom.get("medical", {}).get("fpr", None),
        "medical_recall":   per_dom.get("medical", {}).get("recall", None),
        "legal_FPR":        per_dom.get("legal",   {}).get("fpr", None),
        "legal_recall":     per_dom.get("legal",   {}).get("recall", None),
        "finance_FPR":      per_dom.get("finance", {}).get("fpr", None),
        "finance_recall":   per_dom.get("finance", {}).get("recall", None),
        "code_FPR":         per_dom.get("code",    {}).get("fpr", None),
        "code_recall":      per_dom.get("code",    {}).get("recall", None),
        "struct_OOD_F1_agg":  struct_agg["f1"],
        "struct_OOD_FPR_agg": struct_agg["fpr"],
        "json_wrap_FPR":         per_tr.get("json_wrap",                {}).get("fpr", None),
        "json_wrap_recall":      per_tr.get("json_wrap",                {}).get("recall", None),
        "prompt_dilute_FPR":     per_tr.get("prompt_dilute",            {}).get("fpr", None),
        "prompt_dilute_recall":  per_tr.get("prompt_dilute",            {}).get("recall", None),
        "prefix_mimic_FPR":      per_tr.get("instruction_prefix_mimic", {}).get("fpr", None),
        "prefix_mimic_recall":   per_tr.get("instruction_prefix_mimic", {}).get("recall", None),
    }


def run(hard_neg: bool = False) -> None:
    run_train(num_epochs=2, batch_size=16, lr=5e-5, seed=42, hard_neg=hard_neg)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(
        description="Train DistilBERT binary classifier for prompt injection detection.",
        formatter_class=_ap.ArgumentDefaultsHelpFormatter,
    )
    _parser.add_argument("--epochs",    type=int,   default=2,    help="Training epochs")
    _parser.add_argument("--batch",     type=int,   default=16,   help="Per-device batch size")
    _parser.add_argument("--lr",        type=float, default=5e-5, help="Learning rate")
    _parser.add_argument("--seed",      type=int,   default=42)
    _parser.add_argument(
        "--hard-neg", action="store_true",
        help=(
            "Append hard_negative_train/val.csv to the training data. "
            "Outputs go to outputs/distilbert_hardneg/ instead of "
            "outputs/distilbert_baseline/. "
            "Run data_builder.build_hard_negative first."
        ),
    )
    _args = _parser.parse_args()
    run_train(
        num_epochs=_args.epochs,
        batch_size=_args.batch,
        lr=_args.lr,
        seed=_args.seed,
        hard_neg=_args.hard_neg,
    )
