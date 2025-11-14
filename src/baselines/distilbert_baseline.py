# src/baselines/distilbert_baseline.py
from __future__ import annotations
from pathlib import Path
import os, json
from dataclasses import dataclass
import inspect

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed
)
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score,
    classification_report, confusion_matrix
)

# ---- Safety for macOS MPS (Apple Silicon) ----
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DATA_DIR = Path("data/final_dataset")
OUT_DIR = Path("outputs/distilbert_baseline")
MODEL_DIR = Path("models/distilbert")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_splits() -> DatasetDict:
    tr = pd.read_csv(DATA_DIR / "train.csv")
    va = pd.read_csv(DATA_DIR / "val.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    # Ensure columns + types
    for df in (tr, va, te):
        assert {"text", "label"}.issubset(df.columns), f"Missing required columns in DataFrame: {df.columns}"
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
    return DatasetDict({
        "train": Dataset.from_pandas(tr, preserve_index=False),
        "validation": Dataset.from_pandas(va, preserve_index=False),
        "test": Dataset.from_pandas(te, preserve_index=False),
    })

def tokenize_each_split(ds: DatasetDict, tok: AutoTokenizer) -> DatasetDict:
    def _tok(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=256) #type: ignore
    out = {}
    for split in ds.keys():
        # remove all columns except "label"; do it per-split to avoid mismatches
        cols_to_remove = [c for c in ds[split].column_names if c != "label"]
        out[split] = ds[split].map(_tok, batched=True, remove_columns=cols_to_remove)
    return DatasetDict(out)

@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def run_train(num_epochs: int = 2, batch_size: int = 16, lr: float = 5e-5, seed: int = 42):
    set_seed(seed)
    device = pick_device()

    ds = load_splits()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    ds_tok = tokenize_each_split(ds, tok)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    ta_sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in ta_sig.parameters:
        strategy_kw = {"evaluation_strategy": "epoch"}
    else:
        strategy_kw = {"eval_strategy": "epoch"}  # backwards compat for older HF versions

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "trainer"),
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(OUT_DIR / "logs"),
        logging_steps=50,
        report_to=[],  # disable wandb
        seed=seed,
        save_total_limit=2,
        **strategy_kw,  # handle HF version differences #type: ignore
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,  #type: ignore
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # Save best model + tokenizer
    trainer.save_model(str(MODEL_DIR))
    tok.save_pretrained(str(MODEL_DIR))

    # ---- Evaluate on val/test and save detailed reports ----
    def eval_and_save(split: str):
        preds = trainer.predict(ds_tok[split])  #type: ignore
        y_true = preds.label_ids
        y_pred = np.argmax(preds.predictions, axis=-1)
        rep = classification_report(y_true, y_pred, target_names=["benign", "injection"], digits=4)     #type: ignore
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])    #type: ignore
        out_txt = OUT_DIR / f"{split}_report.txt"
        with open(out_txt, "w") as f:
            f.write(rep)    #type: ignore
            f.write("\nConfusion Matrix [[TN, FP],[FN, TP]]:\n")
            f.write(str(cm))
        return out_txt

    val_path = eval_and_save("validation")
    test_path = eval_and_save("test")

    # Compact run summary
    summary = {
        "best_model_dir": str(MODEL_DIR),
        "device": str(device),
        "args": {"epochs": num_epochs, "batch_size": batch_size, "lr": lr, "seed": seed},
        "reports": {"validation": str(val_path), "test": str(test_path)}
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ DistilBERT training complete.")
    print("Saved model ->", MODEL_DIR)
    print("Reports ->", val_path, test_path)

def run():
    # conservative defaults for CPU/MPS
    run_train(num_epochs=2, batch_size=16, lr=5e-5, seed=42)

if __name__ == "__main__":
    run()
