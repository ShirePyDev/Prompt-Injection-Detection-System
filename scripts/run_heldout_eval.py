"""
Run Phase 5 held-out obfuscation evaluation for internal models.

Usage:
    python scripts/run_heldout_eval.py

Evaluates all 4 internal baselines on the held-out obfuscation stress set
and writes outputs/heldout_results.csv (matches evaluate_heldout.py format).

Models evaluated:
    tfidf_lr             outputs/multi_seed_runs/tfidf/seed_42/model/tfidf_logreg.joblib
    distilbert_ft        models/distilbert/
    distilbert_hardneg   models/distilbert_hardneg/
    deberta_v3_ft        models/deberta_v3/
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, List

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# MPS on Mac, CUDA on Linux, CPU fallback
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Device: {DEVICE}")

HELDOUT_CSV = Path("outputs/heldout_obfuscation.csv")
OUTPUT_CSV = Path("outputs/heldout_results.csv")

DETECTORS = [
    {
        "name": "tfidf_lr",
        "type": "tfidf",
        "path": "outputs/multi_seed_runs/tfidf/seed_42/model/tfidf_logreg.joblib",
        "threshold": 0.518,
    },
    {
        "name": "distilbert_ft",
        "type": "hf",
        "path": "models/distilbert",
        "threshold": 0.45,
        "max_len": 256,
    },
    {
        "name": "distilbert_hardneg",
        "type": "hf",
        "path": "models/distilbert_hardneg",
        "threshold": 0.298,
        "max_len": 256,
    },
    {
        "name": "deberta_v3_ft",
        "type": "hf",
        "path": "models/deberta_v3",
        "threshold": 0.502,
        "max_len": 512,
    },
]


def _patch_logreg(pipe) -> None:
    try:
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from packaging.version import Version
        clf = pipe.steps[-1][1]
        if isinstance(clf, LogisticRegression):
            if Version(sklearn.__version__) < Version("1.8.0") and not hasattr(clf, "multi_class"):
                clf.multi_class = "auto"
    except Exception:
        pass


def load_tfidf(model_path: str, threshold: float) -> Callable:
    model = joblib.load(model_path)
    _patch_logreg(model)

    def predict(texts: List[str]) -> np.ndarray:
        proba = model.predict_proba(texts)[:, 1]
        return (proba >= threshold).astype(int)

    return predict


def load_hf(model_path: str, threshold: float, max_len: int) -> Callable:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE).eval()
    accepted_keys = set(inspect.signature(model.forward).parameters.keys())

    @torch.no_grad()
    def predict(texts: List[str]) -> np.ndarray:
        preds = []
        for i in range(0, len(texts), 16):
            batch = texts[i : i + 16]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items() if k in accepted_keys}
            logits = model(**enc).logits
            proba = torch.softmax(logits, dim=-1)[:, 1]
            preds.extend((proba.cpu().numpy() >= threshold).astype(int).tolist())
        return np.array(preds)

    return predict


def evaluate(name: str, predict_fn: Callable, df: pd.DataFrame, threshold: float) -> List[dict]:
    texts = df["transformed_text"].tolist()
    print(f"  [{name}] predicting {len(texts)} rows ...")
    preds = predict_fn(texts)
    df = df.copy()
    df["pred"] = preds
    results = []
    for transform in sorted(df["transform"].unique()):
        sub = df[df["transform"] == transform]
        n = len(sub)
        correct = int(sub["pred"].sum())
        recall = correct / n if n else 0.0
        results.append({"model": name, "transform": transform,
                         "n": n, "correct": correct, "recall": recall, "threshold": threshold})
        print(f"    {transform:12}  n={n:3}  recall={recall:.4f}")
    n_total = len(df)
    total_correct = int(df["pred"].sum())
    total_recall = total_correct / n_total if n_total else 0.0
    results.append({"model": name, "transform": "AGGREGATE",
                     "n": n_total, "correct": total_correct,
                     "recall": total_recall, "threshold": threshold})
    print(f"    {'AGGREGATE':12}  n={n_total:3}  recall={total_recall:.4f}")
    return results


def main() -> None:
    df = pd.read_csv(HELDOUT_CSV)
    print(f"Loaded {len(df)} held-out rows  "
          f"({df['transform'].value_counts().to_dict()})\n")

    all_results = []
    for cfg in DETECTORS:
        print(f"\n=== {cfg['name']} (threshold={cfg['threshold']}) ===")
        try:
            if cfg["type"] == "tfidf":
                predict_fn = load_tfidf(cfg["path"], cfg["threshold"])
            else:
                predict_fn = load_hf(cfg["path"], cfg["threshold"], cfg.get("max_len", 512))
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue
        results = evaluate(cfg["name"], predict_fn, df, cfg["threshold"])
        all_results.extend(results)
        del predict_fn
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    out_df = pd.DataFrame(all_results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(out_df)} rows to {OUTPUT_CSV}")

    print("\n" + "=" * 65)
    print("TABLE 10 — HELD-OUT OBFUSCATION RECALL")
    print("=" * 65)
    pivot = out_df.pivot(index="model", columns="transform", values="recall")
    cols = [c for c in ["leetspeak", "homoglyph", "zero_width", "AGGREGATE"] if c in pivot.columns]
    print(pivot[cols].round(4).to_string())


if __name__ == "__main__":
    main()
