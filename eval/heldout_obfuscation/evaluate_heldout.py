"""
Evaluate all 8 detectors on the held-out obfuscation stress set.

Usage (Colab):
    !python evaluate_heldout.py \
        --heldout_csv /content/heldout_obfuscation.csv \
        --output_csv /content/heldout_results.csv \
        --internal_checkpoints_dir /content/drive/MyDrive/pids_bench_checkpoints \
        --tfidf_model_path /content/drive/MyDrive/pids_bench_checkpoints/tfidf_lr.pkl \
        --deberta_ft_threshold 0.502 \
        --distilbert_threshold 0.450 \
        --tfidf_threshold 0.518

Detectors evaluated (matches Section V-A of the paper):
    Internal baselines (trained on PIDS-Bench train split):
        - TF-IDF + Logistic Regression
        - DistilBERT (fine-tuned)
        - DeBERTa-v3-FT (fine-tuned)
    External prompt-injection detectors:
        - ProtectAI/deberta-v3-base-prompt-injection-v2
        - deepset/deberta-v3-base-injection
    Broad-safety comparators:
        - meta-llama/Prompt-Guard-2-86M
        - meta-llama/Llama-Guard-3-1B

The rule-based baseline is omitted (it was moved to appendix per Phase 2 of
revision, and its near-zero IID recall makes held-out recall uninformative).

For each detector, we report:
    - recall per transform (leetspeak, homoglyph, zero_width)
    - aggregate held-out recall
    - the threshold used (for internal models: validation-tuned;
      for external and broad-safety: native 0.5)

Output CSV columns:
    model, transform, n, correct, recall, threshold

This file does not evaluate Llama Guard 3-1B's full chat template or
PromptGuard 2's two-class head in their deployed form — we use the simplest
"is this an injection?" probability from each model, matching the paper's
main-table evaluation protocol.
"""

import argparse
import inspect
import pickle
from pathlib import Path
from typing import Callable, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Detector wrappers: each returns predict(texts) -> np.ndarray of 0/1
# ---------------------------------------------------------------------------

def _patch_logreg(pipe) -> None:
    """sklearn >=1.8 removed multi_class; restore it for 1.7.x compatibility."""
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
    """TF-IDF + LogisticRegression joblib pipeline."""
    model = joblib.load(model_path)
    _patch_logreg(model)

    def predict(texts: List[str]) -> np.ndarray:
        proba = model.predict_proba(texts)[:, 1]
        return (proba >= threshold).astype(int)

    return predict


def load_hf_binary(model_name_or_path: str, threshold: float,
                   max_len: int = 512, positive_label_idx: int = 1) -> Callable:
    """
    Generic HuggingFace binary classifier loader.

    Works for:
        - DistilBERT (your fine-tune)
        - DeBERTa-v3-FT (your fine-tune)
        - ProtectAI-small
        - DeBERTa-PI (deepset)

    For PromptGuard 2 and Llama Guard 3-1B we use a dedicated wrapper below
    because their label spaces differ.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path
    ).to(DEVICE).eval()
    accepted_keys = set(inspect.signature(model.forward).parameters.keys())

    @torch.no_grad()
    def predict(texts: List[str]) -> np.ndarray:
        preds = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items() if k in accepted_keys}
            logits = model(**enc).logits
            proba = torch.softmax(logits, dim=-1)[:, positive_label_idx]
            preds.extend((proba.cpu().numpy() >= threshold).astype(int).tolist())
        return np.array(preds)

    return predict


def load_promptguard2(threshold: float = 0.5) -> Callable:
    """
    PromptGuard 2 (86M) has three classes: BENIGN, INJECTION, JAILBREAK.
    We treat INJECTION+JAILBREAK as positive, matching the paper's protocol.
    """
    model_name = "meta-llama/Prompt-Guard-2-86M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    ).to(DEVICE).eval()
    pg2_accepted_keys = set(inspect.signature(model.forward).parameters.keys())

    # Label indices depend on the model config. We read them from config.
    label2id = model.config.label2id
    # Positive class = anything other than "BENIGN"
    pos_indices = [idx for lbl, idx in label2id.items() if lbl.upper() != "BENIGN"]

    @torch.no_grad()
    def predict(texts: List[str]) -> np.ndarray:
        preds = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items() if k in pg2_accepted_keys}
            logits = model(**enc).logits
            proba = torch.softmax(logits, dim=-1)
            pos_proba = proba[:, pos_indices].sum(dim=-1)
            preds.extend((pos_proba.cpu().numpy() >= threshold).astype(int).tolist())
        return np.array(preds)

    return predict


def load_llamaguard3_1b(threshold: float = 0.5) -> Callable:
    """
    Llama Guard 3-1B classifies via a chat template. The standard approach
    is to wrap the user input in the model's chat template and check whether
    the next-token probability of "unsafe" exceeds threshold.

    For this evaluation we follow the same protocol used in the paper's
    main table: compute P(unsafe) and threshold at 0.5.
    """
    from transformers import AutoModelForCausalLM

    model_name = "meta-llama/Llama-Guard-3-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(DEVICE).eval()

    unsafe_token_id = tokenizer.convert_tokens_to_ids("unsafe")
    safe_token_id = tokenizer.convert_tokens_to_ids("safe")

    @torch.no_grad()
    def predict(texts: List[str]) -> np.ndarray:
        preds = []
        for text in texts:
            messages = [{"role": "user", "content": text}]
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt"
            ).to(DEVICE)
            logits = model(input_ids).logits[0, -1]
            # Renormalize over just {safe, unsafe}
            pair_logits = torch.stack([
                logits[safe_token_id], logits[unsafe_token_id]
            ])
            unsafe_proba = torch.softmax(pair_logits, dim=-1)[1].item()
            preds.append(int(unsafe_proba >= threshold))
        return np.array(preds)

    return predict


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_detector(
    name: str, predict_fn: Callable, df: pd.DataFrame, threshold: float
) -> List[Dict]:
    """
    Run predict_fn on every row grouped by transform. Compute recall per
    transform + aggregate.
    """
    results = []
    # aggregate across transforms
    all_texts = df["transformed_text"].tolist()
    print(f"  [{name}] predicting {len(all_texts)} rows...")
    preds = predict_fn(all_texts)
    df = df.copy()
    df["pred"] = preds

    for transform in sorted(df["transform"].unique()):
        sub = df[df["transform"] == transform]
        n = len(sub)
        correct = int(sub["pred"].sum())  # label is always 1
        recall = correct / n if n > 0 else 0.0
        results.append({
            "model": name,
            "transform": transform,
            "n": n,
            "correct": correct,
            "recall": recall,
            "threshold": threshold,
        })
        print(f"    {transform:12} n={n:3}  recall={recall:.4f}")

    # aggregate row
    n_total = len(df)
    total_correct = int(df["pred"].sum())
    total_recall = total_correct / n_total if n_total > 0 else 0.0
    results.append({
        "model": name,
        "transform": "AGGREGATE",
        "n": n_total,
        "correct": total_correct,
        "recall": total_recall,
        "threshold": threshold,
    })
    print(f"    {'AGGREGATE':12} n={n_total:3}  recall={total_recall:.4f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--internal_checkpoints_dir", required=True,
                    help="Dir containing distilbert_ft/ and deberta_v3_ft/")
    ap.add_argument("--tfidf_model_path", required=True)
    ap.add_argument("--deberta_ft_threshold", type=float, default=0.502)
    ap.add_argument("--distilbert_threshold", type=float, default=0.450)
    ap.add_argument("--tfidf_threshold", type=float, default=0.518)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Model names to skip (e.g. --skip llamaguard3)")
    args = ap.parse_args()

    df = pd.read_csv(args.heldout_csv)
    print(f"Loaded {len(df)} held-out obfuscation rows")
    print(f"Device: {DEVICE}")

    ckpt_dir = Path(args.internal_checkpoints_dir)

    detector_configs = [
        ("tfidf_lr", args.tfidf_threshold,
         lambda: load_tfidf(args.tfidf_model_path, args.tfidf_threshold)),
        ("distilbert_ft", args.distilbert_threshold,
         lambda: load_hf_binary(str(ckpt_dir / "distilbert_ft"),
                                args.distilbert_threshold, max_len=256)),
        ("deberta_v3_ft", args.deberta_ft_threshold,
         lambda: load_hf_binary(str(ckpt_dir / "deberta_v3_ft"),
                                args.deberta_ft_threshold, max_len=512)),
        ("protectai_small", 0.5,
         lambda: load_hf_binary(
             "ProtectAI/deberta-v3-base-prompt-injection-v2", 0.5)),
        ("deberta_pi", 0.5,
         lambda: load_hf_binary(
             "deepset/deberta-v3-base-injection", 0.5)),
        ("promptguard2", 0.5,
         lambda: load_promptguard2(0.5)),
        ("llamaguard3_1b", 0.5,
         lambda: load_llamaguard3_1b(0.5)),
    ]

    all_results = []
    for name, threshold, loader in detector_configs:
        if name in args.skip:
            print(f"\n=== SKIPPING {name} (--skip) ===")
            continue
        print(f"\n=== Evaluating {name} (threshold={threshold}) ===")
        try:
            predict_fn = loader()
        except Exception as e:
            print(f"  FAILED to load {name}: {e}")
            continue
        try:
            results = evaluate_detector(name, predict_fn, df, threshold)
            all_results.extend(results)
        except Exception as e:
            print(f"  FAILED to evaluate {name}: {e}")
            continue
        # free memory
        del predict_fn
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nWrote {len(out_df)} result rows to {args.output_csv}")

    # Print publication-ready table
    print("\n" + "=" * 70)
    print("HELD-OUT OBFUSCATION RESULTS (for paper Table 10)")
    print("=" * 70)
    pivot = out_df.pivot(index="model", columns="transform", values="recall")
    cols = ["leetspeak", "homoglyph", "zero_width", "AGGREGATE"]
    pivot = pivot[[c for c in cols if c in pivot.columns]]
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
