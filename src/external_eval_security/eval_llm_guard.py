# src/external_eval/eval_llm_guard.py

import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from llm_guard.input_scanners import PromptInjection


# ============================================================
# Configuration
# ============================================================

IN_PATH = Path("data/pids_bench_v3/test.csv")
OUT_PATH = Path("outputs/external_eval/test_llm_guard_output.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Dataset Loader
# ============================================================

def load_dataset(path):

    df = pd.read_csv(path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Dataset must contain 'text' and 'label'. Found: {df.columns.tolist()}"
        )

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    return df, texts, labels


# ============================================================
# Inference Function
# ============================================================

def run_inference(texts):

    scanner = PromptInjection()

    preds = []
    scores = []

    for text in tqdm(texts, desc="Evaluating LLM Guard"):

        sanitized_prompt, is_valid, risk_score = scanner.scan(text)

        # LLM Guard interpretation
        # True  = safe
        # False = injection

        pred = 0 if is_valid else 1

        preds.append(pred)
        scores.append(risk_score)

    return preds, scores


# ============================================================
# Metrics
# ============================================================

def compute_metrics(labels, preds):

    acc = accuracy_score(labels, preds)

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    return acc, prec, rec, f1


# ============================================================
# Main Evaluation
# ============================================================

def main():

    print("Loading dataset...")
    df, texts, labels = load_dataset(IN_PATH)

    print(f"Loaded {len(texts)} samples")

    print("Running LLM Guard evaluation...")

    start_time = time.time()

    preds, scores = run_inference(texts)

    elapsed = time.time() - start_time

    latency = elapsed / len(texts)

    acc, prec, rec, f1 = compute_metrics(labels, preds)

    print("\n=== LLM Guard Results (TEST SPLIT) ===")

    print("Input :", IN_PATH)
    print("Output:", OUT_PATH)
    print("Samples:", len(labels))

    print("Total Time:", round(elapsed, 2), "sec")
    print("Latency per prompt:", round(latency * 1000, 3), "ms")

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    df["llm_guard_pred"] = preds
    df["llm_guard_score"] = scores

    df.to_csv(OUT_PATH, index=False)

    print("\nSaved predictions to", OUT_PATH)


# ============================================================

if __name__ == "__main__":
    main()