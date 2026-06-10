import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ============================================================
# Paths
# ============================================================

IN_PATH = Path("data/pids_bench_v3/test.csv")
OUT_PATH = Path("outputs/external_eval/test_deberta_pi_output.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">>> Using MPS (Apple GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(">>> Using CUDA")
else:
    device = torch.device("cpu")
    print(">>> Using CPU")


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(IN_PATH)

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()

print(f"Loaded {len(df)} samples from {IN_PATH}")


# ============================================================
# Load DeBERTa PI model
# ============================================================

model_name = "protectai/deberta-v3-base-prompt-injection-v2"

print("Loading model:", model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.to(device)
model.eval()


# ============================================================
# Inference
# ============================================================

batch_size = 8

preds = []
scores = []

t0 = time.time()

for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating DeBERTa PI"):

    batch_texts = texts[i:i + batch_size]

    enc = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)

    batch_scores = probs[:, 1].cpu().tolist()
    batch_preds = (probs[:, 1] > 0.5).int().cpu().tolist()

    preds.extend(batch_preds)
    scores.extend(batch_scores)

elapsed = time.time() - t0


# ============================================================
# Metrics
# ============================================================

acc = accuracy_score(labels, preds)

prec, rec, f1, _ = precision_recall_fscore_support(
    labels,
    preds,
    average="binary",
    pos_label=1,
    zero_division=0
)

latency = elapsed / len(texts)

print("\n=== DeBERTa Prompt Injection Detector ===")
print("N:", len(labels))
print("Time:", round(elapsed, 2), "sec")
print("Latency per prompt:", round(latency * 1000, 3), "ms")

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# Save predictions
# ============================================================

df["deberta_pi_pred"] = preds
df["deberta_pi_score"] = scores

df.to_csv(OUT_PATH, index=False)

print("\nSaved predictions to", OUT_PATH)