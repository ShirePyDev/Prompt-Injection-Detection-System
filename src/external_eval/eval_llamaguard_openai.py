# src/external_eval/eval_llamaguard_openai.py
#
# DEPRECATED for Llama Guard 3: this file uses AutoModelForSequenceClassification, which does not
# match the published HF interface (AutoModelForCausalLM + generation). For correct multiaxis
# evaluation use: python scripts/run_external_multiaxis.py --model llamaguard
import os
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ============================================================
# 0. Paths (UPDATED to use the NEW official test split)
# ============================================================
IN_PATH = Path("data/pids_bench_v3/test.csv")
OUT_PATH = Path("outputs/external_eval/test_llamaguard_output.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Device (MPS / CUDA / CPU)
# ============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">>> Using MPS (Apple GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(">>> Using CUDA GPU")
else:
    device = torch.device("cpu")
    print(">>> Using CPU")

# ============================================================
# 2. Load Dataset
# ============================================================
df = pd.read_csv(IN_PATH)
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError(f"Input file must contain 'text' and 'label'. Got: {df.columns.tolist()}")

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()
print(f"Loaded {len(df)} samples from {IN_PATH}")

# ============================================================
# 3. Load Model + Tokenizer (HF token via env var)
# ============================================================
# In terminal:
#   export HF_TOKEN="your_hf_token"
HF_TOKEN = os.getenv("HF_TOKEN")  # optional, but needed if the model requires auth
model_name = "meta-llama/Llama-Guard-3-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(model_name, token=HF_TOKEN)

# Ensure PAD token exists for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# Make sure the MODEL also knows the padding token id
model.config.pad_token_id = tokenizer.pad_token_id

model.to(device)
model.eval()

# ============================================================
# 4. Batched Inference
# ============================================================
batch_size = 4  # start with 4 on MPS; if OOM, reduce to 2 or 1
preds = []

t0 = time.time()

for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating LlamaGuard"):
    batch_texts = texts[i : i + batch_size]

    enc = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,       # IMPORTANT for batching
        max_length=512,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()

    preds.extend(batch_preds)

elapsed = time.time() - t0

# ============================================================
# 5. Metrics
# ============================================================
acc = accuracy_score(labels, preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    labels, preds, average="binary", pos_label=1, zero_division=0
)

print("\n=== LlamaGuard Results (TEST SPLIT) ===")
print("Input :", str(IN_PATH))
print("Output:", str(OUT_PATH))
print("N     :", len(labels))
print("Time  :", round(elapsed, 2), "sec")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# Save predictions
df["llamaguard_pred"] = preds
df.to_csv(OUT_PATH, index=False)
print("\nSaved predictions to", OUT_PATH)
