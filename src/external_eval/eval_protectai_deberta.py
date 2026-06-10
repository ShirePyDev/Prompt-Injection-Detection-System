# src/external_eval/eval_protectai_deberta.py
from pathlib import Path
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ============================================================
# 0. Paths (UPDATED to use the NEW official test split)
# ============================================================
IN_PATH = Path("data/pids_bench_v3/test.csv")
OUT_PATH = Path("outputs/external_eval/test_deberta_output.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Device selection (MPS / CUDA / CPU)
# ============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ============================================================
# 2. Load evaluation data
# ============================================================
df = pd.read_csv(IN_PATH)
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError(f"Input file must contain 'text' and 'label'. Got: {df.columns.tolist()}")

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()  # 0 = benign, 1 = injection
print(f"Loaded {len(df)} samples from {IN_PATH}")

# ============================================================
# 3. Load ProtectAI DeBERTa prompt-injection model
# ============================================================
model_name = "protectai/deberta-v3-small-prompt-injection-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# ============================================================
# 4. Run batched inference
# ============================================================
batch_size = 32
preds = []

t0 = time.time()

for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating ProtectAI DeBERTa"):
    batch_texts = texts[i : i + batch_size]

    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()

    preds.extend(batch_preds)

elapsed = time.time() - t0

# ============================================================
# 5. Compute metrics
# ============================================================
acc = accuracy_score(labels, preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    labels, preds, average="binary", pos_label=1, zero_division=0
)

print("\n=== ProtectAI DeBERTa Prompt-Injection Results (TEST SPLIT) ===")
print("Input :", str(IN_PATH))
print("Output:", str(OUT_PATH))
print("N     :", len(labels))
print("Time  :", round(elapsed, 2), "sec")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# ============================================================
# 6. Save predictions for later analysis
# ============================================================
df["deberta_pred"] = preds
df.to_csv(OUT_PATH, index=False)
print("Saved predictions to", OUT_PATH)
