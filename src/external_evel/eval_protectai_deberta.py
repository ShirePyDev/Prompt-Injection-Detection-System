import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =========================
# Device selection (MPS / CUDA / CPU)
# =========================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# =========================
# Load evaluation data
# =========================
df = pd.read_csv("data/eval_subset/eval_3k_balanced.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()   # 0 = benign, 1 = injection (same as your other models)

print(f"Loaded {len(df)} samples from eval_3k_balanced.csv")

# =========================
# Load ProtectAI DeBERTa prompt-injection model
# =========================
model_name = "protectai/deberta-v3-small-prompt-injection-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# =========================
# Run batched inference
# =========================
batch_size = 32
preds = []

for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating ProtectAI DeBERTa"):
    batch_texts = texts[i:i + batch_size]
    encodings = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encodings).logits
        batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()

    preds.extend(batch_preds)

# =========================
# Compute metrics
# =========================
acc = accuracy_score(labels, preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    labels, preds, average="binary", pos_label=1
)

print("\n=== ProtectAI DeBERTa Prompt-Injection Results ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# Save predictions for later analysis
df["deberta_pred"] = preds
df.to_csv("data/eval_subset/eval_3k_deberta_output.csv", index=False)
print("Saved predictions to data/eval_subset/eval_3k_deberta_output.csv")
