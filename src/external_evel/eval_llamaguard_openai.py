# -----------------------------
# 1. Login to HuggingFace
# -----------------------------
from huggingface_hub import login
login("")   # replace with your real token


# -----------------------------
# 2. Imports
# -----------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------
# 3. Select Device (MPS if available)
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">>> Using MPS (Apple GPU)")
else:
    device = torch.device("cpu")
    print(">>> Using CPU")

# -----------------------------
# 4. Load Dataset
# -----------------------------
df = pd.read_csv("data/eval_subset/eval_3k_balanced.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()


# -----------------------------
# 5. Load Model + Tokenizer
# -----------------------------
model_name = "meta-llama/Llama-Guard-3-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, token=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token      # use EOS as PAD
    tokenizer.padding_side = "right"


model.to(device)
model.eval()


# -----------------------------
# 6. Batched Evaluation (MUCH faster)
# -----------------------------
batch_size = 1      # try 4 first, then 8 — if MPS runs out of memory, reduce to 2
preds = []

for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating LlamaGuard"):
    batch_texts = texts[i : i + batch_size]

    # Tokenize batch
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=512,
    ).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(**inputs).logits

    batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
    preds.extend(batch_preds)


# -----------------------------
# 7. Metrics
# -----------------------------
print("\n=== LlamaGuard Results ===")
print("Accuracy :", accuracy_score(labels, preds))
print("Precision:", precision_score(labels, preds))
print("Recall   :", recall_score(labels, preds))
print("F1 Score :", f1_score(labels, preds))

df["llamaguard_pred"] = preds
df.to_csv("data/eval_subset/eval_3k_lg_output.csv", index=False)

print("\nSaved predictions to data/eval_subset/eval_3k_lg_output.csv")
