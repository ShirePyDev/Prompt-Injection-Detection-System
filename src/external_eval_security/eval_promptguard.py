# src/external_eval/eval_promptguard.py
import os
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ============================================================
# 0. Paths
# ============================================================
IN_PATH = Path("data/pids_bench_v3/test.csv")
OUT_PATH = Path("outputs/external_eval/test_promptguard_output.csv")
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
# Recommended default: Prompt Guard 2 (binary: BENIGN vs MALICIOUS)
# You can override with:
#   export PROMPTGUARD_MODEL_ID="meta-llama/Prompt-Guard-86M"
# or:
#   export PROMPTGUARD_MODEL_ID="meta-llama/Llama-Prompt-Guard-2-22M"
MODEL_ID = os.getenv("PROMPTGUARD_MODEL_ID", "meta-llama/Llama-Prompt-Guard-2-86M")

# If model is gated:
#   export HF_TOKEN="your_hf_token"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")   # assign your Hugging Face token here

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)

# Ensure PAD token exists for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
tokenizer.padding_side = "right"

# Make sure the model also knows pad token id (common HF batching issue)
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id

model.to(device)
model.eval()

# ============================================================
# 4. Helper: Map model outputs -> (binary pred, score)
# ============================================================
def _safe_upper(x):
    return str(x).upper().strip()

def decide_pred_and_score(logits: torch.Tensor):
    """
    Returns:
      pred (int): 0 benign, 1 injection
      score (float): maliciousness score/probability-like value (best-effort)
    Works for:
      - Prompt Guard 2 (binary softmax, labels like BENIGN/MALICIOUS)
      - Prompt Guard 1 (often 3 outputs: BENIGN / INJECTION / JAILBREAK)
        If multi-label: uses sigmoid and considers injection/jailbreak as malicious.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    # Convert keys to int if they come as strings
    id2label_norm = {}
    for k, v in id2label.items():
        try:
            id2label_norm[int(k)] = v
        except Exception:
            pass

    num_labels = logits.shape[-1]

    # Heuristic: detect if model is multi-label
    problem_type = getattr(model.config, "problem_type", None)
    is_multilabel = (problem_type == "multi_label_classification")

    # Another heuristic: some models may not set problem_type; still treat as single-label
    # unless you explicitly set PROMPTGUARD_FORCE_MULTILABEL=1
    if os.getenv("PROMPTGUARD_FORCE_MULTILABEL", "0") == "1":
        is_multilabel = True

    # Threshold for multi-label decision (default 0.5)
    thr = float(os.getenv("PROMPTGUARD_THRESHOLD", "0.5"))

    if is_multilabel:
        probs = torch.sigmoid(logits).detach().cpu()
        probs_list = probs.tolist()

        preds = []
        scores = []

        for row in probs_list:
            # Identify indices for injection/jailbreak (if present); otherwise, fall back to max non-benign
            inj_ids = []
            jail_ids = []
            benign_ids = []

            for i in range(num_labels):
                lab = _safe_upper(id2label_norm.get(i, f"LABEL_{i}"))
                if "BENIGN" in lab or "SAFE" in lab:
                    benign_ids.append(i)
                if "INJECT" in lab:
                    inj_ids.append(i)
                if "JAIL" in lab or "JB" in lab:
                    jail_ids.append(i)
                if "MAL" in lab or "ATTACK" in lab:
                    # some models use MALICIOUS directly
                    inj_ids.append(i)

            # score = max(injection/jailbreak/malicious scores if known, else max over all labels except benign
            candidate_ids = list(set(inj_ids + jail_ids))
            if not candidate_ids:
                candidate_ids = [i for i in range(num_labels) if i not in benign_ids] or list(range(num_labels))

            score = max(row[i] for i in candidate_ids)
            pred = 1 if score >= thr else 0

            preds.append(pred)
            scores.append(float(score))

        return preds, scores

    # Single-label (softmax)
    probs = torch.softmax(logits, dim=-1).detach().cpu()
    probs_list = probs.tolist()
    preds = []
    scores = []

    # Identify malicious class id if labels exist
    malicious_ids = []
    benign_ids = []
    for i in range(num_labels):
        lab = _safe_upper(id2label_norm.get(i, f"LABEL_{i}"))
        if "MAL" in lab or "ATTACK" in lab or "INJECT" in lab or "JAIL" in lab:
            malicious_ids.append(i)
        if "BENIGN" in lab or "SAFE" in lab:
            benign_ids.append(i)

    # If Prompt Guard 2: typically one malicious id
    # If Prompt Guard 1 (3-way): treat INJECTION/JAILBREAK as malicious (best-effort)
    for row in probs_list:
        # Default predicted class
        pred_class = int(max(range(num_labels), key=lambda j: row[j]))

        # Score: if malicious ids exist, use sum or max of malicious probs; else use prob of pred_class
        if malicious_ids:
            # For 3-way, maliciousness = P(injection or jailbreak) => sum of those probs (capped at 1.0)
            if len(malicious_ids) >= 2:
                score = float(min(1.0, sum(row[j] for j in malicious_ids)))
            else:
                score = float(row[malicious_ids[0]])
        else:
            score = float(row[pred_class])

        # Pred: malicious if predicted class is malicious OR score >= 0.5 when we have a malicious score
        if malicious_ids:
            pred = 1 if (pred_class in malicious_ids or score >= 0.5) else 0
        else:
            # fallback: assume LABEL_1 is malicious (common HF convention)
            pred = 1 if pred_class == 1 else 0

        preds.append(pred)
        scores.append(score)

    return preds, scores


# ============================================================
# 5. Batched Inference
# ============================================================
batch_size = int(os.getenv("BATCH_SIZE", "8"))  # good starting point for MPS; reduce if OOM
preds_all = []
scores_all = []

t0 = time.time()

for i in tqdm(range(0, len(texts), batch_size), desc=f"Evaluating PromptGuard ({MODEL_ID})"):
    batch_texts = texts[i : i + batch_size]

    enc = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    batch_preds, batch_scores = decide_pred_and_score(logits)

    preds_all.extend(batch_preds)
    scores_all.extend(batch_scores)

elapsed = time.time() - t0
latency = elapsed / len(texts)

# ============================================================
# 6. Metrics
# ============================================================
acc = accuracy_score(labels, preds_all)
prec, rec, f1, _ = precision_recall_fscore_support(
    labels, preds_all, average="binary", pos_label=1, zero_division=0
)

print("\n=== PromptGuard Results (TEST SPLIT) ===")
print("Model :", MODEL_ID)
print("Input :", str(IN_PATH))
print("Output:", str(OUT_PATH))
print("N     :", len(labels))
print("Time  :", round(elapsed, 2), "sec")
print("Latency per prompt:", round(latency * 1000, 3), "ms")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# ============================================================
# 7. Save Predictions
# ============================================================
df["promptguard_pred"] = preds_all
df["promptguard_score"] = scores_all
df.to_csv(OUT_PATH, index=False)
print("\nSaved predictions to", OUT_PATH)