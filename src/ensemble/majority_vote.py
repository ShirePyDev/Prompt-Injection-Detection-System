# src/ensemble/majority_vote.py
from __future__ import annotations
from pathlib import Path
import json

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Paths
FINAL_DIR = Path("data/final_dataset")
TFIDF_MODEL = Path("models/tfidf_logreg/tfidf_logreg.joblib")
DISTILBERT_MODEL = Path("models/distilbert")

OUT_DIR = Path("outputs/ensemble_majority")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Load TF-IDF model
# ----------------------------------------------------
def load_tfidf():
    return joblib.load(TFIDF_MODEL)

# ----------------------------------------------------
# DistilBERT prediction helper
# ----------------------------------------------------
def load_distilbert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import math

    tok = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    BATCH_SIZE = 64   # SAFE for MPS

    def predict_texts(texts):
        preds = []
        n = len(texts)
        num_batches = math.ceil(n / BATCH_SIZE)

        for i in range(num_batches):
            batch = texts[i*BATCH_SIZE : (i+1)*BATCH_SIZE]

            enc = tok(
                batch,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                logits = model(**enc).logits

            batch_preds = logits.argmax(dim=-1).cpu().tolist()
            preds.extend(batch_preds)

        return preds

    return predict_texts



# ----------------------------------------------------
# Ensemble Majority Vote
# ----------------------------------------------------
def majority_vote(preds_list):
    preds = np.array(preds_list)  # shape: (n_models, n_samples)
    votes = preds.mean(axis=0)
    final = (votes >= 0.5).astype(int)
    return final

# ----------------------------------------------------
# Run ensemble on val/test
# ----------------------------------------------------
def run_ensemble():
    # Load splits
    val_df = pd.read_csv(FINAL_DIR / "val.csv")
    test_df = pd.read_csv(FINAL_DIR / "test.csv")

    # Load models
    tfidf_model = load_tfidf()
    distilbert_predict = load_distilbert()

    # ---- Validation ----
    X_val, y_val = val_df["text"].tolist(), val_df["label"].tolist()

    tfidf_val = tfidf_model.predict(X_val)
    distil_val = distilbert_predict(X_val)

    final_val = majority_vote([tfidf_val, distil_val])

    val_rep = classification_report(
        y_val, final_val,
        target_names=["benign","injection"],
        digits=4
    )
    val_cm = confusion_matrix(y_val, final_val, labels=[0,1])

    with open(OUT_DIR / "val_report.txt", "w") as f:
        f.write(val_rep)
        f.write("\nConfusion Matrix [[TN, FP],[FN, TP]]:\n")
        f.write(str(val_cm))

    # ---- Test ----
    X_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()

    tfidf_test = tfidf_model.predict(X_test)
    distil_test = distilbert_predict(X_test)

    final_test = majority_vote([tfidf_test, distil_test])

    test_rep = classification_report(
        y_test, final_test,
        target_names=["benign","injection"],
        digits=4
    )
    test_cm = confusion_matrix(y_test, final_test, labels=[0,1])

    with open(OUT_DIR / "test_report.txt", "w") as f:
        f.write(test_rep)
        f.write("\nConfusion Matrix [[TN, FP],[FN, TP]]:\n")
        f.write(str(test_cm))

    print("🚀 Ensemble complete.")
    print("Reports saved:")
    print(" -", OUT_DIR / "val_report.txt")
    print(" -", OUT_DIR / "test_report.txt")


if __name__ == "__main__":
    run_ensemble()
