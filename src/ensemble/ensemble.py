# src/ensemble/ensemble.py
from __future__ import annotations
from pathlib import Path
import sys, json
from dataclasses import dataclass
import argparse

import pandas as pd
import numpy as np
import joblib
import torch

# Ensure project root on path (so we can import src.*)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imports from your project
from src.baselines.rule_based import rule_score  # reuse your heuristic scorer

# Hugging Face
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Combine rule, TF-IDF, and DistilBERT models for prompt injection detection.")
    parser.add_argument(
        "--strategy",
        choices=["majority", "meta", "both"],
        default="both",
        help="Which ensemble strategy to run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for DistilBERT inference ('auto', 'cpu', 'mps', 'cuda', 'cuda:0', ...).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for probabilities used by majority vote.",
    )
    parser.add_argument(
        "--rule-confidence",
        type=float,
        default=1.0,
        help="Scaling factor (0-1) applied to the rule-based scores before ensembling.",
    )
    return parser.parse_args()


def resolve_pipeline_device(device_arg: str):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return -1
    if device_arg == "cpu":
        return -1
    if device_arg == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError("MPS requested but not available on this machine.")
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() == False.")
        if device_arg == "cuda":
            return 0
        # allow cuda:1 style
        _, _, idx = device_arg.partition(":")
        return int(idx) if idx else 0
    return device_arg

# Use the fully merged splits for evaluation/inference
DATA_DIR = Path("data/final_dataset")
OUT_DIR = Path("outputs/ensemble")
MODEL_DIR = Path("models")
ENSEMBLE_DIR = MODEL_DIR / "ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

# --- Utility checks ---

def ensure_inputs_ready():
    missing = []
    for csv in ("val.csv", "test.csv"):
        if not (DATA_DIR / csv).exists():
            missing.append(f"Missing processed split: {DATA_DIR / csv}")
    tfidf_model = MODEL_DIR / "tfidf_logreg" / "tfidf_logreg.joblib"
    if not tfidf_model.exists():
        missing.append(f"Missing TF-IDF model: {tfidf_model}")
    distilbert_dir = MODEL_DIR / "distilbert"
    if not distilbert_dir.exists():
        missing.append(f"Missing DistilBERT directory: {distilbert_dir}")
    if missing:
        raise FileNotFoundError("Cannot run ensemble – prerequisites not met:\n" + "\n".join(missing))

# --- Loaders for individual models ---

def load_tfidf():
    p = MODEL_DIR / "tfidf_logreg" / "tfidf_logreg.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Missing TF-IDF model at {p}")
    return joblib.load(p)

def load_distilbert(device):
    dir_ = MODEL_DIR / "distilbert"
    if not dir_.exists():
        raise FileNotFoundError(f"Missing DistilBERT model dir at {dir_}")
    tok = AutoTokenizer.from_pretrained(str(dir_))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(dir_))
    clf = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        return_all_scores=True,
        truncation=True,
        device=device,
    )
    return clf

# --- Helpers to get predictions/probabilities ---

def proba_rule(texts: list[str], confidence: float = 1.0) -> np.ndarray:
    # Heuristic has no prob; use 0/1 as a hard score, cast to proba
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return np.array([float(rule_score(t)) * confidence for t in texts])

def proba_tfidf(tfidf_clf, texts: list[str]) -> np.ndarray:
    # Probability for class 1 (injection)
    # If model lacks predict_proba (very rare), fallback to decision_function→sigmoid
    if hasattr(tfidf_clf, "predict_proba"):
        probs = tfidf_clf.predict_proba(texts)[:, 1]
    else:
        from scipy.special import expit
        scores = tfidf_clf.decision_function(texts)
        probs = expit(scores)
    return probs.astype(float)

def proba_distilbert(hf_pipe, texts: list[str]) -> np.ndarray:
    # pipeline returns list of list[ {"label": "LABEL_0/1", "score": prob}, ... ]
    out = hf_pipe(texts, truncation=True)
    probs = []
    for row in out:
        # find probability for LABEL_1 (assumed injection)
        p1 = 0.0
        for d in row:
            if d["label"] in ("LABEL_1", "1", "injection"):
                p1 = float(d["score"])
                break
        probs.append(p1)
    return np.array(probs, dtype=float)

# --- Ensemble strategies ---

@dataclass
class EnsembleOutputs:
    val_report_path: Path
    test_report_path: Path
    meta_model_path: Path | None

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

def majority_vote(p_rule: np.ndarray, p_tfidf: np.ndarray, p_bert: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    votes = np.column_stack([
        (p_rule >= thresh).astype(int),
        (p_tfidf >= thresh).astype(int),
        (p_bert >= thresh).astype(int)
    ])
    # majority (>=2) wins
    return (votes.sum(axis=1) >= 2).astype(int)

def meta_learner_train(val_X: np.ndarray, val_y: np.ndarray) -> LogisticRegression:
    meta = LogisticRegression(max_iter=200, class_weight="balanced")
    meta.fit(val_X, val_y)
    return meta

def eval_and_save(y_true: np.ndarray, y_pred: np.ndarray, split: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep = classification_report(y_true, y_pred, target_names=["benign","injection"], digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    out = OUT_DIR / f"{split}_report.txt"
    with open(out, "w") as f:
        f.write(rep)    #type: ignore
        f.write("\nConfusion Matrix [[TN, FP],[FN, TP]]:\n")
        f.write(str(cm))
    return out

def run(args=None):
    if args is None:
        args = parse_args()

    ensure_inputs_ready()
    device = resolve_pipeline_device(args.device)
    if not 0.0 <= args.rule_confidence <= 1.0:
        raise ValueError("--rule-confidence must be between 0 and 1.")

    # --- Load data ---
    val_df  = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    texts_val, y_val = val_df["text"].astype(str).tolist(), val_df["label"].astype(int).values
    texts_tst, y_tst = test_df["text"].astype(str).tolist(), test_df["label"].astype(int).values

    # --- Load models ---
    tfidf = load_tfidf()
    bert_pipe = load_distilbert(device=device)

    # --- Get probabilities ---
    pr_val  = proba_rule(texts_val, confidence=args.rule_confidence)
    pt_val  = proba_tfidf(tfidf, texts_val)
    pb_val  = proba_distilbert(bert_pipe, texts_val)

    pr_tst  = proba_rule(texts_tst, confidence=args.rule_confidence)
    pt_tst  = proba_tfidf(tfidf, texts_tst)
    pb_tst  = proba_distilbert(bert_pipe, texts_tst)

    summary = {"reports": {}, "meta_model_path": None, "notes": "LABEL_1 treated as 'injection' for DistilBERT outputs."}

    if args.strategy in ("majority", "both"):
        y_pred_val_mv = majority_vote(pr_val, pt_val, pb_val, thresh=args.threshold)
        y_pred_tst_mv = majority_vote(pr_tst, pt_tst, pb_tst, thresh=args.threshold)
        mv_val_path = eval_and_save(y_val, y_pred_val_mv, "val_majority")   #type: ignore
        mv_tst_path = eval_and_save(y_tst, y_pred_tst_mv, "test_majority")  #type: ignore
        summary["reports"].update({
            "val_majority": str(mv_val_path),
            "test_majority": str(mv_tst_path),
        })

    if args.strategy in ("meta", "both"):
        val_X = np.column_stack([pr_val, pt_val, pb_val])
        tst_X = np.column_stack([pr_tst, pt_tst, pb_tst])
        meta = meta_learner_train(val_X, y_val) #type: ignore
        y_pred_val_meta = meta.predict(val_X)
        y_pred_tst_meta = meta.predict(tst_X)
        meta_val_path = eval_and_save(y_val, y_pred_val_meta, "val_meta") #type: ignore
        meta_tst_path = eval_and_save(y_tst, y_pred_tst_meta, "test_meta")  #type: ignore
        meta_path = ENSEMBLE_DIR / "meta_logreg.joblib"
        joblib.dump(meta, meta_path)
        summary["reports"].update({
            "val_meta": str(meta_val_path),
            "test_meta": str(meta_tst_path),
        })
        summary["meta_model_path"] = str(meta_path)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Ensemble complete.")
    print("Reports ->", summary["reports"])
    if summary["meta_model_path"]:
        print("Meta model ->", summary["meta_model_path"])
    print("Device used:", device)
