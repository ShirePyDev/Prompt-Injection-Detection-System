# src/analysis/compare_models.py
from __future__ import annotations
from pathlib import Path
import sys, json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# ensure project root in sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import from your own code
from src.baselines.rule_based import rule_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DATA_DIR = Path("data/final_dataset")
REPORT_DIR = Path("reports")
PLOT_DIR = Path("outputs/plots")
MODEL_DIR = Path("models")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(split: str):
    df = pd.read_csv(DATA_DIR / f"{split}.csv")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).values
    return texts, labels


def load_tfidf_model():
    path = MODEL_DIR / "tfidf_logreg" / "tfidf_logreg.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing TF-IDF model at {path}")
    return joblib.load(path)


def load_distilbert_pipeline():
    dist_dir = MODEL_DIR / "distilbert"
    if not dist_dir.exists():
        raise FileNotFoundError(f"Missing DistilBERT model dir at {dist_dir}")
    tok = AutoTokenizer.from_pretrained(str(dist_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(dist_dir))
    clf = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        return_all_scores=True,
        truncation=True,
    )
    return clf


def metric_dict(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def proba_rule(texts):
    return np.array([float(rule_score(t)) for t in texts])


def proba_tfidf(model, texts):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    else:
        from scipy.special import expit
        scores = model.decision_function(texts)
        return expit(scores)


def proba_distilbert(pipe, texts):
    out = pipe(texts, truncation=True)
    probs = []
    for row in out:
        p1 = 0.0
        for d in row:
            if d["label"] in ("LABEL_1", "1", "injection"):
                p1 = float(d["score"])
                break
        probs.append(p1)
    return np.array(probs, dtype=float)


def majority_vote(pr, pt, pb, thresh=0.5):
    votes = np.column_stack([
        (pr >= thresh).astype(int),
        (pt >= thresh).astype(int),
        (pb >= thresh).astype(int),
    ])
    return (votes.sum(axis=1) >= 2).astype(int)


def run():
    # load models
    tfidf_model = load_tfidf_model()
    dist_pipe = load_distilbert_pipeline()

    results = {"val": {}, "test": {}}
    models = ["rule", "tfidf", "distilbert", "ensemble_majority"]

    for split in ["val", "test"]:
        texts, y_true = load_data(split)

        pr = proba_rule(texts)
        pt = proba_tfidf(tfidf_model, texts)
        pb = proba_distilbert(dist_pipe, texts)

        y_rule = (pr >= 0.5).astype(int)
        y_tfidf = (pt >= 0.5).astype(int)
        y_bert = (pb >= 0.5).astype(int)
        y_ens = majority_vote(pr, pt, pb, thresh=0.5)

        preds_map = {
            "rule": y_rule,
            "tfidf": y_tfidf,
            "distilbert": y_bert,
            "ensemble_majority": y_ens,
        }

        results[split] = {}
        for name in models:
            y_pred = preds_map[name]
            results[split][name] = metric_dict(y_true, y_pred)

            # also save a text report
            rep = classification_report(
                y_true, y_pred,     #type: ignore
                target_names=["benign", "injection"],
                digits=4,
            )
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])    #type: ignore
            txt_path = REPORT_DIR / f"{split}_{name}_report.txt"
            with open(txt_path, "w") as f:
                f.write(rep)    #type: ignore
                f.write("\nConfusion Matrix [[TN, FP],[FN, TP]]:\n")
                f.write(str(cm))

    # save JSON summary
    json_path = REPORT_DIR / "model_comparison_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # make a simple F1 bar plot for test split
    test_metrics = results["test"]
    labels = list(test_metrics.keys())
    f1_scores = [test_metrics[m]["f1"] for m in labels]

    plt.figure()
    plt.bar(labels, f1_scores)
    plt.ylabel("F1 score")
    plt.title("Model comparison (test split)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plot_path = PLOT_DIR / "test_f1_comparison.png"
    plt.savefig(plot_path)
    plt.close()

    # also create a markdown table for your report
    md_lines = ["# Model Comparison (Test Split)\n",
                "| Model | Accuracy | Precision | Recall | F1 |",
                "|-------|----------|-----------|--------|----|"]
    for name in labels:
        m = test_metrics[name]
        md_lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
        )

    md_path = REPORT_DIR / "model_comparison_test.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print("✅ Comparison complete.")
    print("JSON summary ->", json_path)
    print("Markdown table ->", md_path)
    print("F1 bar plot ->", plot_path)
