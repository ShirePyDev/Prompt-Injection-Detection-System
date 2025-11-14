# src/baselines/tfidf_logreg.py
from __future__ import annotations
from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

DATA_DIR = Path("data/final_dataset")
OUT_DIR = Path("outputs/tfidf_baseline")
MODEL_DIR = Path("models/tfidf_logreg")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_split(name: str) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run preprocessing (Step 3).")
    return pd.read_csv(p)

def build_pipeline() -> Pipeline:
    # Reasonable defaults; ngram_range boosts performance on injector phrasing.
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )),
        ("clf", LogisticRegression(
            max_iter=500,
            class_weight="balanced",  # helpful if class imbalance exists
            n_jobs=None
        ))
    ])

def train_and_eval(grid_search: bool = False) -> None:
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")

    X_train, y_train = train_df["text"].astype(str), train_df["label"].astype(int)
    X_val,   y_val   = val_df["text"].astype(str),   val_df["label"].astype(int)
    X_test,  y_test  = test_df["text"].astype(str),  test_df["label"].astype(int)

    pipe = build_pipeline()

    if grid_search:
        # Small, safe grid; expands later if needed.
        param_grid = {
            "tfidf__ngram_range": [(1,1), (1,2)],
            "tfidf__min_df": [1, 2, 3],
            "clf__C": [0.25, 1.0, 2.0],
        }
        gs = GridSearchCV(pipe, param_grid, scoring="f1", cv=3, n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train) #type: ignore
        pipe = gs.best_estimator_
        with open(OUT_DIR / "grid_search_results.json", "w") as f:
            json.dump({
                "best_params": gs.best_params_,
                "best_score": gs.best_score_
            }, f, indent=2)
    else:
        pipe.fit(X_train, y_train)

    # Save model
    joblib.dump(pipe, MODEL_DIR / "tfidf_logreg.joblib")

    # Evaluate helper
    def eval_split(name: str, X, y):
        y_pred = pipe.predict(X)
        report = classification_report(y, y_pred, target_names=["benign","injection"], digits=4)
        cm = confusion_matrix(y, y_pred, labels=[0,1])

        out_txt = OUT_DIR / f"{name}_report.txt"
        with open(out_txt, "w") as f:
            f.write(report) # type: ignore
            f.write("\nConfusion Matrix [[TN, FP],[FN, TP]]:\n")
            f.write(str(cm))
        return out_txt, report, cm

    val_txt, val_report, val_cm   = eval_split("val",  X_val,  y_val)
    test_txt, test_report, test_cm = eval_split("test", X_test, y_test)

    # Tiny console summary
    print("✅ TF-IDF + Logistic Regression complete.")
    print("Saved model:", MODEL_DIR / "tfidf_logreg.joblib")
    print("Saved reports:")
    print(" -", val_txt)
    print(" -", test_txt)

def run(grid_search: bool = False) -> None:
    train_and_eval(grid_search=grid_search)

if __name__ == "__main__":
    run(grid_search=False)
