from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

PRED_CSV = Path("outputs/distilbert_baseline/test_predictions.csv")
TEST_CSV = Path("data/final_dataset/test.csv")
OUT_DIR = Path("outputs/distilbert_baseline/source_breakdown")
LOW_SUPPORT_N = 30

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions
df_pred = pd.read_csv(PRED_CSV)
required = {"text", "true_label", "predicted_label"}
if not required.issubset(df_pred.columns):
    raise ValueError(f"Prediction file missing columns: {required - set(df_pred.columns)}")

# Load test metadata
df_test = pd.read_csv(TEST_CSV)
if not {"text", "source"}.issubset(df_test.columns):
    raise ValueError("test.csv must contain columns: text, source")

# Prefer prediction source if already available and complete.
# This is the cleanest path because predictions were generated from test.csv rows.
if "source" in df_pred.columns and not df_pred["source"].isna().any():
    df = df_pred.copy()
    source_col = "source"
else:
    # Fallback: attach source from test split.
    # First try strict row alignment (same order and length).
    if len(df_pred) == len(df_test) and (df_pred["text"].astype(str).values == df_test["text"].astype(str).values).all():
        df = df_pred.copy()
        df["source"] = df_test["source"].astype(str).values
        source_col = "source"
    else:
        # Last-resort merge on text+label to reduce accidental collisions.
        right = df_test.rename(columns={"label": "true_label"})[["text", "true_label", "source"]]
        df = df_pred.merge(right, on=["text", "true_label"], how="left", suffixes=("_pred", "_test"))
        source_col = "source"

if source_col not in df.columns:
    raise ValueError(f"No source column found after source attachment. Columns: {list(df.columns)}")

if df[source_col].isna().any():
    missing_n = int(df[source_col].isna().sum())
    raise ValueError(f"{missing_n} rows missing source after source attachment.")

rows = []
print("Columns after merge:", df.columns.tolist())

for src, sub in df.groupby(source_col):
    y_true = sub["true_label"].astype(int)
    y_pred = sub["predicted_label"].astype(int)
    n = len(sub)

    n_benign = int((y_true == 0).sum())
    n_injection = int((y_true == 1).sum())
    has_both_classes = n_benign > 0 and n_injection > 0

    # Fixed-label confusion matrix keeps column semantics stable across subsets.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Positive-class metrics (injection=1), kept for continuity with previous outputs.
    precision_pos = float(precision_score(y_true, y_pred, zero_division=0))
    recall_pos = float(recall_score(y_true, y_pred, zero_division=0))
    f1_pos = float(f1_score(y_true, y_pred, zero_division=0))

    # Balanced accuracy: mean of class recalls when both classes exist in this source.
    rec_benign = (tn / (tn + fp)) if (tn + fp) > 0 else None
    rec_injection = (tp / (tp + fn)) if (tp + fn) > 0 else None
    if rec_benign is not None and rec_injection is not None:
        balanced_acc = float((rec_benign + rec_injection) / 2.0)
    elif rec_benign is not None:
        balanced_acc = float(rec_benign)
    elif rec_injection is not None:
        balanced_acc = float(rec_injection)
    else:
        balanced_acc = None

    # Macro-F1 is only meaningful for two-class subsets.
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if has_both_classes else None

    rows.append({
        "source": src,
        "n": n,
        "n_benign": n_benign,
        "n_injection": n_injection,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_pos,
        "recall": recall_pos,
        "f1": f1_pos,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "macro_f1_defined": has_both_classes,
        "low_support": n < LOW_SUPPORT_N,
        "low_support_rule": f"n<{LOW_SUPPORT_N}",
    })

out_df = pd.DataFrame(rows).sort_values("n", ascending=False)
out_df.to_csv(OUT_DIR / "source_breakdown.csv", index=False)

print(out_df)
print(f"\nSaved: {OUT_DIR / 'source_breakdown.csv'}")
