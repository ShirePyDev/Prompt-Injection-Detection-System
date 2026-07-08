#!/usr/bin/env python3
"""
Source-prediction probe for the externally-sourced hard-benign set.

Reviewer 3's core worry: the detectors may key on corpus PROVENANCE (style,
vocabulary, formatting of one source vs another) rather than injection INTENT.
This probe separates two things the objection conflates:

  (1) Is source predictable from text?  -- almost certainly yes; the corpora are
      stylistically distinct. We report this honestly instead of denying it.

  (2) Does over-defense run THROUGH that source channel?  -- the decisive
      question. If the rows the detector over-flags are NOT the rows that are
      easy to source-identify, then source is predictable but is not what drives
      the over-flagging. Combined with the structure-tag result, that isolates
      instruction STRUCTURE -- not provenance -- as the trigger.

ANTI-CIRCULARITY
----------------
The probe predicts source WITHIN the benign label only (lmsys vs oasst1 vs
dolly, all benign). It never sees injection rows, so it cannot trivially re-learn
the injection/benign label. Predicting "injection-source vs benign-source" would
be circular; that is deliberately not done.

LENGTH CONTROL (this matters)
-----------------------------
The three benign corpora differ sharply in length (median chars: lmsys 372,
oasst1 162, dolly 78). A raw-text probe could score high just by reading LENGTH,
which would be a trivial, uninterpretable kind of "source predictability." So we
report a LENGTH-ONLY baseline beside the text probe. If the text probe barely
beats length, source predictability is largely a surface artifact -- which
defuses the confound rather than inflating it.

OUTPUT
  outputs/resub_analysis/source_probe_summary.csv
  prints: probe accuracy vs length-only baseline vs chance; and the correlation
  of source-identifiability with (a) over-flagging and (b) the structure tag.

USAGE
  python scripts/source_prediction_probe.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline

REPO = Path("/home/M2026206/khalid/Prompt-Injection-Detector-System")
OUT_DIR = REPO / "outputs" / "resub_analysis"
HB_FILE = REPO / "data" / "pids_bench_v3" / "eval_subsets" / "hard_benign_test.csv"
TAG_FILE = OUT_DIR / "structure_tagged_rows.csv"
PRED = REPO / "outputs/multi_seed_runs/deberta_baseline_clean/seed_42/hard_benign_predictions.csv"
SEED = 42
N_FOLDS = 5


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hb = pd.read_csv(HB_FILE)
    ext = hb[hb["source_type"] == "real"].copy()
    ext["corpus"] = ext["source"].str.replace("_ai_adjacent", "", regex=False)
    ext["text"] = ext["text"].astype(str)
    y = ext["corpus"].values
    X = ext["text"].values

    print(f"Externally-sourced benign rows: {len(ext)}")
    print("Per-corpus support:")
    print(ext["corpus"].value_counts().to_string())

    # majority-class (chance) baseline
    maj = ext["corpus"].value_counts(normalize=True).iloc[0]
    print(f"\nMajority-class accuracy (chance floor): {maj:.4f}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # ---- (1) TEXT probe: TF-IDF -> LR, predict corpus within benign ----
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  ngram_range=(1, 2), min_df=2, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   C=1.0, random_state=SEED)),
    ])
    text_pred = cross_val_predict(text_pipe, X, y, cv=skf)
    text_acc = accuracy_score(y, text_pred)
    text_bal = balanced_accuracy_score(y, text_pred)

    # also keep per-row "source-identifiability": predict-proba of the TRUE corpus
    # (how confidently the probe places each row in its real corpus)
    text_proba = cross_val_predict(text_pipe, X, y, cv=skf, method="predict_proba")
    classes = sorted(np.unique(y))
    col = {c: i for i, c in enumerate(classes)}
    true_conf = np.array([text_proba[i, col[y[i]]] for i in range(len(y))])

    # ---- (2) LENGTH-ONLY baseline: single feature = char length ----
    length_feat = np.array([len(t) for t in X]).reshape(-1, 1)
    len_clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 random_state=SEED)
    len_pred = cross_val_predict(len_clf, length_feat, y, cv=skf)
    len_acc = accuracy_score(y, len_pred)
    len_bal = balanced_accuracy_score(y, len_pred)

    print(f"\n{'='*64}\nSource predictability (within benign label, {N_FOLDS}-fold CV)\n{'='*64}")
    print(f"  chance (majority)     : acc={maj:.4f}")
    print(f"  length-only baseline  : acc={len_acc:.4f}  balanced_acc={len_bal:.4f}")
    print(f"  full text probe       : acc={text_acc:.4f}  balanced_acc={text_bal:.4f}")
    print(f"  text gain over length : {text_acc - len_acc:+.4f} acc "
          f"({text_bal - len_bal:+.4f} balanced)")

    # ---- (3) does source-identifiability predict OVER-FLAGGING? ----
    pred = pd.read_csv(PRED)
    pred = pred.drop(columns=[c for c in ("source", "source_type") if c in pred.columns])
    j = pred.merge(ext[["text", "corpus"]], on="text", how="inner")

    # per-row alignment by text key: map each row to its source-identifiability
    conf_by_text = dict(zip(ext["text"].values, true_conf))
    j["source_ident"] = j["text"].map(conf_by_text)
    j["flagged"] = j["flagged_default"].astype(int)

    # structure tag for the same rows
    tag = pd.read_csv(TAG_FILE)[["text", "struct"]]
    j = j.merge(tag, on="text", how="left")

    # correlations: is being over-flagged explained by source-identifiability
    # or by instruction-structure?
    from scipy.stats import pointbiserialr
    r_src, p_src = pointbiserialr(j["flagged"], j["source_ident"])
    r_str, p_str = pointbiserialr(j["flagged"], j["struct"])

    print(f"\n{'='*64}\nWhat predicts over-flagging?  (DeBERTa baseline, seed 42)\n{'='*64}")
    print(f"  flagged ~ source-identifiability : r={r_src:+.4f}  p={p_src:.2e}")
    print(f"  flagged ~ instruction-structure  : r={r_str:+.4f}  p={p_str:.2e}")
    print("\n  Reading: a near-zero / negative source-identifiability correlation,")
    print("  alongside a clear positive structure correlation, means the over-")
    print("  flagging tracks instruction STRUCTURE, not how source-identifiable a")
    print("  row is. Source being predictable (above) does not make it the channel.")

    # flagging rate by source-identifiability quartile, holding nothing else
    j["ident_q"] = pd.qcut(j["source_ident"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    print("\n  Over-flagging rate by source-identifiability quartile:")
    print(j.groupby("ident_q")["flagged"].mean().round(4).to_string())
    print("\n  Over-flagging rate by structure tag:")
    print(j.groupby("struct")["flagged"].mean().round(4).to_string())

    pd.DataFrame([{
        "n": len(ext), "chance_acc": maj,
        "length_only_acc": len_acc, "length_only_bal_acc": len_bal,
        "text_probe_acc": text_acc, "text_probe_bal_acc": text_bal,
        "text_gain_acc": text_acc - len_acc,
        "corr_flag_source_ident_r": r_src, "corr_flag_source_ident_p": p_src,
        "corr_flag_structure_r": r_str, "corr_flag_structure_p": p_str,
    }]).to_csv(OUT_DIR / "source_probe_summary.csv", index=False)
    print(f"\nWrote: {OUT_DIR/'source_probe_summary.csv'}")


if __name__ == "__main__":
    main()
