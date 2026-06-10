"""
scripts/eval_canonical_at_tau.py

Applies a threshold τ to the canonical DeBERTa-v3-FT probability outputs
(March 30 Colab CUDA run, model saved to models/deberta_v3/) and returns
Table 11 metrics WITHOUT loading or running the model.

Purpose
-------
Produces the "Baseline DeBERTa @ τ_HN" column of Table 11.  For each
hard-negative augmented training run, the augmented model tunes its
threshold τ_HN on the augmented validation split.  Applying that same τ_HN
to the canonical baseline's probability scores isolates the threshold-shift
effect from the training effect — the key comparison in Table 11.

Available prediction files (all from March 30 canonical run)
-------------------------------------------------------------
  outputs/deberta_v3/test_predictions.csv         — test split
  outputs/deberta_v3/hard_benign_predictions.csv  — hard-benign stress set
  outputs/deberta_v3/structural_ood_probability_scores.csv  — structural OOD

NOT available without model inference (no saved per-example probs)
------------------------------------------------------------------
  domain_ood     → None returned for domain_OOD_F1_agg
  obfuscated     → None returned for obf_recall
  These splits would require loading models/deberta_v3/ and running inference
  once; that can be done on A100 if those Table 11 cells are needed.

Usage (CLI)
-----------
    python scripts/eval_canonical_at_tau.py --tau 0.65
    python scripts/eval_canonical_at_tau.py --tau 0.65 --json   # JSON output

Callable API
------------
    from scripts.eval_canonical_at_tau import eval_at_tau
    metrics = eval_at_tau(tau=0.65)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_DIR = _ROOT / "outputs" / "deberta_v3"
HB_TEST_CSV   = _ROOT / "data" / "pids_bench_v3" / "eval_subsets" / "hard_benign_test.csv"
STRUCT_OOD_CSV = _ROOT / "data" / "pids_bench_v3" / "ood" / "structural_ood.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sdiv(a: float, b: float) -> float:
    return a / b if b else 0.0


def _metrics_at(probs: np.ndarray, labels: np.ndarray, tau: float) -> dict:
    pred = (probs >= tau).astype(int)
    tp = int(((labels == 1) & (pred == 1)).sum())
    tn = int(((labels == 0) & (pred == 0)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    return {
        "f1":        float(f1_score(labels, pred, zero_division=0)),
        "precision": _sdiv(tp, tp + fp),
        "recall":    _sdiv(tp, tp + fn),
        "fpr":       _sdiv(fp, fp + tn),
    }


# ── Main evaluation function ──────────────────────────────────────────────────

def eval_at_tau(tau: float, canonical_dir: Path = CANONICAL_DIR) -> dict:
    """Apply tau to canonical prediction files and return Table 11 metrics.

    Parameters
    ----------
    tau : float
        Threshold value (τ_HN from an augmented training run).
    canonical_dir : Path
        Directory containing the canonical prediction CSV files.
        Defaults to outputs/deberta_v3/.

    Returns
    -------
    dict with keys matching METRIC_COLS used in train_multi_seed.py and
    analyze_hardneg_table11.py.  Values are float or None (when the metric
    cannot be computed from saved probability files).
    """
    canonical_dir = Path(canonical_dir)

    # ── Test split ────────────────────────────────────────────────────────────
    test_probs_path = canonical_dir / "test_predictions.csv"
    if not test_probs_path.exists():
        raise FileNotFoundError(f"Missing: {test_probs_path}")

    test_df  = pd.read_csv(test_probs_path)
    test_probs  = test_df["probability_score"].values.astype(float)
    test_labels = test_df["true_label"].values.astype(int)
    test_m = _metrics_at(test_probs, test_labels, tau)

    # Also compute source-type F1 if 'source' column is available
    real_src_f1 = template_f1 = None
    if "source" in test_df.columns:
        # "source" here is data-source tag, not source_type.
        # We need to join with the full test CSV for source_type.
        test_full_path = _ROOT / "data" / "pids_bench_v3" / "test.csv"
        if test_full_path.exists():
            tdf = pd.read_csv(test_full_path)
            if "source_type" in tdf.columns and len(tdf) == len(test_df):
                for st, col in [("real", "real_src_f1"), ("template", "template_f1")]:
                    mask = tdf["source_type"] == st
                    if mask.sum() > 0:
                        sub_p = test_probs[mask.values]
                        sub_l = test_labels[mask.values]
                        pred  = (sub_p >= tau).astype(int)
                        if col == "real_src_f1":
                            real_src_f1 = float(f1_score(sub_l, pred, zero_division=0))
                        else:
                            template_f1 = float(f1_score(sub_l, pred, zero_division=0))

    # ── Hard-benign ───────────────────────────────────────────────────────────
    hb_probs_path = canonical_dir / "hard_benign_predictions.csv"
    if not hb_probs_path.exists():
        raise FileNotFoundError(f"Missing: {hb_probs_path}")

    hb_df    = pd.read_csv(hb_probs_path)
    hb_test  = pd.read_csv(HB_TEST_CSV)

    assert len(hb_df) == len(hb_test), "hard_benign row count mismatch"
    assert (hb_df["text"] == hb_test["text"]).all(), "hard_benign row order mismatch"

    hb_probs  = hb_df["probability_score"].values.astype(float)
    hb_labels = hb_df["true_label"].values.astype(int)  # all 0
    hb_flagged = (hb_probs >= tau).astype(int)

    hb_fpr_agg = float(hb_flagged.mean())

    real_mask  = (hb_test["source_type"] == "real").values
    cur_mask   = (hb_test["source_type"] == "curated").values
    hb_fpr_real    = float(hb_flagged[real_mask].mean()) if real_mask.sum() > 0 else None
    hb_fpr_curated = float(hb_flagged[cur_mask].mean())  if cur_mask.sum()  > 0 else None

    # ── Structural OOD ────────────────────────────────────────────────────────
    struct_probs_path = canonical_dir / "structural_ood_probability_scores.csv"
    struct_f1_agg = struct_fpr_agg = None
    per_transform: dict[str, dict] = {}

    if struct_probs_path.exists():
        s_df    = pd.read_csv(struct_probs_path)
        s_probs = s_df["probability_score"].values.astype(float)
        s_labels = s_df["true_label"].values.astype(int)
        agg_m   = _metrics_at(s_probs, s_labels, tau)
        struct_f1_agg  = agg_m["f1"]
        struct_fpr_agg = agg_m["fpr"]

        if "transform_type" in s_df.columns:
            for tr_name in s_df["transform_type"].unique():
                mask = (s_df["transform_type"] == tr_name).values
                m    = _metrics_at(s_probs[mask], s_labels[mask], tau)
                per_transform[tr_name] = m

    # ── Build output dict (METRIC_COLS-compatible) ────────────────────────────
    result = {
        "tau":               round(tau, 6),
        "IID_F1":            test_m["f1"],
        "real_src_F1":       real_src_f1,
        "template_F1":       template_f1,
        "IID_recall":        test_m["recall"],
        "IID_precision":     test_m["precision"],
        "test_FPR":          test_m["fpr"],
        "ROC_AUC":           None,   # requires continuous scores + sklearn
        "hb_FPR_agg":        hb_fpr_agg,
        "hb_FPR_real":       hb_fpr_real,
        "hb_FPR_curated":    hb_fpr_curated,
        # Not available without model inference:
        "obf_recall":        None,
        "domain_OOD_F1_agg": None,
        # Structural OOD:
        "struct_OOD_F1_agg":  struct_f1_agg,
        "struct_OOD_FPR_agg": struct_fpr_agg,
        "json_wrap_FPR":          per_transform.get("json_wrap", {}).get("fpr"),
        "json_wrap_recall":       per_transform.get("json_wrap", {}).get("recall"),
        "prompt_dilute_FPR":      per_transform.get("prompt_dilute", {}).get("fpr"),
        "prompt_dilute_recall":   per_transform.get("prompt_dilute", {}).get("recall"),
        "prefix_mimic_FPR":       per_transform.get("instruction_prefix_mimic", {}).get("fpr"),
        "prefix_mimic_recall":    per_transform.get("instruction_prefix_mimic", {}).get("recall"),
    }
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply threshold τ to canonical DeBERTa-v3-FT probability files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tau",  type=float, required=True,
                        help="Threshold to apply (e.g. 0.65 for a hardneg seed's τ_HN).")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON (default: human-readable table).")
    parser.add_argument("--canonical-dir", default=str(CANONICAL_DIR),
                        help="Path to canonical prediction directory.")
    args = parser.parse_args()

    result = eval_at_tau(args.tau, Path(args.canonical_dir))

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"\nCanonical DeBERTa-v3-FT baseline @ τ = {args.tau}")
    print(f"Source: {args.canonical_dir}")
    print("-" * 55)
    labels = [
        ("Test F1",                    "IID_F1"),
        ("Precision",                  "IID_precision"),
        ("Recall",                     "IID_recall"),
        ("Test FPR",                   "test_FPR"),
        ("Hard-benign FPR (aggregate)","hb_FPR_agg"),
        ("Hard-benign FPR (real)",     "hb_FPR_real"),
        ("Hard-benign FPR (curated)",  "hb_FPR_curated"),
        ("Structural OOD F1",          "struct_OOD_F1_agg"),
        ("Structural OOD FPR",         "struct_OOD_FPR_agg"),
        ("Obfuscation Recall",         "obf_recall"),
        ("Domain OOD F1",              "domain_OOD_F1_agg"),
    ]
    for label, key in labels:
        v = result.get(key)
        s = f"{v:.6f}" if v is not None else "N/A (no saved probs)"
        print(f"  {label:<38}  {s}")
    print()


if __name__ == "__main__":
    main()
