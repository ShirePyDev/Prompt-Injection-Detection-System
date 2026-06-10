"""
Evaluate external prompt-injection classifiers under two threshold regimes:
  1. Default τ = 0.5
  2. Validation-tuned τ (selected on val split only, then fixed for all eval sets)

Models evaluated:
  protectai/deberta-v3-base-prompt-injection-v2
  deepset/deberta-v3-base-injection

Outputs (all written to outputs/external_threshold_tuning/):
  prob_cache/<model_slug>_<split>.csv  — per-row text, label, inj_prob
  external_threshold_tuning_thresholds.json
  external_threshold_tuning_summary.csv
  external_threshold_tuning_per_split_metrics.csv
  external_threshold_tuning_report.md

Usage (from project root):
  python scripts/eval_external_threshold_tuning.py
  python scripts/eval_external_threshold_tuning.py --force-rerun   # ignore prob cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "pids_bench_v3"
OUT_DIR = ROOT / "outputs" / "external_threshold_tuning"
CACHE_DIR = OUT_DIR / "prob_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------
MODELS: list[dict] = [
    {
        "slug": "protectai_base",
        "hf_id": "protectai/deberta-v3-base-prompt-injection-v2",
        "inj_class_idx": 1,        # verified: id2label = {0: 'SAFE', 1: 'INJECTION'}
        "inj_label_name": "INJECTION",
        "safe_label_name": "SAFE",
        "max_length": 512,
        "batch_size": 16,
    },
    {
        "slug": "deepset_base",
        "hf_id": "deepset/deberta-v3-base-injection",
        "inj_class_idx": 1,        # verified: id2label = {0: 'LEGIT', 1: 'INJECTION'}
        "inj_label_name": "INJECTION",
        "safe_label_name": "LEGIT",
        "max_length": 512,
        "batch_size": 16,
    },
]

# ---------------------------------------------------------------------------
# Split definitions
# kind: "binary" | "benign_only" | "injection_only"
# ---------------------------------------------------------------------------
SPLITS: list[dict] = [
    {
        "name": "val",
        "path": DATA / "val.csv",
        "kind": "binary",
        "role": "tune",          # used ONLY for threshold selection
    },
    {
        "name": "heldout_test",
        "path": DATA / "test.csv",
        "kind": "binary",
        "role": "eval",
    },
    {
        "name": "hard_benign",
        "path": DATA / "eval_subsets" / "hard_benign_test.csv",
        "kind": "benign_only",
        "role": "eval",
    },
    {
        "name": "obfuscated_attacks",
        "path": DATA / "eval_subsets" / "obfuscated_attacks.csv",
        "kind": "injection_only",
        "role": "eval",
    },
    {
        "name": "domain_ood",
        "path": DATA / "ood" / "domain_ood.csv",
        "kind": "binary",
        "role": "eval",
    },
    {
        "name": "structural_ood",
        "path": DATA / "ood" / "structural_ood.csv",
        "kind": "binary",
        "role": "eval",
    },
]

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Step 0: Verify label mapping — hard-stop if unexpected
# ---------------------------------------------------------------------------
def verify_label_mapping(model_cfg: dict, id2label: dict) -> None:
    inj_idx = model_cfg["inj_class_idx"]
    expected_inj  = model_cfg["inj_label_name"].upper()
    expected_safe = model_cfg["safe_label_name"].upper()
    actual_inj    = str(id2label.get(inj_idx, "")).upper()
    actual_safe   = str(id2label.get(1 - inj_idx, "")).upper()

    if expected_inj not in actual_inj:
        raise RuntimeError(
            f"[{model_cfg['hf_id']}] Label mapping mismatch!\n"
            f"  Expected class {inj_idx} = '{expected_inj}', got '{actual_inj}'\n"
            f"  Full id2label: {id2label}\n"
            "  Stopping — do not guess the injection class index."
        )
    print(f"  [OK] id2label verified: {id2label}")
    print(f"       injection class index = {inj_idx} ('{actual_inj}')")
    print(f"       benign class index    = {1 - inj_idx} ('{actual_safe}')")


# ---------------------------------------------------------------------------
# Inference: return injection probability for each row
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(
    model_cfg: dict,
    texts: list[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> np.ndarray:
    inj_idx  = model_cfg["inj_class_idx"]
    max_len  = model_cfg["max_length"]
    bs       = model_cfg["batch_size"]
    probs: list[float] = []

    for i in tqdm(range(0, len(texts), bs), desc="  batches", leave=False):
        batch = texts[i : i + bs]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits                    # (B, num_labels)
        p = torch.softmax(logits, dim=-1)[:, inj_idx]  # injection probability
        probs.extend(p.cpu().tolist())
        if device.type == "mps":
            torch.mps.empty_cache()

    return np.array(probs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Load or run inference, save prob cache CSV
# ---------------------------------------------------------------------------
def get_probs(
    model_cfg: dict,
    split: dict,
    df: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    force_rerun: bool,
) -> np.ndarray:
    cache_path = CACHE_DIR / f"{model_cfg['slug']}_{split['name']}.csv"

    if cache_path.exists() and not force_rerun:
        print(f"    [cache] Loading {cache_path.name}")
        cached = pd.read_csv(cache_path)
        if len(cached) != len(df):
            print(f"    [warn] Cache row mismatch ({len(cached)} vs {len(df)}). Re-running.")
        else:
            return cached["inj_prob"].to_numpy(dtype=np.float32)

    print(f"    [run ] Running inference on {split['name']} ({len(df)} rows)...")
    texts  = df["text"].astype(str).tolist()
    t0     = time.time()
    probs  = run_inference(model_cfg, texts, model, tokenizer, device)
    elapsed = time.time() - t0
    print(f"    [done] {elapsed:.1f}s  ({1000*elapsed/len(texts):.1f} ms/row)")

    # Build cache dataframe — include row id if available
    id_col = next((c for c in ("parent_seed_id", "id", "idx") if c in df.columns), None)
    cache_df = pd.DataFrame({
        **({"row_id": df[id_col].values} if id_col else {}),
        "text": df["text"].astype(str).values,
        "label": df["label"].astype(int).values,
        "inj_prob": probs,
        "pred_default": (probs >= 0.5).astype(int),
    })
    cache_df.to_csv(cache_path, index=False)
    print(f"    [save] {cache_path.relative_to(ROOT)}")
    return probs


# ---------------------------------------------------------------------------
# Threshold tuning on validation only
# ---------------------------------------------------------------------------
def tune_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Sweep τ from 0.0 to 1.0 in steps of 0.002.
    Maximize binary F1 (pos_label=1).
    Ties broken by highest τ (lower FPR preference).
    Returns (best_tau, best_f1).
    """
    thresholds = np.arange(0.0, 1.001, 0.002)
    best_f1  = -1.0
    best_tau = 0.5

    for tau in thresholds:
        preds = (probs >= tau).astype(int)
        f1 = float(f1_score(labels, preds, pos_label=1, zero_division=0))
        if f1 > best_f1 or (f1 == best_f1 and tau > best_tau):
            best_f1  = f1
            best_tau = float(tau)

    return round(best_tau, 4), round(best_f1, 6)


# ---------------------------------------------------------------------------
# Metrics for a single split at a given threshold
# ---------------------------------------------------------------------------
def compute_metrics(
    kind: str,
    labels: np.ndarray,
    probs: np.ndarray,
    tau: float,
) -> dict:
    preds  = (probs >= tau).astype(int)
    labels = labels.astype(int)
    n      = len(labels)
    m: dict = {"n": n, "kind": kind, "tau": round(tau, 4)}

    if kind == "benign_only":
        fp  = int((preds == 1).sum())
        fpr = fp / n if n else 0.0
        m["fpr"] = round(fpr, 6)
        m["n_flagged"] = fp
        return m

    if kind == "injection_only":
        tp     = int((preds == 1).sum())
        recall = tp / n if n else 0.0
        m["recall"] = round(recall, 6)
        m["fnr"]    = round(1.0 - recall, 6)
        m["n_detected"] = tp
        return m

    # binary
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())

    prec   = float(precision_score(labels, preds, zero_division=0))
    rec    = float(recall_score(labels, preds, zero_division=0))
    f1     = float(f1_score(labels, preds, zero_division=0))
    fpr    = fp / (fp + tn) if (fp + tn) else 0.0
    fnr    = fn / (fn + tp) if (fn + tp) else 0.0

    m["precision"]  = round(prec, 6)
    m["recall"]     = round(rec, 6)
    m["f1"]         = round(f1, 6)
    m["fpr"]        = round(fpr, 6)
    m["fnr"]        = round(fnr, 6)
    m["tp"], m["tn"], m["fp"], m["fn"] = tp, tn, fp, fn

    # ROC-AUC: only meaningful when both classes are present
    if len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, probs))
        m["roc_auc"] = round(auc, 6)
    else:
        m["roc_auc"] = None

    return m


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(force_rerun: bool = False) -> None:
    device = pick_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load all split dataframes and verify columns
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 1 — Loading and verifying split CSVs")
    print("=" * 65)
    split_dfs: dict[str, pd.DataFrame] = {}
    for s in SPLITS:
        path = s["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        df = pd.read_csv(path)
        if "text" not in df.columns:
            raise ValueError(f"Split '{s['name']}' has no 'text' column. Columns: {df.columns.tolist()}")
        if "label" not in df.columns:
            raise ValueError(f"Split '{s['name']}' has no 'label' column. Columns: {df.columns.tolist()}")
        label_counts = df["label"].value_counts().to_dict()
        print(f"  {s['name']:20} rows={len(df):5}  cols={list(df.columns)}")
        print(f"  {'':20} label_counts={label_counts}  kind={s['kind']}")
        split_dfs[s["name"]] = df

    # Validate kind vs actual labels
    for s in SPLITS:
        df     = split_dfs[s["name"]]
        labels = df["label"].astype(int).values
        uniq   = set(labels)
        if s["kind"] == "benign_only" and uniq != {0}:
            raise ValueError(f"Split '{s['name']}' declared benign_only but has labels: {uniq}")
        if s["kind"] == "injection_only" and uniq != {1}:
            raise ValueError(f"Split '{s['name']}' declared injection_only but has labels: {uniq}")
        if s["kind"] == "binary" and not (0 in uniq and 1 in uniq):
            raise ValueError(f"Split '{s['name']}' declared binary but labels: {uniq}")

    # ------------------------------------------------------------------
    # Per-model evaluation
    # ------------------------------------------------------------------
    all_thresholds: dict[str, dict] = {}
    all_per_split:  list[dict]      = []

    for model_cfg in MODELS:
        slug  = model_cfg["slug"]
        hf_id = model_cfg["hf_id"]

        print(f"\n{'='*65}")
        print(f"MODEL: {hf_id}  (slug={slug})")
        print("=" * 65)

        # Load model + tokenizer
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model_hf  = AutoModelForSequenceClassification.from_pretrained(hf_id)
        id2label  = {int(k): v for k, v in model_hf.config.id2label.items()}

        # Step 0: hard-stop label check
        print("\n  Verifying label mapping...")
        verify_label_mapping(model_cfg, id2label)

        model_hf.to(device).eval()

        # ----------------------------------------------------------
        # Step 2: Run inference on every split, cache probabilities
        # ----------------------------------------------------------
        split_probs: dict[str, np.ndarray] = {}
        for s in SPLITS:
            df = split_dfs[s["name"]]
            print(f"\n  Split: {s['name']}")
            probs = get_probs(model_cfg, s, df, model_hf, tokenizer, device, force_rerun)
            split_probs[s["name"]] = probs

        # Update cache CSVs with tuned predictions (done after tuning below)

        # ----------------------------------------------------------
        # Step 3: Tune threshold on VALIDATION ONLY
        # ----------------------------------------------------------
        print(f"\n  Threshold tuning on validation split ({len(split_dfs['val'])} rows)...")
        val_labels = split_dfs["val"]["label"].astype(int).values
        val_probs  = split_probs["val"]

        # Default threshold on validation
        val_preds_05 = (val_probs >= 0.5).astype(int)
        val_f1_05    = float(f1_score(val_labels, val_preds_05, pos_label=1, zero_division=0))

        # Tuned threshold
        best_tau, best_val_f1 = tune_threshold(val_probs, val_labels)

        print(f"    Default τ=0.500 → val F1={val_f1_05:.4f}")
        print(f"    Tuned   τ={best_tau:.4f} → val F1={best_val_f1:.4f}")

        all_thresholds[slug] = {
            "model_hf_id":       hf_id,
            "slug":              slug,
            "verified_id2label": id2label,
            "inj_class_idx":     model_cfg["inj_class_idx"],
            "default_tau":       0.5,
            "tuned_tau":         best_tau,
            "val_f1_at_default": round(val_f1_05, 6),
            "val_f1_at_tuned":   round(best_val_f1, 6),
            "val_n":             int(len(val_labels)),
            "tuning_note": (
                "τ swept from 0.000 to 1.000 in steps of 0.002. "
                "Best F1 on validation split selected. "
                "Ties broken by highest τ. "
                "τ fixed — NOT re-tuned on any evaluation split."
            ),
        }

        # Update cache CSVs with tuned predictions
        for s in SPLITS:
            cache_path = CACHE_DIR / f"{slug}_{s['name']}.csv"
            if cache_path.exists():
                cached = pd.read_csv(cache_path)
                cached["pred_tuned"] = (split_probs[s["name"]] >= best_tau).astype(int)
                cached.to_csv(cache_path, index=False)

        # ----------------------------------------------------------
        # Step 4 & 5: Apply both thresholds to eval splits, compute metrics
        # ----------------------------------------------------------
        print(f"\n  Computing metrics on all evaluation splits...")
        for s in SPLITS:
            df     = split_dfs[s["name"]]
            labels = df["label"].astype(int).values
            probs  = split_probs[s["name"]]

            for tau_label, tau in [("default_0.5", 0.5), (f"tuned_{best_tau}", best_tau)]:
                m = compute_metrics(s["kind"], labels, probs, tau)
                row = {
                    "model_slug":  slug,
                    "model_hf_id": hf_id,
                    "split":       s["name"],
                    "split_kind":  s["kind"],
                    "split_role":  s["role"],
                    "threshold_label": tau_label,
                    **m,
                }
                all_per_split.append(row)
                tag = "τ=0.500" if tau_label == "default_0.5" else f"τ={best_tau:.4f}(tuned)"
                if s["kind"] == "binary":
                    print(f"    {s['name']:20} {tag}  F1={m.get('f1','--'):.4f}  "
                          f"rec={m.get('recall','--'):.4f}  FPR={m.get('fpr','--'):.4f}  "
                          f"AUC={m.get('roc_auc') or '--'}")
                elif s["kind"] == "benign_only":
                    print(f"    {s['name']:20} {tag}  FPR={m.get('fpr','--'):.4f}")
                else:
                    print(f"    {s['name']:20} {tag}  recall={m.get('recall','--'):.4f}")

        # Free GPU memory before next model
        del model_hf
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # Step 6: Save outputs
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("STEP 6 — Saving outputs")
    print("=" * 65)

    # Thresholds JSON
    thresh_path = OUT_DIR / "external_threshold_tuning_thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(all_thresholds, f, indent=2, default=str)
    print(f"  Wrote {thresh_path.relative_to(ROOT)}")

    # Per-split metrics CSV
    per_split_df = pd.DataFrame(all_per_split)
    per_split_path = OUT_DIR / "external_threshold_tuning_per_split_metrics.csv"
    per_split_df.to_csv(per_split_path, index=False)
    print(f"  Wrote {per_split_path.relative_to(ROOT)}")

    # Summary CSV: one row per (model, threshold_label) with key metrics
    summary_rows = []
    for slug in [m["slug"] for m in MODELS]:
        thresh_info = all_thresholds[slug]
        for tau_label in ["default_0.5", f"tuned_{thresh_info['tuned_tau']}"]:
            sub = per_split_df[
                (per_split_df["model_slug"] == slug) &
                (per_split_df["threshold_label"] == tau_label)
            ]
            def _get(split_name, col, default=None):
                rows = sub[sub["split"] == split_name]
                if rows.empty or col not in rows.columns:
                    return default
                v = rows.iloc[0].get(col, default)
                return None if pd.isna(v) else v

            summary_rows.append({
                "model_slug":            slug,
                "model_hf_id":           thresh_info["model_hf_id"],
                "threshold_label":       tau_label,
                "tau":                   0.5 if "default" in tau_label else thresh_info["tuned_tau"],
                "val_f1_at_tau":         thresh_info["val_f1_at_default"] if "default" in tau_label else thresh_info["val_f1_at_tuned"],
                "test_f1":               _get("heldout_test", "f1"),
                "test_recall":           _get("heldout_test", "recall"),
                "test_fpr":              _get("heldout_test", "fpr"),
                "test_roc_auc":          _get("heldout_test", "roc_auc"),
                "hard_benign_fpr":       _get("hard_benign", "fpr"),
                "obfuscated_recall":     _get("obfuscated_attacks", "recall"),
                "domain_ood_f1":         _get("domain_ood", "f1"),
                "structural_ood_f1":     _get("structural_ood", "f1"),
                "structural_ood_fpr":    _get("structural_ood", "fpr"),
                "structural_ood_roc_auc": _get("structural_ood", "roc_auc"),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "external_threshold_tuning_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Wrote {summary_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # Step 7: Write markdown report
    # ------------------------------------------------------------------
    report_lines: list[str] = [
        "# External Classifier Threshold Tuning Report",
        "",
        "## Setup",
        "",
        "Two external models evaluated at default (τ=0.5) and validation-tuned τ.",
        "Threshold was selected on the PIDS-Bench **validation split only**, then **fixed** "
        "for all evaluation sets. No re-tuning was done per evaluation split.",
        "",
        "| Item | Value |",
        "| --- | --- |",
        "| Tuning split | val (frozen PIDS-Bench validation split) |",
        "| Tuning criterion | max binary F1 (pos_label=1 = injection) |",
        "| Threshold sweep | 0.000 → 1.000 step 0.002 |",
        "| Tie-break | highest τ (fewer false positives) |",
        "",
        "---",
        "",
        "## Models and label mappings",
        "",
    ]

    for slug, info in all_thresholds.items():
        report_lines += [
            f"### {info['model_hf_id']}",
            "",
            f"| | |",
            f"| --- | --- |",
            f"| HF model ID | `{info['model_hf_id']}` |",
            f"| Verified id2label | `{info['verified_id2label']}` |",
            f"| Injection class index | {info['inj_class_idx']} |",
            f"| Default τ | {info['default_tau']} |",
            f"| Validation F1 at τ=0.500 | {info['val_f1_at_default']} |",
            f"| **Tuned τ** | **{info['tuned_tau']}** |",
            f"| Validation F1 at tuned τ | {info['val_f1_at_tuned']} |",
            f"| Val split n | {info['val_n']} |",
            "",
        ]

    report_lines += [
        "---",
        "",
        "## Results summary",
        "",
        _md_table(summary_df.round(4)),
        "",
        "---",
        "",
        "## Per-split metrics",
        "",
    ]

    for slug in [m["slug"] for m in MODELS]:
        hf_id = all_thresholds[slug]["model_hf_id"]
        report_lines += [f"### {hf_id}", ""]
        sub = per_split_df[per_split_df["model_slug"] == slug].copy()
        # Select readable columns
        display_cols = [c for c in [
            "split", "split_kind", "threshold_label", "n", "tau",
            "f1", "precision", "recall", "fpr", "fnr", "roc_auc",
            "n_flagged", "n_detected",
        ] if c in sub.columns]
        display = sub[display_cols].sort_values(["split", "threshold_label"])
        report_lines += [_md_table(display), ""]

    report_lines += [
        "---",
        "",
        "## Notes and assumptions",
        "",
        "- Injection probability = softmax(logits)[:, injection_class_idx]",
        "- Label mapping verified programmatically via `model.config.id2label` before any "
        "inference. Script would have stopped with an error if mapping was ambiguous.",
        "- ROC-AUC is computed from raw probability scores. It is reported only for binary "
        "splits (both classes present). It is threshold-independent and identical for both "
        "τ rows of the same model × split.",
        "- Obfuscated-attacks split is injection-only (all label=1): only recall is meaningful.",
        "- Hard-benign split is benign-only (all label=0): only FPR is meaningful.",
        "- **The validation split was used solely for threshold selection and is not an "
        "evaluation result.** Its metrics are reported to show how threshold choice was made.",
        "- Original benchmark CSV files were not modified.",
        "",
        "_Generated by `scripts/eval_external_threshold_tuning.py`_",
    ]

    report_path = OUT_DIR / "external_threshold_tuning_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Wrote {report_path.relative_to(ROOT)}")

    print(f"\n{'='*65}")
    print("DONE — Summary:")
    print("=" * 65)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore cached probability CSVs and rerun inference",
    )
    args = ap.parse_args()
    main(force_rerun=args.force_rerun)
