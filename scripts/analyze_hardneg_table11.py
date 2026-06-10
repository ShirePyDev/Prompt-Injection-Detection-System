"""
scripts/analyze_hardneg_table11.py

Computes the numbers needed for Table 11 (Hard-Negative Augmentation).

DistilBERT section (5-seed multi-seed study)
─────────────────────────────────────────────
  Baseline      : 5 seeds — from all_runs.csv
  + hard-negative: 5 seeds — from all_runs.csv

DeBERTa-v3-FT section (canonical single-model + 5-seed augmented study)
────────────────────────────────────────────────────────────────────────
  Baseline      : CANONICAL model (outputs/deberta_v3/, March 30 Colab CUDA)
                  Single seed (42). No std — this is the paper-reported model.
                  Source: summary.json + hard_benign_fpr.json (per-source
                  breakdown computed post-hoc on 2026-05-20).

  Baseline@τ_HN : Canonical probability files re-thresholded at each
                  augmented seed's τ_HN. Variance comes from τ_HN variation
                  across augmented seeds, NOT from multiple baseline checkpoints.
                  Available without model inference: test F1/FPR, hard-benign
                  FPR (agg/real/curated), structural OOD F1.
                  NOT available: obf_recall, domain_OOD_F1_agg
                  (no saved per-example probs for those splits).

  + hard-negative: 5 seeds — from
                  outputs/multi_seed_runs/deberta_hardneg/seed_*/summary.json

Asymmetry note for Table 11 caption
────────────────────────────────────
  DistilBERT baseline numbers are 5-seed mean ± std from the multi-seed
  replication study.  DeBERTa-v3-FT baseline is the single canonical paper
  model (seed 42, τ=0.502); no std is reported because the multi-seed
  baseline checkpoints and probability files were not preserved.

Key outputs
───────────
  Printed table — paste into paper draft.
  CSV saved to outputs/multi_seed_runs/table11_mean_std.csv

Usage (from project root)
─────────────────────────
    python scripts/analyze_hardneg_table11.py

    # Include DeBERTa hardneg runs even if not all 5 seeds are complete:
    python scripts/analyze_hardneg_table11.py --partial-ok
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parent.parent
MULTI_SEED     = _ROOT / "outputs" / "multi_seed_runs"
MASTER_CSV     = MULTI_SEED / "all_runs.csv"
DEBERTA_HN_DIR = MULTI_SEED / "deberta_hardneg"
CANONICAL_DIR  = _ROOT / "outputs" / "deberta_v3"
OUT_CSV        = MULTI_SEED / "table11_mean_std.csv"

SEEDS = [13, 42, 123, 2024, 7777]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "N/A"
    if std is None or np.isnan(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _agg(values: list) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None and not np.isnan(float(v))]
    if not clean:
        return None, None
    arr  = np.array(clean, dtype=float)
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1)) if len(arr) > 1 else None
    return mean, std


def _wrap(v) -> list:
    return [v]


# ── Load DistilBERT rows from all_runs.csv ────────────────────────────────────

def _load_all_runs() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"Not found: {MASTER_CSV}\nRun train_multi_seed.py first.")
    return pd.read_csv(MASTER_CSV)


def _extract_from_csv(df: pd.DataFrame, model_name: str) -> dict[str, list]:
    sub = df[df["model_name"] == model_name].copy()
    metrics: dict[str, list] = {}
    for col in df.columns:
        if col in ("model_name", "seed"):
            continue
        metrics[col] = pd.to_numeric(sub[col], errors="coerce").tolist()
    metrics["_n_seeds"] = len(sub)
    metrics["_seeds"]   = sorted(sub["seed"].tolist())
    return metrics


# ── Load deberta_hardneg summaries ────────────────────────────────────────────

def _load_deberta_hardneg_summaries(partial_ok: bool) -> dict[str, list]:
    found = []
    for seed in SEEDS:
        p = DEBERTA_HN_DIR / f"seed_{seed}" / "summary.json"
        if not p.exists():
            if not partial_ok:
                print(f"  [warn] Missing: {p}  (use --partial-ok to proceed anyway)")
            continue
        try:
            found.append((seed, json.load(open(p))))
        except Exception as e:
            print(f"  [warn] Could not parse {p}: {e}")

    if not found:
        return {}

    def _get(d, *keys, default=None):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d

    metrics: dict[str, list] = {k: [] for k in [
        "tau", "IID_F1", "real_src_F1", "template_F1",
        "IID_recall", "IID_precision", "test_FPR", "ROC_AUC",
        "hb_FPR_agg", "hb_FPR_real", "hb_FPR_curated",
        "obf_recall", "domain_OOD_F1_agg",
        "struct_OOD_F1_agg", "struct_OOD_FPR_agg",
    ]}

    for seed, d in found:
        metrics["tau"].append(_get(d, "val_tuned_threshold"))
        test_t = _get(d, "test_results", "val_tuned_threshold") or {}
        metrics["IID_F1"].append(test_t.get("f1"))
        metrics["IID_recall"].append(test_t.get("recall"))
        metrics["IID_precision"].append(test_t.get("precision"))
        metrics["test_FPR"].append(test_t.get("fpr"))
        metrics["real_src_F1"].append(_get(d, "real_src_f1"))
        metrics["template_F1"].append(_get(d, "template_f1"))
        metrics["ROC_AUC"].append(_get(d, "roc_auc"))
        hb = _get(d, "hard_benign_test_fpr") or {}
        metrics["hb_FPR_agg"].append(hb.get("fpr_tuned"))
        metrics["hb_FPR_real"].append(hb.get("fpr_real_origin"))
        metrics["hb_FPR_curated"].append(hb.get("fpr_curated_origin"))
        metrics["obf_recall"].append(_get(d, "obfuscated_attacks", "overall_recall"))
        metrics["domain_OOD_F1_agg"].append(_get(d, "domain_ood", "aggregate", "f1"))
        struct_agg = _get(d, "structural_ood", "aggregate") or {}
        metrics["struct_OOD_F1_agg"].append(struct_agg.get("f1"))
        metrics["struct_OOD_FPR_agg"].append(struct_agg.get("fpr"))

    metrics["_n_seeds"] = len(found)
    metrics["_seeds"]   = [s for s, _ in found]
    return metrics


# ── DeBERTa canonical baseline (single model, outputs/deberta_v3/) ────────────

def _load_deberta_canonical_baseline() -> dict[str, list]:
    """Single-seed canonical DeBERTa-v3-FT baseline from outputs/deberta_v3/.

    Source: March 30 Colab CUDA run; model weights at models/deberta_v3/.
    hb_FPR_real/curated computed post-hoc on 2026-05-20 by joining
    hard_benign_predictions.csv to hard_benign_test.csv on source_type.
    """
    s_path  = CANONICAL_DIR / "summary.json"
    hb_path = CANONICAL_DIR / "hard_benign_fpr.json"

    if not s_path.exists():
        raise FileNotFoundError(
            f"Canonical summary not found: {s_path}\n"
            "Ensure outputs/deberta_v3/ (March 30 canonical run) is present."
        )

    s  = json.load(open(s_path))
    hb = json.load(open(hb_path)) if hb_path.exists() else {}

    tr  = s.get("test_results", {}).get("val_tuned_threshold", {})
    ood = s.get("ood_results", {})

    return {
        "tau":                _wrap(s.get("val_tuned_threshold")),
        "IID_F1":             _wrap(tr.get("f1")),
        "IID_precision":      _wrap(tr.get("precision")),
        "IID_recall":         _wrap(tr.get("recall")),
        "test_FPR":           _wrap(tr.get("fpr")),
        "ROC_AUC":            [None],
        "real_src_F1":        [None],
        "template_F1":        [None],
        "hb_FPR_agg":         _wrap(hb.get("fpr_tuned")),
        "hb_FPR_real":        _wrap(hb.get("fpr_real_origin")),
        "hb_FPR_curated":     _wrap(hb.get("fpr_curated_origin")),
        "obf_recall":         _wrap(s.get("obfuscated_attacks", {}).get("overall_recall")),
        "domain_OOD_F1_agg":  _wrap(ood.get("domain_ood",     {}).get("f1")),
        "struct_OOD_F1_agg":  _wrap(ood.get("structural_ood", {}).get("f1")),
        "struct_OOD_FPR_agg": _wrap(ood.get("structural_ood", {}).get("fpr")),
        "_n_seeds": 1,
        "_seeds":   [42],
    }


# ── DeBERTa baseline @ τ_HN (canonical probs, augmented τ_HN) ────────────────

def _compute_baseline_at_tau_hn(partial_ok: bool) -> dict[str, list]:
    """Apply each augmented seed's τ_HN to canonical probability files.

    Reads τ_HN from deberta_hardneg/seed_*/summary.json (val_tuned_threshold)
    and evaluates the fixed canonical model at that threshold via
    eval_canonical_at_tau.eval_at_tau().

    Variance across rows reflects τ_HN variation across augmented seeds;
    the underlying model (canonical) is the same for all rows.

    Available:   test F1/FPR, hard-benign FPR (agg/real/curated),
                 structural OOD F1/FPR.
    NOT available (no saved per-example probs):
                 obf_recall, domain_OOD_F1_agg  → shown as N/A.
    """
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from scripts.eval_canonical_at_tau import eval_at_tau  # noqa: PLC0415

    metrics: dict[str, list] = {k: [] for k in [
        "tau", "IID_F1", "IID_recall", "IID_precision", "test_FPR",
        "hb_FPR_agg", "hb_FPR_real", "hb_FPR_curated",
        "struct_OOD_F1_agg", "struct_OOD_FPR_agg",
        "obf_recall", "domain_OOD_F1_agg",
    ]}
    found_seeds: list[int] = []

    for seed in SEEDS:
        p = DEBERTA_HN_DIR / f"seed_{seed}" / "summary.json"
        if not p.exists():
            if not partial_ok:
                print(f"  [warn] No augmented summary for seed {seed} — skipping.")
            continue
        try:
            d      = json.load(open(p))
            tau_hn = d.get("val_tuned_threshold")
            if tau_hn is None:
                print(f"  [warn] No val_tuned_threshold in {p}")
                continue
        except Exception as e:
            print(f"  [warn] Could not parse {p}: {e}")
            continue

        try:
            m = eval_at_tau(float(tau_hn))
        except Exception as e:
            print(f"  [warn] eval_at_tau({tau_hn}) failed for seed {seed}: {e}")
            continue

        for k in metrics:
            metrics[k].append(m.get(k))
        found_seeds.append(seed)

    metrics["_n_seeds"] = len(found_seeds)
    metrics["_seeds"]   = found_seeds
    return metrics


# ── Table row definitions ─────────────────────────────────────────────────────

TABLE11_ROWS = [
    ("Test F1",                      "IID_F1"),
    ("Precision",                    "IID_precision"),
    ("Recall",                       "IID_recall"),
    ("Test FPR",                     "test_FPR"),
    ("Hard-benign FPR (aggregate)",  "hb_FPR_agg"),
    ("Hard-benign FPR (real-origin)","hb_FPR_real"),
    ("Hard-benign FPR (curated)",    "hb_FPR_curated"),
    ("Obfuscation Recall",           "obf_recall"),
    ("Domain OOD F1",                "domain_OOD_F1_agg"),
    ("Structural OOD F1",            "struct_OOD_F1_agg"),
    ("τ",                            "tau"),
]


def _print_block(label: str, metrics: dict, rows: list = TABLE11_ROWS) -> list[dict]:
    n     = metrics.get("_n_seeds", 0)
    seeds = metrics.get("_seeds", [])
    print(f"\n─── {label} ({n} seed{'s' if n != 1 else ''}: {seeds}) "
          f"{'─' * max(0, 52 - len(label))}")
    out_rows = []
    for row_label, key in rows:
        vals = metrics.get(key, [])
        mean, std = _agg(vals)
        print(f"  {row_label:<40}  {_fmt(mean, std)}")
        out_rows.append({"block": label, "metric": row_label, "key": key,
                         "n_seeds": n, "mean": mean, "std": std,
                         "formatted": _fmt(mean, std)})
    return out_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Table 11 numbers from multi-seed and canonical results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--partial-ok", action="store_true",
        help="Allow deberta_hardneg blocks even if fewer than 5 seeds are done.",
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Skip writing table11_mean_std.csv.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TABLE 11 — HARD-NEGATIVE AUGMENTATION RESULTS")
    print("=" * 70)

    df = _load_all_runs()
    all_rows: list[dict] = []

    # ═══════════════════════════════════════════════════════════════════════════
    # DISTILBERT — 5-seed multi-seed study
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("DistilBERT-base-uncased (5-seed multi-seed study)")
    print("═" * 70)

    dis_metrics = _extract_from_csv(df, "distilbert")
    all_rows += _print_block("DistilBERT baseline", dis_metrics)

    dhn_metrics = _extract_from_csv(df, "distilbert_hardneg")
    all_rows += _print_block("DistilBERT + hard-negative", dhn_metrics)

    print("\n  ── DistilBERT hard-negative effect (relative reduction) ──")
    for label, key in [
        ("hb_FPR_agg  (aggregate)", "hb_FPR_agg"),
        ("hb_FPR_real (real-origin)", "hb_FPR_real"),
        ("hb_FPR_curated (curated)", "hb_FPR_curated"),
    ]:
        bl_m, _ = _agg(dis_metrics.get(key, []))
        hn_m, _ = _agg(dhn_metrics.get(key, []))
        if bl_m and hn_m and bl_m > 0:
            rel = (bl_m - hn_m) / bl_m * 100
            print(f"  {label:<42}  {bl_m:.4f} → {hn_m:.4f}  ({rel:+.1f}%)")

    # ═══════════════════════════════════════════════════════════════════════════
    # DEBERTA-v3-FT — canonical single model + 5-seed augmented study
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("DeBERTa-v3-FT (canonical single model + 5-seed augmented study)")
    print("═" * 70)

    # ── Canonical baseline (single seed, no std) ──────────────────────────────
    try:
        can_metrics = _load_deberta_canonical_baseline()
        all_rows += _print_block("DeBERTa-v3-FT baseline (canonical)", can_metrics)
        print("  [src] outputs/deberta_v3/  — March 30 Colab CUDA run, seed=42")
        can_hb_agg = (can_metrics.get("hb_FPR_agg") or [None])[0]
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        can_hb_agg = None

    # ── Baseline @ τ_HN (canonical probs, augmented τ_HN per seed) ────────────
    print(f"\n  Computing baseline@τ_HN from canonical probs "
          f"(reading τ_HN from {DEBERTA_HN_DIR}/seed_*/summary.json) ...")
    bl_tau_hn = _compute_baseline_at_tau_hn(partial_ok=args.partial_ok)
    if bl_tau_hn.get("_n_seeds", 0) > 0:
        all_rows += _print_block(
            "DeBERTa-v3-FT baseline @ τ_HN", bl_tau_hn
        )
        print("  [note] Variance = τ_HN variation across augmented seeds; "
              "baseline model is fixed (canonical).")
        print("  [note] obf_recall, domain_OOD_F1_agg show N/A — "
              "no saved per-example probs for those splits.")
        print(f"  [seeds] {bl_tau_hn['_seeds']}")
    else:
        print("  No augmented summary.json files found yet — "
              "run deberta_hardneg seeds first.")

    # ── DeBERTa + hard-negative (5-seed) ─────────────────────────────────────
    print(f"\n  Looking for deberta_hardneg runs in {DEBERTA_HN_DIR} ...")
    debn_metrics = _load_deberta_hardneg_summaries(partial_ok=args.partial_ok)
    if debn_metrics:
        all_rows += _print_block("DeBERTa-v3-FT + hard-negative", debn_metrics)

        # Relative effect vs canonical baseline
        print("\n  ── DeBERTa hard-negative effect (vs canonical baseline) ──")
        for label, key in [
            ("hb_FPR_agg  (aggregate)", "hb_FPR_agg"),
            ("hb_FPR_real (real-origin)", "hb_FPR_real"),
            ("hb_FPR_curated (curated)", "hb_FPR_curated"),
        ]:
            # Canonical baseline is single value (wrapped in list)
            bl_m, _ = _agg(can_metrics.get(key, []) if 'can_metrics' in dir() else [])
            hn_m, _ = _agg(debn_metrics.get(key, []))
            if bl_m and hn_m and bl_m > 0:
                rel = (bl_m - hn_m) / bl_m * 100
                print(f"  {label:<42}  {bl_m:.4f} → {hn_m:.4f}  ({rel:+.1f}%)")
    else:
        print("\n  No deberta_hardneg runs found yet.")
        print(f"  Run: python scripts/train_multi_seed.py --models deberta_hardneg")

    # ── Caption note ──────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("TABLE 11 CAPTION NOTE (suggested):")
    print(
        "  DistilBERT baseline: 5-seed mean±std from multi-seed replication.\n"
        "  DeBERTa-v3-FT baseline: single canonical paper model (seed 42,\n"
        "  τ=0.502, March 30 Colab CUDA); no std reported because the\n"
        "  multi-seed baseline checkpoints were not preserved.\n"
        "  DeBERTa Baseline@τ_HN: canonical model re-thresholded at each\n"
        "  augmented seed's τ_HN; variance reflects τ_HN variation only.\n"
        "  obf_recall and domain_OOD_F1_agg in the Baseline@τ_HN column\n"
        "  require model inference on those splits and are not reported."
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if not args.no_csv and all_rows:
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(OUT_CSV, index=False)
        print(f"\nTable saved → {OUT_CSV}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
