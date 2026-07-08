#!/usr/bin/env python3
"""
B3 / Block 1 -- Cross-source and within-source hard-benign FPR analysis.
Runs on existing prediction files. No retraining.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/M2026206/khalid/Prompt-Injection-Detector-System")
SEEDS = [13, 42, 123, 2024, 7777]
N_BOOT = 5000
CI = 95
RNG_SEED = 12345
OUT_DIR = REPO / "outputs" / "resub_analysis"
HB_FILE = REPO / "data" / "pids_bench_v3" / "eval_subsets" / "hard_benign_test.csv"

BASELINE_CANDIDATES = {
    "DeBERTa-v3-FT": [
        "outputs/multi_seed_runs/deberta_baseline_clean",
        "outputs/multi_seed_runs/deberta",
    ],
    "DistilBERT": [
        "outputs/multi_seed_runs/distilbert",
        "outputs/multi_seed_runs/distilbert_cluster",
    ],
}

def resolve_baseline_dir(model):
    for rel in BASELINE_CANDIDATES[model]:
        d = REPO / rel
        if all((d / f"seed_{s}" / "hard_benign_predictions.csv").exists() for s in SEEDS):
            return d
    best, best_n = None, -1
    for rel in BASELINE_CANDIDATES[model]:
        d = REPO / rel
        n = sum((d / f"seed_{s}" / "hard_benign_predictions.csv").exists() for s in SEEDS)
        if n > best_n:
            best, best_n = d, n
    print(f"  [warn] {model}: no candidate dir has all 5 seeds; using {best} ({best_n}/5)")
    return best

def corpus_of(source):
    return source.replace("_ai_adjacent", "")

def slice_of(source):
    return "ai_adjacent" if source.endswith("_ai_adjacent") else "plain"

def load_seed(pred_path, hb):
    pred = pd.read_csv(pred_path)
    # the hard-benign file is the authority on source_type; drop any copy in pred
    pred = pred.drop(columns=[c for c in ("source", "source_type") if c in pred.columns])
    merged = pred.merge(hb[["text", "source", "source_type"]], on="text",
                        how="left", validate="one_to_one")
    missing = merged["source"].isna().sum()
    if missing:
        raise SystemExit(f"JOIN FAILED on {pred_path}: {missing} unmatched rows. Aborting.")
    merged["corpus"] = merged["source"].map(corpus_of)
    merged["slice"] = merged["source"].map(slice_of)
    return merged

def fpr(flags):
    return float(np.mean(flags)) if len(flags) else float("nan")

def bootstrap_ci(flags, rng):
    if len(flags) == 0:
        return (float("nan"), float("nan"))
    n = len(flags)
    boot = np.array([np.mean(flags[rng.integers(0, n, n)]) for _ in range(N_BOOT)])
    return (float(np.percentile(boot, (100-CI)/2)), float(np.percentile(boot, 100-(100-CI)/2)))

def pooled_rate_with_ci(per_seed_flags, rng):
    seed_rates = [fpr(f) for f in per_seed_flags]
    pooled = np.concatenate(per_seed_flags) if per_seed_flags else np.array([])
    lo, hi = bootstrap_ci(pooled, rng)
    return {
        "fpr_mean": float(np.mean(seed_rates)),
        "fpr_std": float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else 0.0,
        "ci_lo": lo, "ci_hi": hi,
        "n_examples": int(len(per_seed_flags[0])) if per_seed_flags else 0,
    }

def main():
    if not HB_FILE.exists():
        raise SystemExit(f"Missing hard-benign file: {HB_FILE}")
    hb = pd.read_csv(HB_FILE)
    if hb["text"].duplicated().any():
        raise SystemExit("hard_benign_test.csv has duplicate texts; join unsafe.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    across_rows, within_rows = [], []

    for model in ("DeBERTa-v3-FT", "DistilBERT"):
        bdir = resolve_baseline_dir(model)
        print(f"\n{'='*70}\n{model}  (baseline dir: {bdir})\n{'='*70}")
        seed_frames = []
        for s in SEEDS:
            p = bdir / f"seed_{s}" / "hard_benign_predictions.csv"
            if p.exists():
                seed_frames.append(load_seed(p, hb))
        if not seed_frames:
            print(f"  [skip] no prediction files for {model}")
            continue
        ext = [df[df["source_type"] == "real"] for df in seed_frames]

        print(f"\n  ACROSS-CORPUS hard-benign FPR (externally-sourced, tau=0.5)")
        print(f"  {'corpus':<12} {'n':>5}  {'FPR':>7}  {'std':>7}   {CI}% CI")
        for corpus in ("lmsys", "oasst1", "dolly"):
            flags = [d.loc[d["corpus"] == corpus, "flagged_default"].to_numpy() for d in ext]
            r = pooled_rate_with_ci(flags, rng)
            print(f"  {corpus:<12} {r['n_examples']:>5}  {r['fpr_mean']:>7.4f}  {r['fpr_std']:>7.4f}   [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
            across_rows.append({"model": model, "corpus": corpus, **r})
        flags_all = [d["flagged_default"].to_numpy() for d in ext]
        r_all = pooled_rate_with_ci(flags_all, rng)
        print(f"  {'ALL-ext':<12} {r_all['n_examples']:>5}  {r_all['fpr_mean']:>7.4f}  {r_all['fpr_std']:>7.4f}   [{r_all['ci_lo']:.4f}, {r_all['ci_hi']:.4f}]")
        across_rows.append({"model": model, "corpus": "ALL_external", **r_all})

        print(f"\n  WITHIN-CORPUS FPR: plain vs AI-adjacent (corpus held fixed)")
        print(f"  {'corpus':<12} {'slice':<12} {'n':>5}  {'FPR':>7}  {'std':>7}   {CI}% CI")
        for corpus in ("lmsys", "oasst1", "dolly"):
            for sl in ("plain", "ai_adjacent"):
                flags = [d.loc[(d["corpus"] == corpus) & (d["slice"] == sl), "flagged_default"].to_numpy() for d in ext]
                if all(len(f) == 0 for f in flags):
                    continue
                r = pooled_rate_with_ci(flags, rng)
                print(f"  {corpus:<12} {sl:<12} {r['n_examples']:>5}  {r['fpr_mean']:>7.4f}  {r['fpr_std']:>7.4f}   [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
                within_rows.append({"model": model, "corpus": corpus, "slice": sl, **r})

    pd.DataFrame(across_rows).to_csv(OUT_DIR / "b3_block1_across_corpus.csv", index=False)
    pd.DataFrame(within_rows).to_csv(OUT_DIR / "b3_block1_within_corpus.csv", index=False)
    print(f"\nWrote:\n  {OUT_DIR/'b3_block1_across_corpus.csv'}\n  {OUT_DIR/'b3_block1_within_corpus.csv'}")

if __name__ == "__main__":
    main()
