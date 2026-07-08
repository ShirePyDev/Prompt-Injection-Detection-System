#!/usr/bin/env python3
"""
Paired seed-level analysis of the hard-negative augmentation effect.

Reviewer 3 (#5) asked, specifically, that the changes from baseline to augmented
-- e.g. DeBERTa externally-sourced 0.3144 -> 0.3101, DistilBERT 0.4606 -> 0.3899
-- "not be characterized as negligible or meaningful without a paired analysis."
This script provides that paired analysis.

Each architecture was trained five times (seeds 13/42/123/2024/7777) in both the
baseline and augmented condition. Because the seed is shared, baseline seed-13
pairs with augmented seed-13, and so on -- a matched design. We report the effect
three ways so the conclusion does not depend on any single test's assumptions
(important: with five seeds, a t-test's normality assumption cannot be checked
from the seeds alone):

  1. The five paired per-seed differences, shown in full (raw evidence).
  2. Paired t-test AND Wilcoxon signed-rank test. The Wilcoxon makes no normality
     assumption; when both agree, the distributional concern is moot.
  3. A bootstrap CI on the paired difference computed over EXAMPLES (resampling
     the 872 externally-sourced / 600 curated rows), which does not lean on the
     small number of seeds at all.

Honest reading:
  - A NON-significant externally-sourced change is the expected and correct
    result for DeBERTa; it confirms the paper's "within noise" statement with the
    rigor the reviewer requested. It is not a disappointment.
  - The curated change is expected to be large and clearly significant.
  - If DistilBERT's externally-sourced reduction is modest-but-significant while
    DeBERTa's is null, that architecture difference is reported as-is, not
    flattened into a single story.

Inputs are the authoritative per-seed numbers recomputed from raw predictions
(hardbenign_fpr_recomputed_per_seed.csv) plus the per-row prediction files for
the example-level bootstrap.

OUTPUT
  outputs/resub_analysis/paired_augmentation_test.csv
  prints per-seed differences and all three tests, per architecture x subset.

USAGE
  python scripts/paired_augmentation_test.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/home/M2026206/khalid/Prompt-Injection-Detector-System")
OUT_DIR = REPO / "outputs" / "resub_analysis"
HB_FILE = REPO / "data" / "pids_bench_v3" / "eval_subsets" / "hard_benign_test.csv"
PER_SEED_CSV = OUT_DIR / "hardbenign_fpr_recomputed_per_seed.csv"
SEEDS = ["13", "42", "123", "2024", "7777"]
N_BOOT = 10000
RNG_SEED = 12345

PAIRS = {
    "DeBERTa-v3-FT": ("DeBERTa baseline", "DeBERTa augmented",
                      "outputs/multi_seed_runs/deberta_baseline_clean",
                      "outputs/multi_seed_runs/deberta_hardneg"),
    "DistilBERT":    ("DistilBERT baseline", "DistilBERT augmented",
                      "outputs/multi_seed_runs/distilbert_cluster",
                      "outputs/multi_seed_runs/distilbert_cluster_hardneg"),
}


def per_row_flags(root: str, seed: str, key: pd.DataFrame, subset: str) -> np.ndarray:
    p = pd.read_csv(REPO / root / f"seed_{seed}" / "hard_benign_predictions.csv")
    p = p.drop(columns=[c for c in ("source", "source_type") if c in p.columns])
    m = p.merge(key, on="text", how="left", validate="one_to_one")
    if subset != "aggregate":
        m = m[m["source_type"] == subset]
    return m["flagged_default"].to_numpy(), m["text"].to_numpy()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per = pd.read_csv(PER_SEED_CSV)
    per["seed"] = per["seed"].astype(str)
    hb = pd.read_csv(HB_FILE)
    key = hb[["text", "source_type"]]
    rng = np.random.default_rng(RNG_SEED)

    rows_out = []
    for model, (base_arm, aug_arm, base_root, aug_root) in PAIRS.items():
        print(f"\n{'='*74}\n{model}\n{'='*74}")
        for subset, col in (("real", "real"), ("curated", "cur"), ("aggregate", "agg")):
            b = per[per["arm"] == base_arm].set_index("seed")[col]
            a = per[per["arm"] == aug_arm].set_index("seed")[col]
            seeds = [s for s in SEEDS if s in b.index and s in a.index]
            bvals = np.array([b[s] for s in seeds])
            avals = np.array([a[s] for s in seeds])
            diffs = avals - bvals          # augmented - baseline (negative = reduction)

            label = {"real": "externally-sourced", "curated": "curated",
                     "aggregate": "aggregate"}[subset]
            print(f"\n  --- {label} hard-benign FPR ---")
            print(f"    {'seed':<7}{'baseline':>10}{'augmented':>11}{'diff':>10}")
            for s, bv, av, dv in zip(seeds, bvals, avals, diffs):
                print(f"    {s:<7}{bv:>10.4f}{av:>11.4f}{dv:>+10.4f}")
            mean_d = diffs.mean()
            print(f"    {'mean':<7}{bvals.mean():>10.4f}{avals.mean():>11.4f}{mean_d:>+10.4f}")

            # (2) paired t and Wilcoxon
            t_stat, t_p = stats.ttest_rel(avals, bvals)
            try:
                w_stat, w_p = stats.wilcoxon(avals, bvals)
            except ValueError:
                w_stat, w_p = np.nan, np.nan  # e.g. all-zero differences
            print(f"    paired t : t={t_stat:+.3f}  p={t_p:.4f}")
            print(f"    Wilcoxon : W={w_stat}  p={w_p:.4f}"
                  if not np.isnan(w_p) else "    Wilcoxon : n/a (zero/degenerate diffs)")

            # (3) example-level bootstrap of the paired difference (seed-averaged)
            # build per-row flags averaged across seeds for base and aug, then
            # bootstrap over the shared row set
            base_rows, aug_rows, texts0 = [], [], None
            for s in seeds:
                bf, bt = per_row_flags(base_root, s, key, subset)
                af, at = per_row_flags(aug_root, s, key, subset)
                # align by text within this seed
                bser = pd.Series(bf, index=bt)
                aser = pd.Series(af, index=at)
                common = bser.index.intersection(aser.index)
                base_rows.append(bser.loc[common])
                aug_rows.append(aser.loc[common])
                texts0 = common if texts0 is None else texts0.intersection(common)
            # seed-mean per row
            base_mat = np.column_stack([r.loc[texts0].to_numpy() for r in base_rows])
            aug_mat = np.column_stack([r.loc[texts0].to_numpy() for r in aug_rows])
            row_diff = aug_mat.mean(1) - base_mat.mean(1)   # per-row mean diff
            n = len(row_diff)
            boot = np.array([row_diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"    bootstrap mean diff = {row_diff.mean():+.4f}  "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}]  (n={n} rows)")
            sig_boot = not (lo <= 0 <= hi)
            print(f"    -> example-level effect "
                  f"{'EXCLUDES 0 (significant)' if sig_boot else 'includes 0 (not significant)'}")

            rows_out.append({
                "model": model, "subset": label,
                "baseline_mean": bvals.mean(), "augmented_mean": avals.mean(),
                "mean_diff": mean_d,
                "t_stat": t_stat, "t_p": t_p, "wilcoxon_p": w_p,
                "boot_diff": row_diff.mean(), "boot_lo": lo, "boot_hi": hi,
                "boot_excludes_zero": sig_boot,
            })

    pd.DataFrame(rows_out).to_csv(OUT_DIR / "paired_augmentation_test.csv", index=False)
    print(f"\n\nWrote: {OUT_DIR/'paired_augmentation_test.csv'}")
    print("\nReading guide:")
    print("  curated: expect a large, clearly significant reduction (collapse).")
    print("  externally-sourced: a NULL result confirms the paper's 'within noise'")
    print("    claim with the requested rigor -- report it plainly, do not strain")
    print("    to make it look like movement.")
    print("  Report t and Wilcoxon together; the bootstrap CI is the assumption-")
    print("  free anchor and should be treated as primary for small seed counts.")


if __name__ == "__main__":
    main()
