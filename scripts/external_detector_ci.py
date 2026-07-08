#!/usr/bin/env python3
"""
Sampling uncertainty for external and broad-safety detectors (Reviewer 3 #5).

These detectors are run once, with no across-seed variance. The honest
uncertainty is example-level: we bootstrap the hard-benign rows (and the 872
externally-sourced / 600 curated subsets) to put a 95% CI on each detector's FPR.

SOURCE VERIFICATION
-------------------
External outputs were scattered and one stray file was mislabeled, so each
detector here is pinned to the file+column whose aggregate FPR matches the
paper's Table 5, and the script ASSERTS that match before reporting. If any
source ever drifts, the assertion stops the run rather than emitting a wrong CI.

  ProtectAI tau=0.5   -> prob_cache/protectai_base  pred_default   (0.216)
  ProtectAI tau=0.002 -> prob_cache/protectai_base  pred_tuned     (0.322)
  DeBERTa-PI tau=0.5  -> prob_cache/deepset_base     pred_default  (0.804)
  DeBERTa-PI tau=0.986-> prob_cache/deepset_base     pred_tuned    (0.694)
  PromptGuard2        -> multiaxis/promptguard2_86m  external_pred (0.1515)
  Llama Guard 3-1B    -> multiaxis/llamaguard        external_pred (0.1800)

(The multiaxis/deberta_pi file is deliberately NOT used: it carries 0.216, i.e.
ProtectAI's numbers, not DeBERTa-PI's, and is discarded.)

Origin labels (real/curated) come from the released hard_benign_test.csv via an
exact-text join, so every external CI sits on the same 872/600 split as the
internal-model numbers.

OUTPUT
  outputs/resub_analysis/external_detector_fpr_ci.csv

USAGE
  python scripts/external_detector_ci.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/M2026206/khalid/Prompt-Injection-Detector-System")
OUT_DIR = REPO / "outputs" / "resub_analysis"
HB_FILE = REPO / "data" / "pids_bench_v3" / "eval_subsets" / "hard_benign_test.csv"
N_BOOT = 10000
RNG_SEED = 12345
TOL = 0.002   # max allowed gap between file FPR and paper Table 5 value

PC = "outputs/external_threshold_tuning/prob_cache"
MX = "outputs/external_eval/multiaxis"

# (display, file, column, paper_aggregate_FPR)
SPECS = [
    ("ProtectAI (tau=0.5)",    f"{PC}/protectai_base_hard_benign.csv", "pred_default", 0.216),
    ("ProtectAI (tau=0.002)",  f"{PC}/protectai_base_hard_benign.csv", "pred_tuned",   0.322),
    ("DeBERTa-PI (tau=0.5)",   f"{PC}/deepset_base_hard_benign.csv",   "pred_default", 0.804),
    ("DeBERTa-PI (tau=0.986)", f"{PC}/deepset_base_hard_benign.csv",   "pred_tuned",   0.694),
    ("PromptGuard2 (86M)",     f"{MX}/promptguard2_86m/hard_benign.csv", "external_pred", 0.1515),
    ("Llama Guard 3-1B",       f"{MX}/llamaguard/hard_benign.csv",       "external_pred", 0.1800),
]


def boot_ci(flags, rng):
    n = len(flags)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(flags.mean())
    b = np.array([flags[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hb = pd.read_csv(HB_FILE)
    key = hb[["text", "source_type"]]
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    print(f"{'detector':<24}{'subset':<20}{'FPR':>8}   95% CI")
    print("-" * 70)
    for name, rel, col, paper_fpr in SPECS:
        f = REPO / rel
        d = pd.read_csv(f)
        if col not in d.columns:
            raise SystemExit(f"{name}: column {col} not in {rel}")

        # ASSERT the source matches the paper before doing anything with it
        agg = d[col].mean()
        if abs(agg - paper_fpr) > TOL:
            raise SystemExit(
                f"{name}: aggregate FPR {agg:.4f} != paper {paper_fpr} "
                f"(file {rel}, col {col}). Source mismatch -- not reporting.")

        d = d[["text", col]].drop_duplicates("text")
        m = d.merge(key, on="text", how="left", validate="one_to_one")
        if m["source_type"].isna().any():
            raise SystemExit(f"{name}: {m['source_type'].isna().sum()} rows failed origin join")

        for sub_name, sub in (
            ("aggregate", m),
            ("externally-sourced", m[m["source_type"] == "real"]),
            ("curated", m[m["source_type"] == "curated"]),
        ):
            flags = sub[col].to_numpy().astype(float)
            point, lo, hi = boot_ci(flags, rng)
            print(f"{name:<24}{sub_name:<20}{point:>8.4f}   [{lo:.4f}, {hi:.4f}]  (n={len(flags)})")
            rows.append({"detector": name, "subset": sub_name, "n": len(flags),
                         "fpr": point, "ci_lo": lo, "ci_hi": hi})
        print()

    pd.DataFrame(rows).to_csv(OUT_DIR / "external_detector_fpr_ci.csv", index=False)
    print(f"Wrote: {OUT_DIR/'external_detector_fpr_ci.csv'}")
    print("\nAll sources verified against Table 5 before reporting.")
    print("CIs are example-level bootstrap -- the correct uncertainty for")
    print("single-run detectors with no training-seed variance.")


if __name__ == "__main__":
    main()
