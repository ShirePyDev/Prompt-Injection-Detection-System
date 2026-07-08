#!/usr/bin/env python3
"""
fixed_tau_hb_fpr.py

Apples-to-apples hard-benign FPR at ONE fixed threshold, applied identically to
every run. Reads probability_score directly from each seed's
hard_benign_predictions.csv and recomputes the flag at the SAME tau for all
runs, so no comparison depends on a per-run tuned threshold.

For each (pool-config), reports across seeds:
  - externally-sourced (source_type == 'real') FPR @ tau
  - curated FPR @ tau
  - aggregate FPR @ tau
as mean +/- std, plus per-seed values.

Every hard-benign row has true_label == 0, so any flag (score >= tau) is a
false positive. The script asserts this invariant.

USAGE
  python scripts/fixed_tau_hb_fpr.py            # tau = 0.5
  python scripts/fixed_tau_hb_fpr.py --tau 0.5
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/M2026206/khalid/Prompt-Injection-Detector-System")
SEEDS = [13, 42, 123, 2024, 7777]
RUNS_ROOT = REPO / "outputs" / "multi_seed_runs"

# The configs to compare. dir_name -> label.
CONFIGS = {
    "deberta_baseline_clean": "pool_0_baseline",
    "deberta_hardneg_pool100": "pool_100",
    "deberta_hardneg_pool200": "pool_200",
    "deberta_hardneg_pool300": "pool_300",
    "deberta_hardneg":         "pool_419_anchor",
}


def fpr_at_tau(df: pd.DataFrame, tau: float, subset: str | None) -> float | None:
    """FPR at fixed tau for a subset. subset in {'real','curated',None(all)}."""
    d = df
    if subset is not None:
        d = d[d["source_type"] == subset]
    if len(d) == 0:
        return None
    # Every row is benign (true_label==0); a flag is a false positive.
    assert (d["true_label"] == 0).all(), "Non-benign row in hard-benign set!"
    flags = (d["probability_score"].to_numpy() >= tau).astype(float)
    return float(flags.mean())


def summarize(per_seed: list[float]) -> str:
    vals = [v for v in per_seed if v is not None]
    if not vals:
        return "n/a"
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return f"{m:.4f} +/- {s:.4f}  (n_seeds={len(vals)})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()
    tau = args.tau

    print(f"Fixed threshold tau = {tau} applied identically to every run.\n")
    print(f"{'config':<20} {'externally-sourced':<28} {'curated':<28} {'aggregate':<28}")
    print("-" * 104)

    for dir_name, label in CONFIGS.items():
        real_ps, cur_ps, agg_ps = [], [], []
        per_seed_real = {}
        for s in SEEDS:
            f = RUNS_ROOT / dir_name / f"seed_{s}" / "hard_benign_predictions.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            if "source_type" not in df.columns or "probability_score" not in df.columns:
                print(f"  [skip] {dir_name} seed {s}: missing columns")
                continue
            r = fpr_at_tau(df, tau, "real")
            c = fpr_at_tau(df, tau, "curated")
            a = fpr_at_tau(df, tau, None)
            real_ps.append(r); cur_ps.append(c); agg_ps.append(a)
            per_seed_real[s] = r

        print(f"{label:<20} {summarize(real_ps):<28} {summarize(cur_ps):<28} {summarize(agg_ps):<28}")
        # per-seed externally-sourced, for transparency
        if per_seed_real:
            detail = "  ".join(f"s{k}={v:.4f}" for k, v in per_seed_real.items() if v is not None)
            print(f"{'':<20} ext per-seed: {detail}")
    print()
    print("Note: externally-sourced = source_type 'real'; every value at the SAME tau.")


if __name__ == "__main__":
    main()
