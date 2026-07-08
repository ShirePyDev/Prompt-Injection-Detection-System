#!/usr/bin/env python3
"""
Authoritative per-seed hard-benign FPR, recomputed from raw per-row predictions.

WHY THIS EXISTS
---------------
A provenance check found that the per-seed summary.json files agree with the raw
predictions on AGGREGATE hard-benign FPR (exact to 4 dp) but drift on the
real-origin / curated-origin SPLIT, because the summaries were written against an
earlier origin assignment than the currently released hard_benign_test.csv. The
aggregate (the over-defense total) is therefore solid; only the 872/600 bucketing
drifted.

To be safe for resubmission, every per-origin number is recomputed here from the
raw per-row predictions joined to the CURRENT released hard_benign_test.csv -- the
exact file a reviewer would use. The summary.json files are not trusted for the
split. This also fixes the drift as a byproduct.

It then prints the 5-seed mean +/- std for each arm beside the values the paper
currently reports (Table 7, Table 10), and flags any cell that has moved, so we
know exactly which paper text must be updated to match the released artifact.

This is the clean foundation the paired seed-level test (Reviewer 3 #5) is built on.

OUTPUT
  outputs/resub_analysis/hardbenign_fpr_recomputed_per_seed.csv
  prints per-seed table, 5-seed mean+/-std, and paper-vs-recomputed drift flags.

USAGE
  python scripts/recompute_hardbenign_fpr.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/M2026206/khalid/Prompt-Injection-Detector-System")
OUT_DIR = REPO / "outputs" / "resub_analysis"
HB_FILE = REPO / "data" / "pids_bench_v3" / "eval_subsets" / "hard_benign_test.csv"
SEEDS = ["13", "42", "123", "2024", "7777"]

ARMS = {
    "DeBERTa baseline":   "outputs/multi_seed_runs/deberta_baseline_clean",
    "DeBERTa augmented":  "outputs/multi_seed_runs/deberta_hardneg",
    "DistilBERT baseline":  "outputs/multi_seed_runs/distilbert_cluster",
    "DistilBERT augmented": "outputs/multi_seed_runs/distilbert_cluster_hardneg",
}

# values the paper currently prints (Table 7 + Table 10), to flag drift
PAPER = {
    "DeBERTa baseline":    {"agg": 0.3825, "real": 0.3144, "cur": 0.4813},
    "DeBERTa augmented":   {"agg": 0.1841, "real": 0.3101, "cur": 0.0010},
    "DistilBERT baseline": {"agg": 0.4702, "real": 0.4606, "cur": 0.4843},
    "DistilBERT augmented":{"agg": 0.2360, "real": 0.3899, "cur": 0.0123},
}


def recompute(pred_path: Path, key: pd.DataFrame):
    p = pd.read_csv(pred_path)
    p = p.drop(columns=[c for c in ("source", "source_type") if c in p.columns])
    m = p.merge(key, on="text", how="left", validate="one_to_one")
    if m["source_type"].isna().any():
        raise SystemExit(f"JOIN FAILED on {pred_path}")
    f = m["flagged_default"]
    return (float(f.mean()),
            float(f[m["source_type"] == "real"].mean()),
            float(f[m["source_type"] == "curated"].mean()))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hb = pd.read_csv(HB_FILE)
    key = hb[["text", "source_type"]]

    rows = []
    means = {}
    for arm, root in ARMS.items():
        print(f"\n######## {arm} ########")
        print(f"  {'seed':<6}{'agg':>9}{'real(ext)':>12}{'curated':>10}")
        per = {"agg": [], "real": [], "cur": []}
        for s in SEEDS:
            pred = REPO / root / f"seed_{s}" / "hard_benign_predictions.csv"
            if not pred.exists():
                print(f"  {s:<6} MISSING")
                continue
            agg, real, cur = recompute(pred, key)
            per["agg"].append(agg); per["real"].append(real); per["cur"].append(cur)
            print(f"  {s:<6}{agg:>9.4f}{real:>12.4f}{cur:>10.4f}")
            rows.append({"arm": arm, "seed": s, "agg": agg, "real": real, "cur": cur})
        means[arm] = {k: (np.mean(v), np.std(v, ddof=1)) for k, v in per.items() if v}

    # 5-seed mean +/- std, beside paper, with drift flags
    print(f"\n{'='*78}\n5-seed mean +/- std  vs  paper-reported  (drift flagged at >0.005)\n{'='*78}")
    print(f"  {'arm':<22}{'metric':<8}{'recomputed':<20}{'paper':<10}{'drift'}")
    for arm in ARMS:
        for k, lbl in (("agg", "agg"), ("real", "real/ext"), ("cur", "curated")):
            if arm not in means or k not in means[arm]:
                continue
            mu, sd = means[arm][k]
            pv = PAPER[arm][k]
            d = mu - pv
            flag = "  <-- UPDATE PAPER" if abs(d) > 0.005 else ""
            print(f"  {arm:<22}{lbl:<8}{mu:.4f} +/- {sd:.4f}    {pv:<10.4f}{d:+.4f}{flag}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "hardbenign_fpr_recomputed_per_seed.csv", index=False)
    print(f"\nWrote: {OUT_DIR/'hardbenign_fpr_recomputed_per_seed.csv'}")
    print("\nNote: aggregate matched the summaries exactly in the provenance check;")
    print("any real/curated drift below reflects the origin-assignment revision in")
    print("the released hard_benign_test.csv, and these recomputed values are the")
    print("authoritative ones to report.")


if __name__ == "__main__":
    main()
