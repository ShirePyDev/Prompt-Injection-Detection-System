#!/usr/bin/env python3
"""
make_second_annotator_sample.py

Prepares a second-annotator spot-check of the hard-benign labels, so that an
inter-annotator agreement statistic (Cohen's kappa) can be reported alongside the
first-annotator audit.

It draws a stratified sample from the 200 already-audited rows in
hard_benign_audit_sample.csv and writes two files:

  1. second_annotator_blind.csv  -- sent to the annotator. Contains only an item
     id and the prompt text, plus two empty columns for their judgments. The first
     annotator's labels are NOT included, so the second annotator labels blind.

  2. second_annotator_key.csv    -- kept on the author side. Contains the same item
     ids with the first annotator's labels, for computing kappa once the blind
     sheet is returned.

The sample is stratified by source_type (externally-sourced vs curated) and by the
first annotator's attack-adjacent flag, so the spot-check spans the same range of
inputs the full audit covered. Rows are shuffled before writing so their order
carries no signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

AUDIT_FILE = "data/pids_bench_v3/eval_subsets/hard_benign_audit_sample.csv"
RANDOM_SEED = 20260630


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--n", type=int, default=50, help="sample size")
    parser.add_argument("--out-dir", default="outputs/second_annotator")
    args = parser.parse_args()

    root = Path(args.repo_root)
    audit = pd.read_csv(root / AUDIT_FILE)

    # Stratify by source_type and by the first annotator's attack-adjacent flag,
    # so the spot-check covers curated and externally-sourced rows and both
    # ordinary-benign and security-adjacent cases in proportion.
    audit = audit.copy()
    audit["_stratum"] = (
        audit["source_type"].astype(str)
        + "|adj=" + audit["audit_is_attack_adjacent"].astype(str)
    )

    n = min(args.n, len(audit))
    # Proportional allocation across strata, with at least one row per stratum
    # that exists, drawn reproducibly.
    frac = n / len(audit)
    picks = []
    for _, grp in audit.groupby("_stratum"):
        k = max(1, round(len(grp) * frac))
        picks.append(grp.sample(min(k, len(grp)), random_state=RANDOM_SEED))
    sample = pd.concat(picks)

    # If rounding under/over-shot, trim or top up to exactly n.
    if len(sample) > n:
        sample = sample.sample(n, random_state=RANDOM_SEED)
    elif len(sample) < n:
        remaining = audit.drop(index=sample.index)
        topup = remaining.sample(min(n - len(sample), len(remaining)),
                                 random_state=RANDOM_SEED)
        sample = pd.concat([sample, topup])

    # Shuffle so order carries no signal, and assign clean item ids.
    sample = sample.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    sample.insert(0, "item_id", [f"hb_{i:03d}" for i in range(len(sample))])

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Blind sheet for the annotator: id, text, and two empty judgment columns.
    blind = sample[["item_id", "text"]].copy()
    blind["is_benign"] = ""            # 1 = benign, 0 = not clearly benign
    blind["is_security_adjacent"] = "" # 1 = discusses security/AI topics, else 0
    blind["notes"] = ""
    blind_path = out_dir / "second_annotator_blind.csv"
    blind.to_csv(blind_path, index=False)

    # 2. Held-back key with the first annotator's labels, for kappa.
    key = sample[["item_id", "parent_seed_id", "source_type",
                  "audit_is_benign", "audit_is_attack_adjacent", "audit_notes"]].copy()
    key_path = out_dir / "second_annotator_key.csv"
    key.to_csv(key_path, index=False)

    print(f"Sample size: {len(sample)}")
    print("Stratum counts:")
    print(sample["_stratum"].value_counts().to_string())
    print(f"\nBlind sheet (send this): {blind_path}")
    print(f"Held-back key (keep):    {key_path}")
    print("\nThe blind sheet contains no first-annotator labels.")


if __name__ == "__main__":
    main()
