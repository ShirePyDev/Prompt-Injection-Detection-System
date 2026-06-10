"""
Manual validation of held-out obfuscation samples.

You review N samples per transform (default 15 x 3 = 45 total) and confirm
that each one still reads as a coherent injection attempt. This is the
"semantic preservation was checked" evidence cited in the paper.

Usage:
    python validate_samples.py \
        --heldout_csv /path/to/heldout_obfuscation.csv \
        --output_json /path/to/validation_results.json \
        --n_per_transform 15

What you'll see for each sample:
    - the ORIGINAL text
    - the TRANSFORMED text
    - a prompt: "Does the transformed text still read as a coherent injection
      attempt to you? [y/n/skip]"

What gets recorded:
    - seed_id, transform, your judgment, optional note
    - aggregate counts per transform

Design notes (for the paper):
- We use stratified random sampling across transforms, not just the first N
  rows, so the audit covers the distribution fairly.
- The random state is fixed so a re-run reproduces the same sample order.
- If fewer than n_per_transform rows exist for a transform, we take all.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42


def run_validation(df: pd.DataFrame, n_per_transform: int):
    """Interactive loop. Returns list of judgment dicts."""
    records = []
    transforms = sorted(df["transform"].unique())

    total_samples = sum(
        min(n_per_transform, len(df[df["transform"] == t]))
        for t in transforms
    )
    print(f"\nYou will review {total_samples} samples "
          f"({n_per_transform} per transform, {len(transforms)} transforms).")
    print("For each, answer: y = still coherent injection, "
          "n = broken/incoherent, s = skip.\n")
    print("=" * 70)

    sample_idx = 0
    for transform in transforms:
        pool = df[df["transform"] == transform]
        n_take = min(n_per_transform, len(pool))
        sample = pool.sample(n=n_take, random_state=RANDOM_SEED)

        print(f"\n### Transform: {transform} ({n_take} samples) ###\n")

        for _, row in sample.iterrows():
            sample_idx += 1
            print(f"[{sample_idx}/{total_samples}] transform={transform}")
            print(f"    seed_id   : {row['seed_id']}")
            print(f"    subtype   : {row['subtype']}")
            print(f"    ORIGINAL  : {row['original_text'][:200]}")
            print(f"    TRANSFORMED: {row['transformed_text'][:200]}")
            print()

            while True:
                ans = input("    Still coherent injection? [y/n/s]: ").strip().lower()
                if ans in ("y", "n", "s"):
                    break
                print("    Please enter y, n, or s.")

            note = ""
            if ans == "n":
                note = input("    Optional note on why: ").strip()

            records.append({
                "seed_id": row["seed_id"],
                "transform": transform,
                "subtype": row["subtype"],
                "judgment": ans,
                "note": note,
            })
            print()

    return records


def summarize(records):
    """Print per-transform coherence rates."""
    df = pd.DataFrame(records)
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    for transform in sorted(df["transform"].unique()):
        sub = df[df["transform"] == transform]
        n = len(sub)
        y = (sub["judgment"] == "y").sum()
        nn = (sub["judgment"] == "n").sum()
        s = (sub["judgment"] == "s").sum()
        rate = y / (y + nn) if (y + nn) > 0 else 0.0
        print(f"  {transform:12} n={n:3}  coherent={y:3}  "
              f"broken={nn:3}  skip={s:3}  rate={rate:.1%}")
    total_y = (df["judgment"] == "y").sum()
    total_n = (df["judgment"] == "n").sum()
    total_rate = total_y / (total_y + total_n) if (total_y + total_n) > 0 else 0.0
    print(f"  {'OVERALL':12} rate={total_rate:.1%}")
    print("=" * 70)
    return {
        "per_transform": {
            t: {
                "n": int((df["transform"] == t).sum()),
                "coherent": int(((df["transform"] == t) & (df["judgment"] == "y")).sum()),
                "broken": int(((df["transform"] == t) & (df["judgment"] == "n")).sum()),
                "skip": int(((df["transform"] == t) & (df["judgment"] == "s")).sum()),
            }
            for t in sorted(df["transform"].unique())
        },
        "overall_coherence_rate": total_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout_csv", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--n_per_transform", type=int, default=15)
    args = ap.parse_args()

    df = pd.read_csv(args.heldout_csv)
    print(f"Loaded {len(df)} rows from {args.heldout_csv}")

    records = run_validation(df, args.n_per_transform)
    summary = summarize(records)

    output = {
        "n_per_transform": args.n_per_transform,
        "records": records,
        "summary": summary,
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2))
    print(f"\nWrote {len(records)} judgments to {args.output_json}")


if __name__ == "__main__":
    main()
