"""
IAA pipeline — two steps:

  Step 1 (sample):  python eval/iaa_sample_and_compute.py --step sample
      Produces: outputs/iaa/annotation_sheet.csv
      You and a second annotator each fill in their column independently.

  Step 2 (compute): python eval/iaa_sample_and_compute.py --step compute
      Reads:    outputs/iaa/annotation_sheet.csv  (both columns filled)
      Prints:   Cohen's kappa + Landis-Koch band
      Produces: outputs/iaa/kappa_result.json
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ── config ───────────────────────────────────────────────────────────────────
SEED = 42
N_TOTAL = 150
OUT_DIR = "outputs/iaa"

# Target proportions (Table 1 in paper) — must sum to 1.0
PROPORTIONS = {
    "direct_override":          0.31,
    "contextual_manipulation":  0.21,
    "roleplay_attack":          0.16,
    "data_exfiltration":        0.13,
    "encoded_attack":           0.10,
    "tool_injection":           0.09,
}

SUBTYPE_LABELS = list(PROPORTIONS.keys())

LANDIS_KOCH = [
    (0.00, "Slight"),
    (0.20, "Fair"),
    (0.40, "Moderate"),
    (0.60, "Substantial"),
    (0.80, "Almost perfect"),
]


def landis_koch_band(kappa: float) -> str:
    band = "Slight"
    for threshold, label in LANDIS_KOCH:
        if kappa >= threshold:
            band = label
    return band


# ── step 1: sample ───────────────────────────────────────────────────────────

def run_sample():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all splits and keep injections only
    dfs = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"data/pids_bench_v3/{split}.csv")
        dfs.append(df)
    pool = pd.concat(dfs, ignore_index=True)
    pool = pool[pool["label"] == 1].copy()

    # Compute exact per-stratum counts that sum to N_TOTAL
    raw_counts = {k: v * N_TOTAL for k, v in PROPORTIONS.items()}
    floor_counts = {k: int(v) for k, v in raw_counts.items()}
    remainders = {k: raw_counts[k] - floor_counts[k] for k in raw_counts}
    shortfall = N_TOTAL - sum(floor_counts.values())
    # Distribute shortfall to subtypes with largest remainder
    top_up = sorted(remainders, key=remainders.get, reverse=True)[:shortfall]
    for k in top_up:
        floor_counts[k] += 1

    assert sum(floor_counts.values()) == N_TOTAL, "Counts do not sum to 150"

    print("Stratum sizes:")
    for k, v in floor_counts.items():
        print(f"  {k}: {v}")

    rng = random.Random(SEED)
    sampled_rows = []
    for subtype, n in floor_counts.items():
        subset = pool[pool["attack_type"] == subtype]
        if len(subset) < n:
            raise ValueError(
                f"Not enough rows for {subtype}: need {n}, have {len(subset)}"
            )
        chosen = subset.sample(n=n, random_state=SEED)
        sampled_rows.append(chosen)

    sample = pd.concat(sampled_rows, ignore_index=True)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle
    sample["row_id"] = [f"iaa_{i+1:03d}" for i in range(len(sample))]

    # Build annotation sheet — keep only what annotators need
    sheet = sample[["row_id", "text", "attack_type"]].copy()
    sheet = sheet.rename(columns={"attack_type": "assigned_subtype"})

    # Split into two files:
    #   annotation_sheet.csv  — shown to annotators (no true label)
    #   annotation_key.csv    — kept by researcher (has true label + row_id)

    annotator_sheet = sheet[["row_id", "text"]].copy()  # assigned_subtype withheld from annotators
    annotator_sheet["annotator_1"] = ""
    annotator_sheet["annotator_2"] = ""
    annotator_sheet["instructions"] = (
        "Label each prompt with exactly one of: "
        + " | ".join(SUBTYPE_LABELS)
    )

    key = sheet.copy()

    annotator_out = os.path.join(OUT_DIR, "annotation_sheet.csv")
    key_out = os.path.join(OUT_DIR, "annotation_key.csv")

    annotator_sheet.to_csv(annotator_out, index=False)
    key.to_csv(key_out, index=False)

    print(f"\nAnnotation sheet saved: {annotator_out}")
    print(f"Answer key saved:       {key_out}")
    print(
        "\nNext steps:"
        "\n  1. Share annotation_sheet.csv with your second annotator."
        "\n  2. Both of you fill in your own column (annotator_1, annotator_2)"
        "\n     independently — no discussion until both are done."
        "\n  3. Merge the two filled sheets back into annotation_sheet.csv."
        "\n  4. Run: python eval/iaa_sample_and_compute.py --step compute"
    )


# ── step 2: compute ──────────────────────────────────────────────────────────

def run_compute(sheet_path: str = None):
    if sheet_path is None:
        sheet_path = os.path.join(OUT_DIR, "annotation_sheet.csv")
    sheet = pd.read_csv(sheet_path)

    # Auto-detect annotator columns (any column that is not row_id / text / instructions)
    skip_cols = {"row_id", "text", "instructions"}
    annotator_cols = [c for c in sheet.columns if c not in skip_cols]
    if len(annotator_cols) != 2:
        raise ValueError(
            f"Expected exactly 2 annotator columns, found: {annotator_cols}"
        )
    col_a1, col_a2 = annotator_cols
    print(f"Annotator columns detected: '{col_a1}' vs '{col_a2}'")

    # Validate columns are filled
    missing_a1 = sheet[col_a1].isna() | (sheet[col_a1].astype(str).str.strip() == "")
    missing_a2 = sheet[col_a2].isna() | (sheet[col_a2].astype(str).str.strip() == "")

    if missing_a1.any():
        raise ValueError(f"'{col_a1}' has {missing_a1.sum()} empty cells. Fill all rows first.")
    if missing_a2.any():
        raise ValueError(f"'{col_a2}' has {missing_a2.sum()} empty cells. Fill all rows first.")

    # Validate label values
    a1 = sheet[col_a1].astype(str).str.strip().str.lower()
    a2 = sheet[col_a2].astype(str).str.strip().str.lower()
    valid = set(SUBTYPE_LABELS)
    bad_a1 = set(a1.unique()) - valid
    bad_a2 = set(a2.unique()) - valid
    if bad_a1:
        raise ValueError(f"'{col_a1}' contains unrecognised labels: {bad_a1}")
    if bad_a2:
        raise ValueError(f"'{col_a2}' contains unrecognised labels: {bad_a2}")

    kappa = cohen_kappa_score(a1.tolist(), a2.tolist())
    band = landis_koch_band(kappa)

    agreement_rate = (a1 == a2).mean()

    print(f"\n{'='*40}")
    print(f"  Cohen's κ : {kappa:.4f}")
    print(f"  Band      : {band}  (Landis & Koch 1977)")
    print(f"  Raw agree : {agreement_rate:.1%}  ({(a1==a2).sum()}/{len(sheet)} rows)")
    print(f"{'='*40}")
    print(f"\nInsert into paper:")
    print(f"  κ = {kappa:.3f}  →  {band} agreement")

    # Per-subtype breakdown
    print(f"\nPer-subtype agreement (by '{col_a1}' label):")
    sheet["_a1"] = a1
    sheet["_a2"] = a2
    for sub in SUBTYPE_LABELS:
        sub_rows = sheet[sheet["_a1"] == sub]
        if len(sub_rows) == 0:
            continue
        match = (sub_rows["_a1"] == sub_rows["_a2"]).mean()
        print(f"  {sub:<30} {match:.0%} ({len(sub_rows)} rows)")

    result = {
        "n_annotated": len(sheet),
        "annotator_columns": [col_a1, col_a2],
        "kappa": round(kappa, 4),
        "landis_koch_band": band,
        "raw_agreement_rate": round(float(agreement_rate), 4),
    }
    out_path = os.path.join(OUT_DIR, "kappa_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["sample", "compute"], required=True)
    parser.add_argument("--sheet", default=None,
                        help="Path to filled annotation sheet (compute step only). "
                             "Defaults to outputs/iaa/annotation_sheet.csv")
    args = parser.parse_args()

    if args.step == "sample":
        run_sample()
    else:
        run_compute(sheet_path=args.sheet)
