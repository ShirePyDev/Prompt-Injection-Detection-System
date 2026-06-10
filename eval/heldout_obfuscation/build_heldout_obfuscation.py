"""
Build the held-out obfuscation stress set for PIDS-Bench.

Usage:
    python build_heldout_obfuscation.py \
        --test_csv /path/to/test.csv \
        --train_csv /path/to/train.csv \
        --output_csv /path/to/heldout_obfuscation.csv

Produces:
    heldout_obfuscation.csv with columns:
        seed_id, transform, original_text, transformed_text, label, subtype

Design decisions (documented for Section IV-F of the paper):

- Seeds are drawn ONLY from the held-out test split. No training-set examples
  are used, so evaluation on this set is truly out-of-distribution relative
  to both (a) the training attack distribution and (b) the training obfuscation
  distribution.

- Seeds are stratified across attack subtypes: direct_override,
  contextual_manip, roleplay_attack, data_exfiltration. We exclude
  encoded_attack (already encoded; re-encoding would confound measurement)
  and tool_injection (the transforms do not apply cleanly to tool-call
  JSON payloads).

- We take up to 150 seeds per transform. If the filtered test pool is smaller,
  we take what exists and report the actual count. This prevents silent
  quota failures.

- We validate that NO selected seed text (after transformation) exists in
  the training set, using exact-match string comparison on the transformed
  output. This would only fail if a training example already contained the
  transformation naturally, which should be impossible given we selected
  plain-text seeds, but we check anyway.

- The same 150 seed ids are used across all three transforms so that
  cross-transform comparison is apples-to-apples: a drop in recall between
  transforms reflects the transform, not the underlying seed difficulty.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from transforms import TRANSFORMS, apply_transform


# Subtypes eligible for transformation. See docstring above for rationale.
ELIGIBLE_SUBTYPES = [
    "direct_override",
    "contextual_manipulation",
    "roleplay_attack",
    "data_exfiltration",
]

# Target seeds per transform. Actual count may be lower if pool is small.
TARGET_SEEDS_PER_TRANSFORM = 150

# Seeds per subtype (approximate; leftover goes to direct_override which is
# the largest pool).
SEEDS_PER_SUBTYPE = {
    "direct_override": 60,
    "contextual_manipulation": 30,
    "roleplay_attack": 30,
    "data_exfiltration": 30,
}

RANDOM_SEED = 42  # matches main paper seed for reproducibility


def sha256_short(text: str) -> str:
    """Return a short checksum for a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def select_seeds(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select plain-text attack seeds from the held-out test split.

    Stratified by subtype. Deterministic given RANDOM_SEED.
    """
    # Filter to attacks only
    attacks = test_df[test_df["label"] == 1].copy()
    print(f"[select_seeds] held-out attacks total: {len(attacks)}")

    # Check subtype column exists; fail loudly if not
    if "attack_type" not in attacks.columns:
        raise KeyError(
            "Expected column 'attack_type' in test CSV. If your column has a "
            "different name, edit this script. Do not silently proceed."
        )

    # Filter to eligible subtypes
    attacks = attacks[attacks["attack_type"].isin(ELIGIBLE_SUBTYPES)].copy()
    print(f"[select_seeds] after subtype filter: {len(attacks)}")
    print(f"[select_seeds] per-subtype counts:")
    for st, n in attacks["attack_type"].value_counts().items():
        print(f"    {st:25} {n}")

    # Stratified sample
    selected_parts = []
    for subtype, quota in SEEDS_PER_SUBTYPE.items():
        pool = attacks[attacks["attack_type"] == subtype]
        n_take = min(quota, len(pool))
        if n_take < quota:
            print(
                f"[select_seeds] WARNING: subtype '{subtype}' has only "
                f"{len(pool)} rows, requested {quota}, taking {n_take}"
            )
        sample = pool.sample(n=n_take, random_state=RANDOM_SEED)
        selected_parts.append(sample)

    selected = pd.concat(selected_parts, ignore_index=True)
    print(f"[select_seeds] total selected: {len(selected)}")
    return selected


def build_heldout_set(seeds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply each transform to each seed. Return long-format dataframe:
    one row per (seed, transform) pair.
    """
    rows = []
    for _, seed in seeds_df.iterrows():
        original_text = str(seed["text"])
        seed_id = sha256_short(original_text)
        for transform_name in TRANSFORMS.keys():
            transformed = apply_transform(transform_name, original_text)
            # Sanity: transformed text should differ from original for any
            # seed that contains at least one attack keyword. If it's
            # identical, the seed had no keywords and we skip it.
            if transformed == original_text:
                continue
            rows.append({
                "seed_id": seed_id,
                "transform": transform_name,
                "original_text": original_text,
                "transformed_text": transformed,
                "label": 1,  # all are attacks
                "subtype": seed["attack_type"],
            })
    out_df = pd.DataFrame(rows)
    print(f"[build_heldout_set] produced {len(out_df)} rows "
          f"across {out_df['transform'].nunique()} transforms")
    print(f"[build_heldout_set] per-transform counts:")
    for t, n in out_df["transform"].value_counts().items():
        print(f"    {t:15} {n}")
    return out_df


def verify_no_training_overlap(
    heldout_df: pd.DataFrame, train_df: pd.DataFrame
) -> None:
    """
    Check that no transformed text appears in training.

    Uses exact string match. If training contained a leetspeak/homoglyph/
    zero-width example naturally, this would catch it. Given we filtered
    training-side encodings to Base64/ROT13/Unicode substitution, this
    should find zero overlap.
    """
    train_texts = set(train_df["text"].astype(str))
    overlaps = heldout_df[heldout_df["transformed_text"].isin(train_texts)]
    if len(overlaps) > 0:
        print(f"[verify] ERROR: {len(overlaps)} transformed texts appear "
              f"in training set. Example:")
        print(overlaps.iloc[0]["transformed_text"])
        sys.exit(1)
    print(f"[verify] OK: 0 of {len(heldout_df)} transformed texts appear "
          f"in training set.")


def write_manifest(output_csv: Path, df: pd.DataFrame) -> None:
    """Write a JSON manifest with checksums and counts."""
    manifest = {
        "csv_file": str(output_csv.name),
        "row_count": len(df),
        "transforms": df["transform"].value_counts().to_dict(),
        "subtypes": df["subtype"].value_counts().to_dict(),
        "random_seed": RANDOM_SEED,
        "csv_sha256": hashlib.sha256(
            output_csv.read_bytes()
        ).hexdigest(),
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {manifest_path}")
    print(f"[manifest] csv sha256: {manifest['csv_sha256']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True,
                    help="Path to held-out test split CSV")
    ap.add_argument("--train_csv", required=True,
                    help="Path to training split CSV (for overlap check)")
    ap.add_argument("--output_csv", required=True,
                    help="Path to write held-out obfuscation CSV")
    args = ap.parse_args()

    test_df = pd.read_csv(args.test_csv)
    train_df = pd.read_csv(args.train_csv)
    print(f"Loaded test={len(test_df)} rows, train={len(train_df)} rows")

    seeds = select_seeds(test_df)
    heldout = build_heldout_set(seeds)
    verify_no_training_overlap(heldout, train_df)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(output_path, index=False)
    print(f"[write] wrote {len(heldout)} rows to {output_path}")

    write_manifest(output_path, heldout)


if __name__ == "__main__":
    main()
