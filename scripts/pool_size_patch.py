#!/usr/bin/env python3
"""
Reference implementation for the --pool-size parameterization of the hard-negative
augmentation experiment.

This is NOT a standalone script. It contains the two pieces to merge into
src/baselines/deberta_v3_hardneg.py:

  1. subsample_by_category()  -- a new helper, added near the top of the file.
  2. a modified verify_hard_negatives() -- replaces the existing one, adding a
     pool_size parameter that draws a stratified subsample and relaxes only the
     count assertion. The label invariant (all labels == 0) is KEPT unchanged.

The training path is untouched: with --pool-size 419 (or omitted) the behavior is
identical to the current script, so the existing 419-pool runs remain the anchor.

Argument to add in the CLI parser:

    parser.add_argument(
        "--pool-size", type=int, default=None,
        help="Number of hard-negative TRAIN rows to use (stratified by "
             "hardneg_category). Default None uses all rows, reproducing the "
             "current behavior. The validation pool is never subsampled.")

And pass args.pool_size into verify_hard_negatives(paths, pool_size=args.pool_size).
"""

import pandas as pd

# Keep these as the FULL-pool reference sizes.
HN_TRAIN_FULL = 419
HN_VAL_EXPECTED = 116


def subsample_by_category(
    hn_tr: pd.DataFrame, pool_size: int, seed: int
) -> pd.DataFrame:
    """Draw `pool_size` rows from the hard-negative train pool, preserving the
    hardneg_category proportions as closely as integer allocation allows.

    Reproducible given `seed`. If pool_size >= len(hn_tr), returns the full pool
    unchanged. The category mix is held constant so that pool SIZE is the only
    variable across the sweep, not category balance.
    """
    if pool_size >= len(hn_tr):
        return hn_tr.copy()
    if "hardneg_category" not in hn_tr.columns:
        # No category column: fall back to a plain reproducible draw.
        return hn_tr.sample(pool_size, random_state=seed).reset_index(drop=True)

    frac = pool_size / len(hn_tr)
    parts = []
    for _, grp in hn_tr.groupby("hardneg_category"):
        k = max(1, round(len(grp) * frac))
        parts.append(grp.sample(min(k, len(grp)), random_state=seed))
    drawn = pd.concat(parts)

    # Integer rounding can over- or under-shoot; correct to exactly pool_size.
    if len(drawn) > pool_size:
        drawn = drawn.sample(pool_size, random_state=seed)
    elif len(drawn) < pool_size:
        rest = hn_tr.drop(index=drawn.index)
        topup = rest.sample(min(pool_size - len(drawn), len(rest)), random_state=seed)
        drawn = pd.concat([drawn, topup])

    return drawn.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def verify_hard_negatives(paths, pool_size=None, seed=42):
    """Load and verify the hard-negative data, optionally subsampling the TRAIN
    pool to `pool_size` rows (stratified by category). The validation pool is
    never subsampled, so validation is held fixed across the sweep.
    """
    print("\n" + "=" * 65)
    print("STEP 3 — Loading and verifying hard-negative data")
    print("=" * 65)
    hn_tr = pd.read_csv(paths["hn_train"])
    hn_va = pd.read_csv(paths["hn_val"])

    # Verify the full pool first, so the source data is always checked at full size.
    assert len(hn_tr) == HN_TRAIN_FULL, (
        f"Expected {HN_TRAIN_FULL} train hard-negatives in the source file, "
        f"got {len(hn_tr)}"
    )
    assert len(hn_va) == HN_VAL_EXPECTED, (
        f"Expected {HN_VAL_EXPECTED} val hard-negatives, got {len(hn_va)}"
    )
    # The label invariant is a genuine data check and is NEVER relaxed.
    assert (hn_tr["label"] == 0).all(), "Non-zero label found in hard_negative_train!"
    assert (hn_va["label"] == 0).all(), "Non-zero label found in hard_negative_val!"

    # Optional stratified subsample of the TRAIN pool only.
    if pool_size is not None and pool_size < len(hn_tr):
        hn_tr = subsample_by_category(hn_tr, pool_size, seed)
        assert (hn_tr["label"] == 0).all(), "Subsample introduced a non-zero label!"
        print(f"\nPool-size sweep: using {len(hn_tr)} of {HN_TRAIN_FULL} train "
              f"hard-negatives (stratified by category, seed={seed}).")

    print(f"\nhard_negative_train.csv  : {len(hn_tr)} rows in use")
    print(f"  columns    : {list(hn_tr.columns)}")
    print(f"  label dist : {hn_tr['label'].value_counts().to_dict()}")
    if "hardneg_category" in hn_tr.columns:
        print(f"  categories : {hn_tr['hardneg_category'].value_counts().to_dict()}")
    print(f"\nhard_negative_val.csv    : {len(hn_va)} rows (not subsampled)")

    return hn_tr, hn_va
