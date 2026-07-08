#!/usr/bin/env python3
"""
strip_lmsys_text.py

Removes the `text` of the verbatim LMSYS rows in
eval_subsets/hard_benign_test.csv, which the LMSYS license does not permit us to
redistribute. Every other column is left untouched, including:
  - parent_seed_id : stable row id
  - text_sha256    : the fingerprint of the removed text (the rebuild key)

Only LMSYS rows are stripped. All non-LMSYS rows keep their text.

Safety: a backup copy of the file is written next to it (suffix .prestrip.bak)
before anything is changed, and the strip is verified by reading the file back.
The backup is for local recovery only and must NOT be committed or released, since
it still contains the LMSYS text.

Usage:
  python strip_lmsys_text.py --data-root data/pids_bench_v3
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

TARGET = "eval_subsets/hard_benign_test.csv"
LMSYS_PREFIX = "lmsys"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/pids_bench_v3")
    args = parser.parse_args()

    path = Path(args.data_root) / TARGET
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    df = pd.read_csv(path)
    is_lmsys = df["source"].astype(str).str.lower().str.startswith(LMSYS_PREFIX)
    n_lmsys = int(is_lmsys.sum())
    if n_lmsys == 0:
        raise SystemExit("No LMSYS rows found; nothing to strip.")

    # Refuse to run if any LMSYS row lacks a fingerprint -- without it that row
    # could not be rebuilt, so stripping it would lose data.
    missing_fp = df.loc[is_lmsys, "text_sha256"].astype(str).str.len().eq(0).sum()
    if missing_fp:
        raise SystemExit(
            f"{missing_fp} LMSYS row(s) have no text_sha256; refusing to strip. "
            "Run add_restricted_fingerprints.py first."
        )

    # Backup the current file (still contains the LMSYS text) for local recovery.
    backup = path.with_suffix(".prestrip.bak")
    shutil.copy2(path, backup)
    print(f"Backup written: {backup}  (local only -- do not commit or release)")

    # Strip: blank the text of LMSYS rows. Keep everything else.
    df.loc[is_lmsys, "text"] = ""
    df.to_csv(path, index=False)
    print(f"Stripped text from {n_lmsys} LMSYS rows in {TARGET}")

    # Verify by reading back.
    check = pd.read_csv(path)
    is_lmsys2 = check["source"].astype(str).str.lower().str.startswith(LMSYS_PREFIX)
    lmsys_text_empty = check.loc[is_lmsys2, "text"].astype(str).str.len().eq(0).sum()
    nonlmsys_text_present = check.loc[~is_lmsys2, "text"].astype(str).str.len().gt(0).sum()
    lmsys_fp_present = check.loc[is_lmsys2, "text_sha256"].astype(str).str.len().gt(0).sum()

    print("\nVerification after strip:")
    print(f"  LMSYS rows with empty text     : {lmsys_text_empty}/{int(is_lmsys2.sum())}")
    print(f"  LMSYS rows keeping fingerprint : {lmsys_fp_present}/{int(is_lmsys2.sum())}")
    print(f"  non-LMSYS rows keeping text    : {nonlmsys_text_present}/{int((~is_lmsys2).sum())}")

    ok = (
        lmsys_text_empty == int(is_lmsys2.sum())
        and lmsys_fp_present == int(is_lmsys2.sum())
        and nonlmsys_text_present == int((~is_lmsys2).sum())
    )
    print("\nResult:", "OK -- strip clean and reversible" if ok else "PROBLEM -- review before proceeding")


if __name__ == "__main__":
    main()
