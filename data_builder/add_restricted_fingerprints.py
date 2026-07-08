#!/usr/bin/env python3
"""
add_restricted_fingerprints.py

Adds a per-row content fingerprint to the LMSYS rows in the two files that contain
them, computed from the text that is currently still present. Run this BEFORE
stripping the text. The fingerprint column becomes the key the rebuild uses to put
the right text back in the right row, with no reliance on row order.

For non-LMSYS rows the new column is left blank; their text is not being removed.

Files touched:
  eval_subsets/hard_benign_test.csv
  ood/structural_ood.csv

After this runs, each LMSYS row carries `text_sha256`. The value matches the
fingerprint already stored in restricted_rebuild_manifest.csv, so the rebuild can
match a stripped row to its text by hash alone.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

LMSYS_PREFIX = "lmsys"

FILES = (
    "eval_subsets/hard_benign_test.csv",
    "ood/structural_ood.csv",
)


def normalize_text(text: str) -> str:
    """Identical to build_hard_benign._normalize_text. Must not diverge."""
    text = str(text or "").encode("utf-8", errors="replace").decode("utf-8")
    text = "".join(ch for ch in text if ch in "\n\t\r" or ord(ch) >= 32)
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/pids_bench_v3")
    args = parser.parse_args()
    data_root = Path(args.data_root)

    for rel_path in FILES:
        path = data_root / rel_path
        if not path.exists():
            print(f"[skip] {rel_path} not found")
            continue

        df = pd.read_csv(path)
        if "text_sha256" not in df.columns:
            df["text_sha256"] = ""

        is_lmsys = df["source"].astype(str).str.startswith(LMSYS_PREFIX)
        count = 0
        for i in df.index[is_lmsys]:
            normalized = normalize_text(df.at[i, "text"])
            df.at[i, "text_sha256"] = sha256_text(normalized)
            count += 1

        df.to_csv(path, index=False)
        print(f"{rel_path}: wrote fingerprints for {count} LMSYS rows")

    print("\nDone. Verify the fingerprints match the manifest before stripping text:")
    print("  (next step: cross-check against restricted_rebuild_manifest.csv)")


if __name__ == "__main__":
    main()
