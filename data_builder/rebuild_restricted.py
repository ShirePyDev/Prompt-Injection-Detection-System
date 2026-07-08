#!/usr/bin/env python3
"""
rebuild_restricted.py

Restores the LMSYS-Chat-1M text that PIDS-Bench does not redistribute.

The public release ships eval_subsets/hard_benign_test.csv with the `text` of its
664 LMSYS rows removed. Each of those rows still carries `text_sha256`, the
fingerprint of its original (normalized) text. This script lets anyone who has
accepted the LMSYS-Chat-1M license recover that text locally: it streams the LMSYS
dataset, normalizes each candidate message the same way the benchmark did, hashes
it, and fills any stripped row whose fingerprint matches.

Matching is purely by content fingerprint, so it does not depend on the order LMSYS
returns rows or on any selection heuristic. A row is filled only when its exact
fingerprint is found in the licensed copy; rows that cannot be matched are reported,
not guessed.

Scope: only LMSYS rows in hard_benign_test.csv are stripped, so only that file is
rebuilt. All other files keep their text and are untouched.

Prerequisites:
  - Accept the LMSYS-Chat-1M license on Hugging Face; have a token.
  - pip install datasets pandas
  - export HF_TOKEN=hf_xxx   (or pass --hf-token)

Usage:
  python rebuild_restricted.py --data-root data/pids_bench_v3
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import pandas as pd

TARGET = "eval_subsets/hard_benign_test.csv"
LMSYS_PREFIX = "lmsys"
DEFAULT_MAX_SCAN = 250_000


def normalize_text(text: str) -> str:
    """Identical to build_hard_benign._normalize_text. The fingerprints were
    computed on text normalized this way, so the rebuild must match it exactly."""
    text = str(text or "").encode("utf-8", errors="replace").decode("utf-8")
    text = "".join(ch for ch in text if ch in "\n\t\r" or ord(ch) >= 32)
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def match_from_local(local_path: str, wanted: set) -> dict:
    """Match wanted fingerprints against a local CSV that contains the LMSYS text
    (e.g. a staged candidates file). Reads the file, normalizes and hashes each
    row's text, and returns {fingerprint: normalized_text} for any that match."""
    p = Path(local_path)
    if not p.exists():
        sys.exit(f"--local-source file not found: {p}")
    src = pd.read_csv(p)
    tcol = "text" if "text" in src.columns else src.columns[0]
    found: dict[str, str] = {}
    for value in src[tcol]:
        normalized = normalize_text(value)
        h = sha256_text(normalized)
        if h in wanted and h not in found:
            found[h] = normalized
    print(f"Local match: {len(found):,}/{len(wanted):,} fingerprints found in {p.name}.")
    return found


def match_from_stream(wanted: set, hf_token: str, max_scan: int) -> dict:
    """Stream LMSYS from Hugging Face and match wanted fingerprints. For end users
    who have accepted the LMSYS license."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets to run the rebuild.")

    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        streaming=True,
        token=hf_token or True,
    )

    found: dict[str, str] = {}
    scanned = 0
    for row in ds:
        if scanned >= max_scan or len(found) >= len(wanted):
            break
        scanned += 1
        if str(row.get("language", "")).strip().lower() != "english":
            continue
        if row.get("redacted", False):
            continue
        conversation = row.get("conversation", [])
        if not conversation:
            continue
        first_user = next(
            (m.get("content", "") for m in conversation if m.get("role") == "user"),
            "",
        )
        normalized = normalize_text(first_user)
        h = sha256_text(normalized)
        if h in wanted and h not in found:
            found[h] = normalized
        if scanned % 20000 == 0:
            print(f"  scanned={scanned:,} | matched={len(found):,}/{len(wanted):,}")

    print(f"Scan done: matched {len(found):,}/{len(wanted):,} fingerprints.")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/pids_bench_v3")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--max-scan", type=int, default=DEFAULT_MAX_SCAN)
    parser.add_argument(
        "--local-source",
        default="",
        help=(
            "Path to a local CSV containing the LMSYS text (e.g. a staged "
            "candidates file). If given, the rebuild matches against this file "
            "instead of streaming LMSYS from Hugging Face. The local file is your "
            "own licensed copy and must not be redistributed."
        ),
    )
    args = parser.parse_args()

    path = Path(args.data_root) / TARGET
    if not path.exists():
        sys.exit(f"Not found: {path}")

    df = pd.read_csv(path)
    is_lmsys = df["source"].astype(str).str.lower().str.startswith(LMSYS_PREFIX)
    lmsys = df[is_lmsys]
    if lmsys.empty:
        sys.exit("No LMSYS rows; nothing to rebuild.")

    # Rows still needing text: empty text but a fingerprint to match against.
    needs_text = lmsys["text"].isna() | (lmsys["text"].astype(str).str.len() == 0) \
        | (lmsys["text"].astype(str).str.lower() == "nan")
    wanted = set(lmsys.loc[needs_text, "text_sha256"])
    if not wanted:
        print("All LMSYS rows already have text; nothing to do.")
        return
    print(f"Need to recover {needs_text.sum()} rows ({len(wanted)} unique fingerprints).")

    if args.local_source:
        found = match_from_local(args.local_source, wanted)
    else:
        found = match_from_stream(wanted, args.hf_token, args.max_scan)

    # Fill each stripped row from its own fingerprint. No order assumption.
    filled = 0
    unmatched = 0
    for i in df.index[is_lmsys]:
        cur = df.at[i, "text"]
        cur_empty = pd.isna(cur) or str(cur).strip().lower() in ("", "nan")
        if not cur_empty:
            continue
        h = df.at[i, "text_sha256"]
        if h in found:
            df.at[i, "text"] = found[h]
            filled += 1
        else:
            unmatched += 1

    df.to_csv(path, index=False)
    print(f"\nFilled {filled} rows; {unmatched} could not be matched.")
    if unmatched:
        print(
            "Unmatched rows are conversations not present in your LMSYS copy. "
            "They are left empty; the rest of the benchmark is intact."
        )
    else:
        print("All stripped LMSYS rows restored. hard_benign_test.csv is complete.")


if __name__ == "__main__":
    main()
