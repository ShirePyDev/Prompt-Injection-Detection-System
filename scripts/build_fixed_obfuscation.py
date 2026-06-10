"""
Build a fixed obfuscation evaluation dataset from test.csv clean attack seeds.

Output files:
  data/pids_bench_v3/heldout_obfuscation_fixed.csv       (all transforms)
  data/pids_bench_v3/heldout_obfuscation_fixed_seeds.csv (seeds only)

Reads ONLY from data/pids_bench_v3/test.csv. Does not touch train/val.
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
# INPUT_PATH  = ROOT / "data/pids_bench_v3/test.csv"
# OUT_FULL    = ROOT / "data/pids_bench_v3/heldout_obfuscation_fixed.csv"
# OUT_SEEDS   = ROOT / "data/pids_bench_v3/heldout_obfuscation_fixed_seeds.csv"

INPUT_PATH = Path("/content/drive/MyDrive/pids_bench_v3/test.csv")
OUT_FULL = Path("/content/drive/MyDrive/pids_bench_v3/heldout_obfuscation_fixed.csv")
OUT_SEEDS = Path("/content/drive/MyDrive/pids_bench_v3/heldout_obfuscation_fixed_seeds.csv")

EXPECTED_COLS = [
    "parent_seed_id", "text", "label", "source",
    "attack_type", "language", "generator_model",
    "paraphrase_round", "source_type", "obfuscation",
]

TARGET_SEEDS     = 150
MIN_TEXT_LEN     = 100
CHUNK_REPEAT_MAX = 2          # reject if any chunk appears > this many times
THIRDS_SIM_THRESH = 0.85      # reject if all three thirds are near-identical
RANDOM_STATE     = 42


# ── repetition detection ───────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Collapse whitespace for comparison only; does not change output."""
    return re.sub(r"\s+", " ", text).strip()


def _split_chunks(text: str) -> list[str]:
    """Split text into sentence/line-like chunks for repetition analysis."""
    # Split on newlines first, then on sentence-ending punctuation
    lines = re.split(r"\n+", text)
    chunks = []
    for line in lines:
        parts = re.split(r"(?<=[.!?])\s+", line.strip())
        chunks.extend(p for p in parts if p)
    return chunks


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _is_repetitive(text: str) -> bool:
    """
    Return True if the text is obviously repetitive:
      - any normalized chunk appears 3+ times, OR
      - the three equal-length thirds of the text are all near-identical.
    """
    chunks = [_normalize(c) for c in _split_chunks(text) if len(c.strip()) > 10]

    # chunk frequency check
    freq: dict[str, int] = {}
    for c in chunks:
        freq[c] = freq.get(c, 0) + 1
    if any(v > CHUNK_REPEAT_MAX for v in freq.values()):
        return True

    # thirds similarity check
    n = len(text)
    if n < 60:
        return False
    t1 = _normalize(text[:n // 3])
    t2 = _normalize(text[n // 3: 2 * n // 3])
    t3 = _normalize(text[2 * n // 3:])
    if (
        _similarity(t1, t2) >= THIRDS_SIM_THRESH
        and _similarity(t2, t3) >= THIRDS_SIM_THRESH
    ):
        return True

    return False


# ── transformations ────────────────────────────────────────────────────────────

_LEETSPEAK_MAP = str.maketrans("aAeEiIoOsSlL", "443311005511")

def apply_leetspeak(text: str) -> str:
    return text.translate(_LEETSPEAK_MAP)


# Cyrillic look-alikes for homoglyph substitution
_HOMOGLYPH_MAP = str.maketrans({
    "e": "е",   # е  Cyrillic small e
    "E": "Е",   # Е  Cyrillic capital E
    "a": "а",   # а  Cyrillic small a
    "A": "А",   # А  Cyrillic capital A
    "o": "о",   # о  Cyrillic small o
    "O": "О",   # О  Cyrillic capital O
    "c": "с",   # с  Cyrillic small es
    "C": "С",   # С  Cyrillic capital Es
    "p": "р",   # р  Cyrillic small er
    "P": "Р",   # Р  Cyrillic capital Er
    "x": "х",   # х  Cyrillic small ha
    "X": "Х",   # Х  Cyrillic capital Ha
})

def apply_homoglyph(text: str) -> str:
    return text.translate(_HOMOGLYPH_MAP)


_ZWS = "​"  # U+200B ZERO WIDTH SPACE

def apply_zero_width(text: str) -> str:
    """Insert a zero-width space between words only (not between every char)."""
    return _ZWS.join(text.split(" "))


TRANSFORMS = {
    "clean":      lambda t: t,
    "leetspeak":  apply_leetspeak,
    "homoglyph":  apply_homoglyph,
    "zero_width": apply_zero_width,
}


# ── stratified sample ─────────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, n: int, col: str, seed: int) -> pd.DataFrame:
    """
    Sample up to n rows from df, preserving proportions of col.
    If df has <= n rows, return df as-is.
    """
    if len(df) <= n:
        return df.copy()

    counts = df[col].value_counts()
    total  = counts.sum()
    frames = []
    allocated = 0

    for i, (val, cnt) in enumerate(counts.items()):
        if i == len(counts) - 1:
            # give remainder to last group to avoid rounding loss
            k = n - allocated
        else:
            k = max(1, round(n * cnt / total))
        k = min(k, cnt)
        frames.append(
            df[df[col] == val].sample(k, random_state=seed)
        )
        allocated += k

    return pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. load
    if not INPUT_PATH.exists():
        sys.exit(f"ERROR: input file not found: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows from {INPUT_PATH}")

    # 2. column check
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing expected columns: {missing}")

    original_attack_types = set(df[df["label"] == 1]["attack_type"].unique())

    # 3. filter: label==1, obfuscation=="none"
    pool = df[(df["label"] == 1) & (df["obfuscation"] == "none")].copy()
    print(f"Clean attack pool (label=1, obfuscation=none): {len(pool)}")

    # 4. text length filter
    pool = pool[pool["text"].notna()]
    pool = pool[pool["text"].str.strip().str.len() >= MIN_TEXT_LEN]
    print(f"After text length filter (>={MIN_TEXT_LEN} chars): {len(pool)}")

    # 5. repetition filter
    mask_keep = pool["text"].apply(lambda t: not _is_repetitive(str(t)))
    pool = pool[mask_keep]
    print(f"After repetition filter: {len(pool)}")

    # 6. deduplicate parent_seed_id
    pool = pool.drop_duplicates(subset="parent_seed_id", keep="first")
    print(f"After dedup parent_seed_id: {len(pool)}")

    # warn if any attack_type vanished
    remaining_attack_types = set(pool["attack_type"].unique())
    lost = original_attack_types - remaining_attack_types
    if lost:
        print(f"WARNING: attack_type(s) lost after filtering: {lost}")

    # 7. stratified sample
    seeds_df = stratified_sample(pool, TARGET_SEEDS, "attack_type", RANDOM_STATE)
    seeds_df = seeds_df.reset_index(drop=True)
    seeds_df["seed_id"] = seeds_df.index
    n_seeds = len(seeds_df)
    print(f"Selected seeds: {n_seeds}")

    # 8. build output rows
    rows = []
    for _, row in seeds_df.iterrows():
        original = str(row["text"])
        for transform_name, fn in TRANSFORMS.items():
            transformed = fn(original)
            rows.append({
                "seed_id":          row["seed_id"],
                "transform":        transform_name,
                "original_text":    original,
                "transformed_text": transformed,
                "label":            row["label"],
                "attack_type":      row["attack_type"],
                "source":           row["source"],
                "source_type":      row["source_type"],
                "parent_seed_id":   row["parent_seed_id"],
            })

    out_df = pd.DataFrame(rows)

    # ── sanity checks ──────────────────────────────────────────────────────────

    # no empty transformed_text
    empty_transformed = out_df["transformed_text"].isna() | (out_df["transformed_text"].str.strip() == "")
    assert not empty_transformed.any(), "FAIL: some transformed_text values are empty"

    # clean transform == original_text
    clean_rows = out_df[out_df["transform"] == "clean"]
    mismatch = clean_rows["transformed_text"] != clean_rows["original_text"]
    assert not mismatch.any(), "FAIL: clean transformed_text != original_text"

    # all labels == 1
    assert (out_df["label"] == 1).all(), "FAIL: non-injection labels in output"

    # no duplicate seed_id + transform
    dupes = out_df.duplicated(subset=["seed_id", "transform"])
    assert not dupes.any(), "FAIL: duplicate seed_id + transform pairs"

    # exactly four transforms present
    transforms_found = set(out_df["transform"].unique())
    transforms_expected = {"clean", "leetspeak", "homoglyph", "zero_width"}
    assert transforms_found == transforms_expected, \
        f"FAIL: transform mismatch — found {transforms_found}"

    # every seed has exactly 4 rows
    seed_counts = out_df.groupby("seed_id").size()
    assert (seed_counts == 4).all(), \
        f"FAIL: some seeds don't have exactly 4 rows: {seed_counts[seed_counts != 4]}"

    print("\nAll sanity checks passed.")

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Total rows in test.csv:              {total_rows}")
    print(f"Final selected seeds:                {n_seeds}")
    print(f"Final output rows:                   {len(out_df)}")
    print(f"\nTransform counts:")
    print(out_df["transform"].value_counts().to_string())
    print(f"\nAttack type counts (by seed):")
    print(seeds_df["attack_type"].value_counts().to_string())
    print(f"\nOutput files:")
    print(f"  {OUT_FULL}")
    print(f"  {OUT_SEEDS}")

    # ── save ───────────────────────────────────────────────────────────────────
    if OUT_FULL.exists():
        raise RuntimeError(f"Refusing to overwrite existing file: {OUT_FULL}")

    if OUT_SEEDS.exists():
        raise RuntimeError(f"Refusing to overwrite existing file: {OUT_SEEDS}")

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_FULL, index=False)

    seeds_out_cols = [
        "seed_id", "parent_seed_id", "text", "label",
        "attack_type", "source", "source_type",
    ]
    seeds_df[seeds_out_cols].to_csv(OUT_SEEDS, index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
