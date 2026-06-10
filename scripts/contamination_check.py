"""
Contamination check: deepset/prompt-injections vs PIDS-Bench evaluation splits.

Outputs (written to repo root):
  contamination_exact_match_summary.csv
  contamination_near_duplicate_summary.csv
  contamination_near_duplicate_pairs.csv
  contamination_report.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT  # write outputs at repo root

# ---------------------------------------------------------------------------
# 1. Load public deepset dataset
# ---------------------------------------------------------------------------
print("=" * 65)
print("STEP 1 — Loading deepset/prompt-injections from Hugging Face")
print("=" * 65)

from datasets import load_dataset

ds = load_dataset("deepset/prompt-injections", split="train")
print(f"Dataset type : {type(ds)}")
print(f"Num rows     : {len(ds)}")
print(f"Column names : {ds.column_names}")
print(f"\nFirst example row:")
print(dict(list(ds[0].items())))

# Identify the text column
TEXT_COL_DS = "text"
assert TEXT_COL_DS in ds.column_names, (
    f"Expected column '{TEXT_COL_DS}' but got: {ds.column_names}"
)

deepset_texts_raw = ds[TEXT_COL_DS]
print(f"\nDeepset text column  : '{TEXT_COL_DS}'")
print(f"Example text         : {repr(deepset_texts_raw[0][:120])}")

# Normalised deepset texts
deepset_norm = [t.lower().strip() for t in deepset_texts_raw]
deepset_df = pd.DataFrame({"deepset_idx": range(len(deepset_texts_raw)),
                            "deepset_text": deepset_texts_raw,
                            "deepset_norm": deepset_norm})

# ---------------------------------------------------------------------------
# 2. Load my frozen evaluation splits
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 2 — Loading frozen PIDS-Bench evaluation splits")
print("=" * 65)

SPLITS: dict[str, Path] = {
    "heldout_test":   REPO_ROOT / "data/pids_bench_v3/test.csv",
    "hard_benign":    REPO_ROOT / "data/pids_bench_v3/eval_subsets/hard_benign_test.csv",
    "domain_ood":     REPO_ROOT / "data/pids_bench_v3/ood/domain_ood.csv",
    "structural_ood": REPO_ROOT / "data/pids_bench_v3/ood/structural_ood.csv",
}

split_dfs: dict[str, pd.DataFrame] = {}
for name, path in SPLITS.items():
    df = pd.read_csv(path)
    print(f"\n  [{name}]")
    print(f"    Path    : {path.relative_to(REPO_ROOT)}")
    print(f"    Rows    : {len(df)}")
    print(f"    Columns : {list(df.columns)}")
    print(f"    Example : {repr(str(df.iloc[0].to_dict())[:120])}")
    split_dfs[name] = df

# Identify text column in each split (all happen to be 'text')
SPLIT_TEXT_COL = "text"
for name, df in split_dfs.items():
    assert SPLIT_TEXT_COL in df.columns, (
        f"Split '{name}' has no column '{SPLIT_TEXT_COL}'. Columns: {list(df.columns)}"
    )
print(f"\nAll splits use text column: '{SPLIT_TEXT_COL}'")

# ---------------------------------------------------------------------------
# 3. Exact-match overlap
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 3 — Exact-match overlap")
print("=" * 65)

deepset_norm_set = set(deepset_norm)

exact_results = []
for name, df in split_dfs.items():
    eval_texts = df[SPLIT_TEXT_COL].astype(str).tolist()
    eval_norm  = [t.lower().strip() for t in eval_texts]
    matches    = sum(1 for t in eval_norm if t in deepset_norm_set)
    pct        = 100.0 * matches / len(eval_norm) if eval_norm else 0.0
    exact_results.append({
        "split": name,
        "eval_rows": len(eval_norm),
        "exact_match_count": matches,
        "exact_match_pct": round(pct, 4),
    })
    print(f"  {name:20}  {len(eval_norm):5} rows  matches={matches:4}  ({pct:.4f}%)")

exact_df = pd.DataFrame(exact_results)
exact_df.to_csv(OUT_DIR / "contamination_exact_match_summary.csv", index=False)
print(f"\nSaved: contamination_exact_match_summary.csv")

# ---------------------------------------------------------------------------
# 4. Near-duplicate overlap via sentence-transformers
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 4 — Near-duplicate overlap (cosine >= 0.90)")
print("=" * 65)

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

THRESHOLD = 0.90
BATCH = 256

print(f"\nEncoding deepset texts ({len(deepset_texts_raw)} rows) ...")
deepset_embs = model.encode(
    deepset_texts_raw, batch_size=BATCH, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True
)
print(f"Deepset embeddings shape: {deepset_embs.shape}")

near_dup_results = []
all_pairs: list[dict] = []

for name, df in split_dfs.items():
    eval_texts = df[SPLIT_TEXT_COL].astype(str).tolist()
    print(f"\n  [{name}] encoding {len(eval_texts)} eval rows ...")
    eval_embs = model.encode(
        eval_texts, batch_size=BATCH, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
    )

    # Cosine similarity matrix: eval_rows x deepset_rows
    # Both are L2-normalised so dot product == cosine similarity
    # Process in chunks of 500 eval rows to keep memory bounded
    CHUNK = 500
    pairs_for_split: list[dict] = []

    for start in range(0, len(eval_texts), CHUNK):
        end = min(start + CHUNK, len(eval_texts))
        chunk_embs = eval_embs[start:end]        # (chunk, D)
        sim_matrix = chunk_embs @ deepset_embs.T  # (chunk, deepset_n)

        # Find pairs above threshold
        rows, cols = np.where(sim_matrix >= THRESHOLD)
        for r, c in zip(rows.tolist(), cols.tolist()):
            eval_idx = start + r
            pairs_for_split.append({
                "split": name,
                "eval_idx": eval_idx,
                "eval_text": eval_texts[eval_idx],
                "deepset_idx": int(c),
                "deepset_text": deepset_texts_raw[int(c)],
                "cosine_similarity": float(sim_matrix[r, c]),
            })

    nd_count = len(pairs_for_split)
    nd_pct   = 100.0 * nd_count / len(eval_texts) if eval_texts else 0.0

    # Unique eval rows flagged (same eval row might match multiple deepset rows)
    unique_eval_flagged = len({p["eval_idx"] for p in pairs_for_split})
    unique_pct = 100.0 * unique_eval_flagged / len(eval_texts) if eval_texts else 0.0

    near_dup_results.append({
        "split": name,
        "eval_rows": len(eval_texts),
        "near_dup_pairs": nd_count,
        "near_dup_pairs_pct": round(nd_pct, 4),
        "unique_eval_rows_flagged": unique_eval_flagged,
        "unique_eval_rows_flagged_pct": round(unique_pct, 4),
    })
    print(f"    near-dup pairs      : {nd_count}  ({nd_pct:.4f}%)")
    print(f"    unique eval rows    : {unique_eval_flagged}  ({unique_pct:.4f}%)")

    all_pairs.extend(pairs_for_split)

nd_df      = pd.DataFrame(near_dup_results)
pairs_df   = pd.DataFrame(all_pairs)

nd_df.to_csv(OUT_DIR / "contamination_near_duplicate_summary.csv", index=False)
pairs_df.to_csv(OUT_DIR / "contamination_near_duplicate_pairs.csv", index=False)
print(f"\nSaved: contamination_near_duplicate_summary.csv")
print(f"Saved: contamination_near_duplicate_pairs.csv  ({len(pairs_df)} total pairs)")

# ---------------------------------------------------------------------------
# 5. Write markdown report
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("STEP 5 — Writing contamination_report.md")
print("=" * 65)

def _df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


lines: list[str] = [
    "# Contamination Check Report",
    "",
    "**External dataset checked:** `deepset/prompt-injections` (Hugging Face, train split)",
    f"**Deepset dataset size:** {len(deepset_texts_raw)} examples",
    "",
    "## Method",
    "",
    "### Exact-match",
    "Text from both datasets was lowercased and whitespace-stripped. "
    "A PIDS-Bench example is flagged if its normalised text appears verbatim "
    "in the normalised deepset set.",
    "",
    "### Near-duplicate (semantic)",
    f"Model: `{MODEL_NAME}`  ",
    f"Threshold: cosine similarity >= {THRESHOLD}  ",
    "Embeddings are L2-normalised; dot product equals cosine similarity. "
    "All eval rows were compared against all deepset rows. "
    "A pair is flagged if similarity >= threshold.",
    "",
    "---",
    "",
    "## Results",
    "",
    "### Exact-match summary",
    "",
    _df_to_md_table(exact_df),
    "",
    "### Near-duplicate summary",
    "",
    _df_to_md_table(nd_df),
    "",
]

# Per-split near-dup detail with example pairs
if not pairs_df.empty:
    lines.append("### Near-duplicate example pairs (up to 5 per split)")
    lines.append("")
    for name in nd_df["split"]:
        sub = pairs_df[pairs_df["split"] == name].sort_values(
            "cosine_similarity", ascending=False
        ).head(5)
        if sub.empty:
            continue
        lines.append(f"#### {name}")
        lines.append("")
        for _, row in sub.iterrows():
            lines.append(
                f"- **sim={row['cosine_similarity']:.4f}**  "
                f"eval[{int(row['eval_idx'])}]: *{str(row['eval_text'])[:120]}*  "
                f"deepset[{int(row['deepset_idx'])}]: *{str(row['deepset_text'])[:120]}*"
            )
        lines.append("")

lines += [
    "---",
    "",
    "## Important Caveat",
    "",
    "This check compares only against the **publicly available** `deepset/prompt-injections` "
    "dataset on Hugging Face. DeBERTa-PI (the `deepset/deberta-v3-base-injection` model) may "
    "have been trained on additional unpublished or private data. Overlap with that private "
    "training data **cannot be ruled out** by this analysis.",
    "",
    "If exact-match or near-duplicate counts are non-zero, those examples should be considered "
    "potentially seen by DeBERTa-PI during training, and results on those examples should be "
    "interpreted with appropriate caution in the paper.",
    "",
    "---",
    "",
    "_Generated by `scripts/contamination_check.py`_",
]

report_path = OUT_DIR / "contamination_report.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: contamination_report.md")

print("\n" + "=" * 65)
print("DONE")
print("=" * 65)
print(exact_df.to_string(index=False))
print()
print(nd_df.to_string(index=False))
