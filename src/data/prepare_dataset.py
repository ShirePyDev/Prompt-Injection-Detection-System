# src/data/prepare_dataset.py
from pathlib import Path
import json
import re
from typing import Dict, List

import pandas as pd
from datasets import load_from_disk, DatasetDict

RAW_DIR = Path("data/raw/prompt_injections")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {0: "benign", 1: "injection"}  # adjust if dataset differs
REVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}

def basic_clean(text: str) -> str:
    """Light, safe normalization—no aggressive changes."""
    if text is None:
        return ""
    t = text.strip()
    # collapse whitespace, normalize quotes
    t = re.sub(r"\s+", " ", t)
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    return t

def to_df(split) -> pd.DataFrame:
    cols = split.column_names
    # Expect at least 'text' and 'label'
    if "text" not in cols or "label" not in cols:
        raise ValueError(f"Expected columns ['text','label'], got {cols}")
    df = pd.DataFrame({
        "text": [basic_clean(x) for x in split["text"]],
        "label": split["label"],
    })
    # Sanity: drop empty texts
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    # Ensure labels are 0/1 ints
    df["label"] = df["label"].astype(int)
    return df

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DIR}. Run download step first.")

    ds: DatasetDict = load_from_disk(str(RAW_DIR)) # type: ignore
    # Handle various split names robustly
    split_names = list(ds.keys())
    # Try to map common variants to train/val/test
    train_key = "train" if "train" in split_names else split_names[0]
    test_key  = "test"  if "test"  in split_names else split_names[-1]
    val_key   = "validation" if "validation" in split_names else ("val" if "val" in split_names else None)

    train_df = to_df(ds[train_key])
    test_df  = to_df(ds[test_key])
    if val_key:
        val_df = to_df(ds[val_key])
    else:
        # If no validation split, carve 10% from train for validation (stratified)
        val_df = train_df.groupby("label", group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
        train_df = train_df.drop(val_df.index).reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    # Save CSVs
    train_csv = OUT_DIR / "train.csv"
    val_csv   = OUT_DIR / "val.csv"
    test_csv  = OUT_DIR / "test.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Save label map
    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    # Quick summary
    def stats(df: pd.DataFrame) -> Dict[str, int]:
        return {"rows": len(df), "benign": int((df["label"]==0).sum()), "injection": int((df["label"]==1).sum())}

    summary = {
        "splits": {
            "train": stats(train_df),
            "val":   stats(val_df),
            "test":  stats(test_df),
        },
        "columns": list(train_df.columns),
        "label_map": LABEL_MAP,
        "examples": {
            "train_head": train_df.head(2).to_dict(orient="records"),
            "val_head":   val_df.head(2).to_dict(orient="records"),
            "test_head":  test_df.head(2).to_dict(orient="records"),
        }
    }
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(json.dumps(summary, indent=2))
    print("✅ Saved:", train_csv, val_csv, test_csv)
    print("✅ Label map:", OUT_DIR / "label_map.json")
    print("✅ Summary:", OUT_DIR / "summary.txt")
    print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))

if __name__ == "__main__":
    main()
