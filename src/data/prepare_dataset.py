# src/data/prepare_dataset.py
from pathlib import Path
import json
import re
from typing import Dict
import argparse

import pandas as pd
from datasets import load_from_disk, DatasetDict

RAW_DIR = Path("data/raw/prompt_injections")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {0: "benign", 1: "injection"}  # adjust if dataset differs

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

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare train/val/test CSVs for prompt-injection datasets.")
    parser.add_argument("--train-split", default="train", help="Dataset split name to use for training.")
    parser.add_argument(
        "--val-split",
        default=None,
        help="Dataset split name to use for validation (defaults to 'validation'/'val' if present, otherwise carved from train)."
    )
    parser.add_argument("--test-split", default="test", help="Dataset split name to use for testing.")
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of the train split to reserve for validation when an explicit val split is unavailable."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DIR}. Run download step first.")

    ds: DatasetDict = load_from_disk(str(RAW_DIR)) # type: ignore
    split_names = list(ds.keys())

    def ensure_split(name: str) -> str:
        if name not in ds:
            raise ValueError(f"Split '{name}' not found. Available splits: {split_names}")
        return name

    train_key = ensure_split(args.train_split)
    test_key = ensure_split(args.test_split)
    if train_key == test_key:
        raise ValueError(f"Train split '{train_key}' and test split '{test_key}' must be different.")

    train_df = to_df(ds[train_key])
    test_df = to_df(ds[test_key])

    val_key = args.val_split
    if val_key is None:
        for candidate in ("validation", "val"):
            if candidate in ds:
                val_key = candidate
                break

    if val_key:
        val_key = ensure_split(val_key)
        if val_key in (train_key, test_key):
            raise ValueError("Validation split must differ from train/test splits.")
        val_df = to_df(ds[val_key])
    else:
        if not 0 < args.val_frac < 0.5:
            raise ValueError("--val-frac must be between 0 and 0.5 when creating a validation split.")
        # Stratified sample from train
        sampled = (
            train_df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(frac=args.val_frac, random_state=args.seed))
        )
        val_indices = sampled.index
        val_df = sampled.reset_index(drop=True)
        train_df = train_df.drop(val_indices).reset_index(drop=True)

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
