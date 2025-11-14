# src/data/split_expanded.py
from __future__ import annotations
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

IN_PATH = Path("data/processed_expanded/all_prompts.csv")
OUT_DIR = Path("data/processed_expanded")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}. Run expand_datasets step first.")

    df = pd.read_csv(IN_PATH)

    # Basic sanity checks
    required_cols = {"text", "label", "source"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Expected columns {required_cols}, got {df.columns}")

    # Shuffle once for randomness
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Stratified split: 80% train, 10% val, 10% test (approx)
    train_temp, test = train_test_split(
        df,
        test_size=0.10,
        stratify=df["label"],
        random_state=42,
    )
    train, val = train_test_split(
        train_temp,
        test_size=0.1111,   # 0.1111 of 90% ≈ 10% of total
        stratify=train_temp["label"],
        random_state=42,
    )

    # Reset indices
    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    # Save to CSV
    train_path = OUT_DIR / "train.csv"
    val_path   = OUT_DIR / "val.csv"
    test_path  = OUT_DIR / "test.csv"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    # Build a small summary
    def stats(df_part):
        return {
            "rows": int(len(df_part)),
            "benign": int((df_part["label"] == 0).sum()),
            "injection": int((df_part["label"] == 1).sum()),
        }

    summary = {
        "total_rows": int(len(df)),
        "splits": {
            "train": stats(train),
            "val":   stats(val),
            "test":  stats(test),
        }
    }

    with open(OUT_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Expanded splits saved:")
    print(" -", train_path)
    print(" -", val_path)
    print(" -", test_path)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
