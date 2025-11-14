from pathlib import Path
import pandas as pd
import json

BASE_SPLITS_DIR = Path("data/processed")  # 662 rows (old small set)
REAL_LARGE_FILE = Path("data/processed_expanded/all_prompts.csv")  # 5662 rows
SYN_FILE = Path("data/processed_expanded/synthetic/synthetic_injections_llama3.csv")

OUT_DIR = Path("data/final_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Load small original splits (662 total)
    train = pd.read_csv(BASE_SPLITS_DIR / "train.csv")
    val   = pd.read_csv(BASE_SPLITS_DIR / "val.csv")
    test  = pd.read_csv(BASE_SPLITS_DIR / "test.csv")

    small_original = pd.concat([train, val, test], ignore_index=True)
    print("Small original dataset:", len(small_original))

    # Load large real dataset (5662)
    big_real = pd.read_csv(REAL_LARGE_FILE)
    print("Large real dataset:", len(big_real))

    # Load synthetic injections (6638)
    syn = pd.read_csv(SYN_FILE)
    syn["label"] = 1
    print("Synthetic:", len(syn))

    # Combine all together
    full = pd.concat([small_original, big_real, syn], ignore_index=True)

    # Clean duplicates
    before = len(full)
    full = full.drop_duplicates(subset=["text"]).reset_index(drop=True)
    after = len(full)
    print(f"Deduped: removed {before - after} duplicates.")

    print("Final dataset size:", after)

    # Shuffle
    full = full.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Stratified train/val/test split
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        full,
        test_size=0.30,
        stratify=full["label"],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42
    )

    # Save splits
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    summary = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "total": len(full),
        "label_distribution": full["label"].value_counts().to_dict()
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved final dataset to:", OUT_DIR)

if __name__ == "__main__":
    main()
