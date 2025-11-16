# src/data/merge_datasets.py
from __future__ import annotations
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
SMALL_DIR = Path("data/processed")  # old 662-sample split
LARGE_PATH = Path("data/processed_expanded/all_prompts.csv")  # 5662 real samples
SYN_DIR = Path("data/processed_expanded/synthetic")  # synthetic batches
OUT_DIR = Path("data/final_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_small_base() -> pd.DataFrame:
    """
    Load old train/val/test (the ~662 rows) and merge into one DF.
    """
    dfs = []
    for name in ["train", "val", "test"]:
        p = SMALL_DIR / f"{name}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["source"] = df.get("source", "small_base")
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No small-base splits found in data/processed.")
    return pd.concat(dfs, ignore_index=True)


def load_large_base() -> pd.DataFrame:
    if not LARGE_PATH.exists():
        raise FileNotFoundError(f"Missing large dataset at {LARGE_PATH}")
    df = pd.read_csv(LARGE_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{LARGE_PATH} must have columns ['text','label']")
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["source"] = df.get("source", "large_base")
    return df

def load_manual_benign() -> pd.DataFrame:
    manual_path = Path("data/manual/hard_benign.csv")
    if not manual_path.exists():
        print("No manual benign file found.")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = pd.read_csv(manual_path)
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df["source"] = df.get("source", "manual_hard_benign")
    return df



def load_synthetics() -> pd.DataFrame:
    """
    Load ALL synthetic CSVs from SYN_DIR (batch1, batch2, etc.).
    """
    csvs = list(SYN_DIR.glob("*.csv"))
    if not csvs:
        return pd.DataFrame(columns=["text", "label", "source"])

    dfs = []
    for p in csvs:
        df = pd.read_csv(p)
        if "text" not in df.columns:
            continue
        df["text"] = df["text"].astype(str)
        # If label missing, assume injection (1)
        if "label" not in df.columns:
            df["label"] = 1
        df["label"] = df["label"].astype(int)
        # Source = file name
        df["source"] = df.get("source", p.stem)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["text", "label", "source"])
    return pd.concat(dfs, ignore_index=True)


def print_stats(name: str, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    stats = {
        "rows": int(len(df)),
        "benign": int((df["label"] == 0).sum()),
        "injection": int((df["label"] == 1).sum()),
    }
    print(
        f"{name}: {stats['rows']} rows "
        f"(benign={stats['benign']}, injection={stats['injection']})"
    )
    return {name: stats}


def main():
    small_df = load_small_base()
    large_df = load_large_base()
    synth_df = load_synthetics()
    manual_df = load_manual_benign()


    print("=== SOURCE STATS (before merge) ===")
    stats_all: Dict[str, Dict[str, int]] = {}
    stats_all.update(print_stats("small_base", small_df))
    stats_all.update(print_stats("large_base", large_df))
    if len(synth_df) > 0:
        stats_all.update(print_stats("synthetic", synth_df))
    else:
        print("synthetic: 0 rows")

    # Combine all
    df_all = pd.concat(
    [small_df, large_df, synth_df, manual_df],
    ignore_index=True
)


    # Basic cleaning
    df_all["text"] = df_all["text"].astype(str).str.strip()
    df_all = df_all[df_all["text"].str.len() > 0].reset_index(drop=True)
    df_all["label"] = df_all["label"].astype(int)

    # Deduplicate by text
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed = before - len(df_all)
    print(f"Deduped: removed {removed} duplicate texts.")
    print(f"Final combined rows (before split): {len(df_all)}")

    # Stratified train/val/test split (70/15/15)
    train_tmp, test_df = train_test_split(
        df_all,
        test_size=0.15,
        stratify=df_all["label"],
        random_state=42,
    )
    train_df, val_df = train_test_split(
        train_tmp,
        test_size=0.15 / 0.85,  # final val ≈ 15%
        stratify=train_tmp["label"],
        random_state=42,
    )

    # Save final splits (drop 'source' so they match previous format)
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = OUT_DIR / f"{split_name}.csv"
        df_out = df[["text", "label"]].copy()
        df_out.to_csv(out_path, index=False)
        print(f"Saved {split_name}: {out_path} (rows={len(df_out)})")

    print("✅ Merge complete. New final_dataset ready.")


if __name__ == "__main__":
    main()
