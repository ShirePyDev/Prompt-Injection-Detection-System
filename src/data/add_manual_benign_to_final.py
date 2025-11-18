# src/data/add_manual_benign_to_final.py
from __future__ import annotations
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
SMALL_BASE = Path("data/processed/train.csv")  # original small train from earlier step
LARGE_BASE = Path("data/processed_expanded/all_prompts.csv")  # 5.6k real prompts
SYNTHETIC_DIR = Path("data/processed_expanded/synthetic")      # all synthetic injections
MANUAL_BENIGN_DIR = Path("data/manual_benign")                 # manual + templated benign
OUT_DIR = Path("data/final_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_small_base() -> pd.DataFrame:
    if not SMALL_BASE.exists():
        raise FileNotFoundError(f"Missing {SMALL_BASE}")
    df = pd.read_csv(SMALL_BASE)
    df["source"] = df.get("source", "small_base")
    return df[["text", "label", "source"]]

def load_large_base() -> pd.DataFrame:
    if not LARGE_BASE.exists():
        raise FileNotFoundError(f"Missing {LARGE_BASE}")
    df = pd.read_csv(LARGE_BASE)
    df["source"] = df.get("source", "large_base")
    return df[["text", "label", "source"]]

def load_synthetic() -> pd.DataFrame:
    rows = []
    if not SYNTHETIC_DIR.exists():
        return pd.DataFrame(columns=["text", "label", "source"])
    for p in SYNTHETIC_DIR.glob("*.csv"):
        df = pd.read_csv(p)
        df["source"] = df.get("source", p.stem)
        rows.append(df[["text", "label", "source"]])
    if not rows:
        return pd.DataFrame(columns=["text", "label", "source"])
    df_all = pd.concat(rows, ignore_index=True)
    # Guarantee injections
    df_all["label"] = 1
    return df_all

def load_manual_benign() -> pd.DataFrame:
    if not MANUAL_BENIGN_DIR.exists():
        return pd.DataFrame(columns=["text", "label", "source"])
    rows = []
    for p in MANUAL_BENIGN_DIR.glob("*.csv"):
        df = pd.read_csv(p)
        df["label"] = 0  # make sure
        df["source"] = df.get("source", p.stem)
        rows.append(df[["text", "label", "source"]])
    if not rows:
        return pd.DataFrame(columns=["text", "label", "source"])
    return pd.concat(rows, ignore_index=True)

def stratified_split(df: pd.DataFrame, seed: int = 42) -> Dict[str, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=seed,
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=seed,
        shuffle=True,
    )
    return {"train": train_df, "val": val_df, "test": test_df}

def main():
    small = load_small_base()
    large = load_large_base()
    synth = load_synthetic()
    manual_benign = load_manual_benign()

    print("=== SOURCE STATS (before merge) ===")
    for name, df in [
        ("small_base", small),
        ("large_base", large),
        ("synthetic", synth),
        ("manual_benign", manual_benign),
    ]:
        if len(df) == 0:
            print(f"{name}: 0 rows")
            continue
        print(
            f"{name}: {len(df)} rows (benign={(df['label']==0).sum()}, "
            f"injection={(df['label']==1).sum()})"
        )

    combined = pd.concat([small, large, synth, manual_benign], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    after = len(combined)
    print(f"Deduped: removed {before - after} duplicate texts.")

    print(f"Final combined rows (before split): {len(combined)}")
    print("Label counts:", combined["label"].value_counts().to_dict())

    splits = stratified_split(combined, seed=42)

    for name, df_split in splits.items():
        out_path = OUT_DIR / f"{name}.csv"
        df_split.to_csv(out_path, index=False)
        print(f"Saved {name}: {out_path} (rows={len(df_split)})")

    print("✅ Merge complete. New final_dataset ready.")

if __name__ == "__main__":
    main()
