# src/data/merge_all_datasets.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
SMALL_PATH   = Path("data/processed/train.csv")                    # small base
LARGE_PATH   = Path("data/processed_expanded/all_prompts.csv")     # large real dataset
SYNTH_DIR    = Path("data/processed_expanded/synthetic")            # synthetic injections
MANUAL_DIR   = Path("data/manual_benign")                           # +6000 benign
OUT_DIR      = Path("data/final_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_small():
    df = pd.read_csv(SMALL_PATH)
    df["source"] = "small_base"
    return df[["text", "label", "source"]]

def load_large():
    df = pd.read_csv(LARGE_PATH)
    df["source"] = "large_base"
    return df[["text", "label", "source"]]

def load_synthetic():
    dfs = []
    for p in SYNTH_DIR.glob("*.csv"):
        df = pd.read_csv(p)
        df["label"] = 1
        df["source"] = p.stem
        dfs.append(df[["text", "label", "source"]])
    return pd.concat(dfs, ignore_index=True)

def load_manual():
    dfs = []
    for p in MANUAL_DIR.glob("*.csv"):
        df = pd.read_csv(p)
        df["label"] = 0
        df["source"] = p.stem
        dfs.append(df[["text", "label", "source"]])
    return pd.concat(dfs, ignore_index=True)

def main():
    small  = load_small()
    large  = load_large()
    synth  = load_synthetic()
    manual = load_manual()

    print("small:", len(small))
    print("large:", len(large))
    print("synth:", len(synth))
    print("manual:", len(manual))

    combined = pd.concat([small, large, synth, manual], ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    after = len(combined)

    print("deduped:", before - after)
    print("total:", after)
    print("label counts:", combined["label"].value_counts().to_dict())

    # 70/15/15 split
    train, temp = train_test_split(combined, test_size=0.30, stratify=combined["label"], random_state=42)
    val, test   = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)

    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    print("Saved train:", len(train))
    print("Saved val:", len(val))
    print("Saved test:", len(test))

if __name__ == "__main__":
    main()
