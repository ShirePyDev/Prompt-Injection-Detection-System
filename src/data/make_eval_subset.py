import pandas as pd
from pathlib import Path

def main():
    data_dir = Path("data/final_dataset")

    # Load splits
    train = pd.read_csv(data_dir / "train.csv")
    val   = pd.read_csv(data_dir / "val.csv")
    test  = pd.read_csv(data_dir / "test.csv")

    # Merge to one big DF (≈25k rows)
    full_df = pd.concat([train, val, test], ignore_index=True)

    # --- Adjust this if your label column has a different name ---
    LABEL_COL = "label"  # e.g. 0 = benign, 1 = injection

    benign_df    = full_df[full_df[LABEL_COL] == 0]
    injection_df = full_df[full_df[LABEL_COL] == 1]

    print("Benign total   :", len(benign_df))
    print("Injection total:", len(injection_df))

    # We want 1,500 per class -> total 3,000
    n_per_class = 1500

    if len(benign_df) < n_per_class or len(injection_df) < n_per_class:
        raise ValueError("Not enough samples in one of the classes to sample 1500.")

    benign_sample    = benign_df.sample(n=n_per_class, random_state=42)
    injection_sample = injection_df.sample(n=n_per_class, random_state=42)

    # Combine and shuffle
    eval_df = pd.concat([benign_sample, injection_sample], ignore_index=True)
    eval_df = eval_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_dir = Path("data/eval_subset")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_3k_balanced.csv"

    eval_df.to_csv(out_path, index=False)
    print(f"Saved BALANCED subset with {len(eval_df)} rows to {out_path}")

if __name__ == "__main__":
    main()
