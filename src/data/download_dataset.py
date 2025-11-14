# src/data/download_dataset.py
from datasets import load_dataset, DatasetDict
from pathlib import Path   # ✅ this line defines Path

OUT_DIR = Path("data/raw/prompt_injections")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds: DatasetDict = load_dataset("deepset/prompt-injections") # type: ignore

    ds.save_to_disk(str(OUT_DIR))

    print("✅ Dataset downloaded and saved to:", OUT_DIR)
    print("Splits:", list(ds.keys()))
    for split in ds.keys():
        split_ds = ds[split]
        split_len = len(split_ds)
        print(f"{split}: {split_len} rows")
        print("Columns:", split_ds.column_names)
        if split_len:
            # Keep inspection safe on tiny splits by grabbing the first row
            print("Example:", split_ds[0])
        else:
            print("Example: <empty split>")

if __name__ == "__main__":
    main()
