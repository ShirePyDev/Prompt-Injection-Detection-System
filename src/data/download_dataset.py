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
        print(f"{split}: {len(ds[split])} rows")
        print("Columns:", ds[split].column_names)
        print("Example:", ds[split][110])

if __name__ == "__main__":
    main()
