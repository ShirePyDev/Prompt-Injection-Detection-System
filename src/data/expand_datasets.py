# src/data/expand_datasets.py
from __future__ import annotations
from pathlib import Path
import json

import pandas as pd
from datasets import load_dataset, Dataset

OUT_DIR = Path("data/processed_expanded")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# We unify these into a single schema: text (str), label (0=benign, 1=injection), source (str)
DATASETS = [
    {
        "id": "deepset/prompt-injections",
        "source": "deepset_prompt_injections",
        "text_col": "text",
        "label_col": "label",
        "label_type": "int_01",  # 0=benign, 1=injection
    },
    {
        # 5,000 prompts: text + label (jailbreak/benign)
        "id": "qualifire/prompt-injections-benchmark",
        "source": "qualifire_benchmark",
        "text_col": "text",
        "label_col": "label",
        "label_type": "str_jailbreak_benign",  # "jailbreak"->1, "benign"->0
    },
    # You can add more later with similar configs.
]

def normalize_row(text: str, label_raw, cfg) -> dict | None:
    if not isinstance(text, str) or not text.strip():
        return None

    lt = cfg["label_type"]
    if lt == "int_01":
        try:
            lab = int(label_raw)
        except Exception:
            return None
        if lab not in (0, 1):
            return None
        return {"text": text.strip(), "label": lab, "source": cfg["source"]}

    elif lt == "str_jailbreak_benign":
        # map strings to 0/1
        if isinstance(label_raw, str):
            lr = label_raw.strip().lower()
        else:
            lr = str(label_raw).strip().lower()

        if lr == "benign":
            lab = 0
        elif lr == "jailbreak":
            lab = 1
        else:
            # unknown label -> skip
            return None
        return {"text": text.strip(), "label": lab, "source": cfg["source"]}

    # unknown label type -> skip
    return None

def load_and_collect() -> pd.DataFrame:
    all_rows = []

    for cfg in DATASETS:
        ds_id = cfg["id"]
        print(f"🔄 Loading dataset: {ds_id}")
        try:
            ds = load_dataset(ds_id)
        except Exception as e:
            print(f"⚠️  Failed to load {ds_id}: {e}")
            continue

        # Flatten all splits into one list
        for split_name, split in ds.items():    #type: ignore
            if cfg["text_col"] not in split.column_names or cfg["label_col"] not in split.column_names: #type: ignore
                print(f"⚠️  Skipping split {split_name} of {ds_id}: missing columns.")
                continue

            texts = split[cfg["text_col"]]
            labels = split[cfg["label_col"]]
            for t, lab in zip(texts, labels):
                row = normalize_row(t, lab, cfg)
                if row is not None:
                    all_rows.append(row)

        print(f"✅ Collected from {ds_id}: {len(all_rows)} total rows so far.")

    if not all_rows:
        raise RuntimeError("No rows collected from any dataset. Check configs or HF access.")

    df = pd.DataFrame(all_rows)
    # Drop exact duplicate rows (text+label+source)
    df = df.drop_duplicates(subset=["text", "label", "source"]).reset_index(drop=True)
    return df

def main():
    df = load_and_collect()
    csv_path = OUT_DIR / "all_prompts.csv"
    df.to_csv(csv_path, index=False)

    # Simple stats per source + global
    summary = {
        "total_rows": int(len(df)),
        "per_source": {},
    }
    for src, group in df.groupby("source"):
        summary["per_source"][src] = {
            "rows": int(len(group)),
            "benign": int((group["label"] == 0).sum()),
            "injection": int((group["label"] == 1).sum()),
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Expanded dataset saved to:", csv_path)
    print("✅ Summary saved to:", OUT_DIR / "summary.json")
    print("Stats:", summary)

if __name__ == "__main__":
    main()
