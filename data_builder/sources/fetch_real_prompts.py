"""Fetch real prompts from Hugging Face and save to data/real_prompts/.

Injection sources (label=1) — prompt injection only, no jailbreaks:
  1. reshabhs/SPML_Chatbot_Prompt_Injection  (Sharma et al., arXiv 2402.11755)
  2. deepset/prompt-injections               (Haystack, most-cited PI dataset)
  3. qualifire/prompt-injections-benchmark   (real API traffic, gated)

Benign sources (label=0):
  5. tatsu-lab/alpaca                        (Stanford Alpaca, Taori et al. 2023)
  6. alespalla/chatbot_instruction_prompts   (322k instruction-response pairs)
  7. OpenAssistant/oasst1                    (human prompter messages → sharegpt.csv)
  8. databricks/databricks-dolly-15k        (human instructions → stackexchange.csv)
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = _PROJECT_ROOT / "data/real_prompts"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = _PROJECT_ROOT / "data/_cache_hf_datasets"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_col(column_names: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in column_names:
            return c
    return None


def _is_injection(row: dict, label_col: str | None) -> bool:
    if label_col is None:
        return True
    val = row.get(label_col)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return int(val) == 1
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "injection", "attack")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# INJECTION SOURCES (label = 1)
# ──────────────────────────────────────────────────────────────────────────────

def build_spml_injection():
    """SPML Chatbot Prompt Injection — Sharma et al., arXiv 2402.11755.
    Filters rows where Prompt_Injection == 1; uses the User Prompt field.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "reshabhs/SPML_Chatbot_Prompt_Injection",
        split="train",
        cache_dir=str(CACHE_DIR),
    )
    print(f"  SPML columns: {ds.column_names}")

    label_col = _find_col(ds.column_names, ["Prompt Injection", "Prompt injection", "prompt_injection", "label", "injection"])
    text_col  = _find_col(ds.column_names, ["User Prompt", "user_prompt", "text", "prompt", "input"])
    if text_col is None:
        raise ValueError(f"No text column in SPML: {ds.column_names}")

    rows = []
    for r in ds:
        if not _is_injection(r, label_col):
            continue
        t = str(r.get(text_col, "")).strip()
        if t and len(t) >= 10:
            rows.append({"text": t})

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    out = OUTDIR / "spml_injection.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} injection rows -> {out}")


def build_deepset_injection():
    """deepset/prompt-injections — most cited prompt injection dataset in NLP.
    Binary labels: 1 = injection, 0 = benign.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "deepset/prompt-injections",
        split="train",
        cache_dir=str(CACHE_DIR),
    )
    print(f"  deepset columns: {ds.column_names}")

    rows = []
    for r in ds:
        if int(r.get("label", 0)) != 1:
            continue
        t = str(r.get("text", "")).strip()
        if t and len(t) >= 10:
            rows.append({"text": t})

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    out = OUTDIR / "deepset_injection.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} injection rows -> {out}")



def build_qualifire():
    """Qualifire — real-world API traffic prompt injections (gated, already local).
    qualifire.csv already exists in data/real_prompts/ from prior download.
    This function is a no-op: it prints the existing file stats.
    """
    path = OUTDIR / "qualifire.csv"
    if path.exists():
        df = pd.read_csv(path)
        print(f"  Qualifire already exists: {len(df)} rows -> {path} (skipping re-download)")
    else:
        print(f"  WARNING: qualifire.csv not found. Please download manually from"
              f" https://huggingface.co/datasets/qualifire/prompt-injections-benchmark")


# ──────────────────────────────────────────────────────────────────────────────
# BENIGN SOURCES (label = 0)
# ──────────────────────────────────────────────────────────────────────────────

def build_alpaca_benign():
    """Stanford Alpaca (Taori et al., 2023) — 52k diverse human instructions.
    Only uses standalone instructions (no input context required).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=str(CACHE_DIR))
    print(f"  Alpaca columns: {ds.column_names}")

    rows = []
    for r in ds:
        instruction = str(r.get("instruction", "")).strip()
        inp = str(r.get("input", "")).strip()
        if instruction and not inp and len(instruction) >= 10:
            rows.append({"text": instruction})

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    out = OUTDIR / "alpaca_benign.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} benign rows -> {out}")


def build_chatbot_instructions_benign():
    """alespalla/chatbot_instruction_prompts — 322k instruction-response pairs.
    Uses only 10k rows to keep memory manageable.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "alespalla/chatbot_instruction_prompts",
        split="train",
        cache_dir=str(CACHE_DIR),
    )
    print(f"  ChatBot Instructions columns: {ds.column_names}")

    text_col = _find_col(ds.column_names, ["prompt", "instruction", "text", "input", "question"])
    if text_col is None:
        raise ValueError(f"No text column in chatbot_instruction_prompts: {ds.column_names}")

    rows = []
    for r in ds:
        t = str(r.get(text_col, "")).strip()
        if t and len(t) >= 10:
            rows.append({"text": t})
        if len(rows) >= 10000:
            break

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    out = OUTDIR / "chatbot_instructions_benign.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} benign rows -> {out}")


def build_oasst1_benign():
    """OpenAssistant/oasst1 — human prompter messages (saved as sharegpt.csv)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("OpenAssistant/oasst1", split="train", cache_dir=str(CACHE_DIR))
    rows = []
    for row in ds:
        role = str(row.get("role", "")).strip().lower()
        text = str(row.get("text", "")).strip()
        lang = str(row.get("lang", "")).strip().lower()
        if role == "prompter" and lang == "en" and text and len(text) >= 10:
            rows.append({"text": text})
        if len(rows) >= 1500:
            break
    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    df.to_csv(OUTDIR / "sharegpt.csv", index=False)
    print(f"  Saved {len(df)} rows -> {OUTDIR / 'sharegpt.csv'}")


def build_dolly_benign():
    """databricks/databricks-dolly-15k — human-written instructions (saved as stackexchange.csv)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "databricks/databricks-dolly-15k",
        split="train",
        cache_dir=str(CACHE_DIR),
    )
    rows = []
    for row in ds:
        text = str(row.get("instruction", "")).strip()
        if text and len(text) >= 10:
            rows.append({"text": text})
        if len(rows) >= 600:
            break
    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    df.to_csv(OUTDIR / "stackexchange.csv", index=False)
    print(f"  Saved {len(df)} rows -> {OUTDIR / 'stackexchange.csv'}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=== Downloading injection sources (prompt injection only, no jailbreaks) ===")
    build_spml_injection()
    build_deepset_injection()
    build_qualifire()

    print("\n=== Downloading benign sources ===")
    build_alpaca_benign()
    build_chatbot_instructions_benign()
    build_oasst1_benign()
    build_dolly_benign()

    print("\nDone. All files saved to data/real_prompts/")


if __name__ == "__main__":
    main()
