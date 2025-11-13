# src/baselines/rule_based.py
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path("data/processed")
OUT_DIR = Path("outputs/rule_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Heuristic patterns (expandable) ---
KEYWORDS = [
    r"\bignore (all|any|previous|above) (instructions|directions)\b",
    r"\bdisregard (previous|prior) (instructions|rules)\b",
    r"\boverride\b", r"\bbypass\b", r"\bexploit\b",
    r"\b(as|act) (a|as) (developer|system|admin|jailbreak)\b",
    r"\bDAN\b", r"\bdeveloper mode\b", r"\bimpersonate\b",
    r"\breveal (your )?system prompt\b", r"\bwhat is your system prompt\b",
    r"\brepeat after me\b", r"\bverbatim\b",
    r"\bforget (your|all) rules\b", r"\bself\-destruct\b",
    r"\bprompt injection\b", r"\bignore safety\b", r"\bturn off (safety|guardrails)\b",
    r"\bsudo\b", r"rm\s-\w*\s*/",
]
URLISH = r"https?://|www\."
CODEBLOCK = r"```"

KEYWORD_RE = re.compile("|".join(KEYWORDS), flags=re.IGNORECASE)
URL_RE = re.compile(URLISH, flags=re.IGNORECASE)
CODE_RE = re.compile(CODEBLOCK, flags=0)

def rule_score(text: str) -> int:
    """Return 1 if likely injection, else 0."""
    if not isinstance(text, str) or not text.strip():
        return 0
    t = text.strip()
    if KEYWORD_RE.search(t):
        return 1
    hits = 0
    hits += 1 if URL_RE.search(t) else 0
    hits += 1 if CODE_RE.search(t) else 0
    hits += 1 if t.lower().count("ignore") >= 2 else 0
    return 1 if hits >= 2 else 0

def evaluate_split(csv_path: Path, split_name: str) -> Path:
    df = pd.read_csv(csv_path)
    y_true = df["label"].astype(int).values
    y_pred = [rule_score(x) for x in df["text"].astype(str).tolist()]

    report = classification_report(
        y_true, y_pred, target_names=["benign","injection"], digits=4. # type: ignore
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0,1]) # type: ignore

    out_txt = OUT_DIR / f"{split_name}_report.txt"
    with open(out_txt, "w") as f:
        f.write(report) # type: ignore
        f.write("\nConfusion Matrix [[TN, FP],[FN, TP]]:\n")
        f.write(str(cm))
    return out_txt

def run():
    val_csv = DATA_DIR / "val.csv"
    test_csv = DATA_DIR / "test.csv"
    if not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Expected data/processed/val.csv and test.csv. Run Step 3 first.")

    val_out = evaluate_split(val_csv, "val")
    test_out = evaluate_split(test_csv, "test")
    print("✅ Baseline complete.")
    print("Saved reports:")
    print(" -", val_out)
    print(" -", test_out)

if __name__ == "__main__":
    run()
