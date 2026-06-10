"""Build evaluation subsets (balanced_subtype_test, obfuscated_attacks) from main corpus."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

INJECTION_TYPES = [
    "direct_override",
    "roleplay_attack",
    "contextual_manipulation",
    "data_exfiltration",
    "encoded_attack",
    "tool_injection",
]
EVAL_SUBDIR = "eval_subsets"


def load_splits(data_dir: Path, splits: list[str] | None = None) -> pd.DataFrame:
    splits = splits or ["train", "val", "test"]
    dfs = []
    for split in splits:
        p = data_dir / f"{split}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["_split"] = split
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No splits found in {data_dir}")
    return pd.concat(dfs, ignore_index=True)


def build_balanced_subtype(
    df: pd.DataFrame,
    n_benign: int = 3000,
    n_per_subtype: int = 500,
) -> pd.DataFrame:
    """Build balanced subtype test from df. No overlap with train (sample from test only)."""
    rows = []
    benign = df[df["label"] == 0]
    if len(benign) >= n_benign:
        rows.append(benign.sample(n=n_benign, random_state=42))
    else:
        rows.append(benign)
    for at in INJECTION_TYPES:
        sub = df[(df["label"] == 1) & (df["attack_type"] == at)]
        if len(sub) >= n_per_subtype:
            rows.append(sub.sample(n=n_per_subtype, random_state=42))
        elif len(sub) > 0:
            rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out["split"] = "balanced_subtype_test"
    return out


def build_obfuscated_attacks(df: pd.DataFrame) -> pd.DataFrame:
    """Injections where obfuscation != 'none'."""
    sub = df[(df["label"] == 1) & (df["obfuscation"] != "none")]
    sub = sub.copy()
    sub["split"] = "obfuscated_attacks"
    return sub


def run(output_dir: Path, eval_splits_config: dict | None = None) -> None:
    """
    Build eval subsets from train/val/test in output_dir.

    IMPORTANT: balanced_subtype_test is sampled from the TEST split only to prevent
    train/eval leakage. Always run this AFTER the main paraphrase build is complete.

    eval_splits_config: e.g. {"balanced_subtype_test": 2400}
      → n_benign = 2400 // 2 = 1200
      → n_per_subtype = 1200 // 6 = 200
    """
    df = load_splits(output_dir)  # train+val+test for obfuscated_attacks
    out_dir = output_dir / EVAL_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = eval_splits_config or {}
    n_benign = 3000
    n_per_subtype = 500
    if "balanced_subtype_test" in cfg:
        total = int(cfg["balanced_subtype_test"])
        n_benign = total // 2
        n_per_subtype = total // 2 // 6

    # balanced_subtype_test: sample from TEST ONLY (no overlap with train)
    test_df = load_splits(output_dir, splits=["test"])
    balanced = build_balanced_subtype(test_df, n_benign=n_benign, n_per_subtype=n_per_subtype)
    balanced.to_csv(out_dir / "balanced_subtype_test.csv", index=False)
    print(f"Saved balanced_subtype_test: {out_dir / 'balanced_subtype_test.csv'} ({len(balanced)} rows)")

    obf = build_obfuscated_attacks(df)  # from train+val+test
    obf.to_csv(out_dir / "obfuscated_attacks.csv", index=False)
    print(f"Saved obfuscated_attacks: {out_dir / 'obfuscated_attacks.csv'} ({len(obf)} rows)")
