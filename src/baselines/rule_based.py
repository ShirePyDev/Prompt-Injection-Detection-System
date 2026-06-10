# src/baselines/rule_based.py
from __future__ import annotations
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for the rule-based baseline. "
        "Install project dependencies via `pip install -r requirements.txt`."
    ) from exc

DATA_DIR = Path("data/pids_bench_v3")
EVAL_DIR = DATA_DIR / "eval_subsets"
OOD_DIR  = DATA_DIR / "ood"
OUT_DIR  = Path("outputs/rule_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Heuristic patterns ---
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
URLISH    = r"https?://|www\."
CODEBLOCK = r"```"

KEYWORD_RE = re.compile("|".join(KEYWORDS), flags=re.IGNORECASE)
URL_RE     = re.compile(URLISH, flags=re.IGNORECASE)
CODE_RE    = re.compile(CODEBLOCK, flags=0)


def rule_score(text: str) -> int:
    """Return 1 if likely injection, else 0."""
    if not isinstance(text, str) or not text.strip():
        return 0
    t = text.strip()
    if KEYWORD_RE.search(t):
        return 1
    hits  = 0
    hits += 1 if URL_RE.search(t) else 0
    hits += 1 if CODE_RE.search(t) else 0
    hits += 1 if t.lower().count("ignore") >= 2 else 0
    return 1 if hits >= 2 else 0


# ── Shared evaluation helpers ────────────────────────────────────────────────

def _save_report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    report  = classification_report(
        y_true, y_pred, target_names=["benign", "injection"], digits=4  # type: ignore
    )
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1])  # type: ignore
    out_txt = OUT_DIR / f"{name}_report.txt"
    with open(out_txt, "w") as f:
        f.write(report)  # type: ignore
        f.write("\nConfusion Matrix [[TN, FP],[FN, TP]]:\n")
        f.write(str(cm))
    return out_txt


def _evaluate_split(csv_path: Path, split_name: str) -> tuple[Path, dict]:
    df     = pd.read_csv(csv_path)
    y_true = df["label"].astype(int).values
    y_pred = np.array([rule_score(x) for x in df["text"].astype(str).tolist()])
    out    = _save_report(split_name, y_true, y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n  = len(y_true)
    sdiv = lambda a, b: a / b if b else 0.0
    metrics = {
        "n":         n,
        "f1":        round(sdiv(2 * tp, 2 * tp + fp + fn), 4),
        "precision": round(sdiv(tp, tp + fp), 4),
        "recall":    round(sdiv(tp, tp + fn), 4),
        "fpr":       round(sdiv(fp, fp + tn), 4),
        "fnr":       round(sdiv(fn, fn + tp), 4),
    }
    return out, metrics


def _evaluate_hard_benign(csv_path: Path) -> dict:
    """
    hard_benign_test is 100% benign (label=0).
    FPR = fraction the rule flags as injection.
    No threshold tuning — the rule is binary by design.
    """
    df        = pd.read_csv(csv_path)
    texts     = df["text"].astype(str).tolist()
    preds     = np.array([rule_score(t) for t in texts])
    n         = len(preds)
    n_flagged = int(preds.sum())
    result    = {
        "n":         n,
        "n_flagged": n_flagged,
        "fpr":       round(n_flagged / n, 6) if n else 0.0,
        "note":      "Rule is binary (no threshold). FPR = flagged / total.",
    }
    with open(OUT_DIR / "hard_benign_fpr.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def _evaluate_obfuscated_attacks(csv_path: Path) -> dict:
    """
    obfuscated_attacks.csv is 100% injection (label=1).
    Metric: recall (detection rate) per obfuscation type.
    No negatives in this set — FPR is undefined and not computed.
    """
    df    = pd.read_csv(csv_path).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    preds = np.array([rule_score(t) for t in df["text"].tolist()])

    n          = len(df)
    n_detected = int(preds.sum())

    per_obf: dict = {}
    for obf, group in df.groupby("obfuscation"):
        idx        = group.index.tolist()
        grp_preds  = preds[idx]
        grp_n      = len(idx)
        grp_det    = int(grp_preds.sum())
        per_obf[str(obf)] = {
            "n":              grp_n,
            "n_detected":     grp_det,
            "recall":         round(grp_det / grp_n, 6) if grp_n else 0.0,
        }

    result = {
        "n":              n,
        "n_detected":     n_detected,
        "overall_recall": round(n_detected / n, 6) if n else 0.0,
        "overall_fnr":    round(1 - n_detected / n, 6) if n else 1.0,
        "per_obfuscation": per_obf,
    }
    with open(OUT_DIR / "obfuscated_attacks_report.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def _evaluate_balanced_subtype(csv_path: Path) -> dict:
    """
    balanced_subtype_test.csv has both benign and injection rows with
    attack_type labels. Reports overall + per-subtype classification metrics.
    """
    df     = pd.read_csv(csv_path).reset_index(drop=True)
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    preds  = np.array([rule_score(t) for t in df["text"].tolist()])
    y_true = df["label"].values

    out = _save_report("balanced_subtype", y_true, preds)

    tp = int(((y_true == 1) & (preds == 1)).sum())
    tn = int(((y_true == 0) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    fn = int(((y_true == 1) & (preds == 0)).sum())
    n  = len(y_true)
    sdiv = lambda a, b: a / b if b else 0.0

    per_subtype: dict = {}
    if "attack_type" in df.columns:
        for atype, group in df.groupby("attack_type"):
            idx       = group.index.tolist()
            g_true    = y_true[idx]
            g_pred    = preds[idx]
            g_tp      = int(((g_true == 1) & (g_pred == 1)).sum())
            g_tn      = int(((g_true == 0) & (g_pred == 0)).sum())
            g_fp      = int(((g_true == 0) & (g_pred == 1)).sum())
            g_fn      = int(((g_true == 1) & (g_pred == 0)).sum())
            g_n       = len(g_true)
            per_subtype[str(atype)] = {
                "n":       g_n,
                "f1":      round(sdiv(2 * g_tp, 2 * g_tp + g_fp + g_fn), 4),
                "recall":  round(sdiv(g_tp, g_tp + g_fn), 4),
                "fpr":     round(sdiv(g_fp, g_fp + g_tn), 4),
            }

    result = {
        "n":            n,
        "f1":           round(sdiv(2 * tp, 2 * tp + fp + fn), 4),
        "precision":    round(sdiv(tp, tp + fp), 4),
        "recall":       round(sdiv(tp, tp + fn), 4),
        "fpr":          round(sdiv(fp, fp + tn), 4),
        "fnr":          round(sdiv(fn, fn + tp), 4),
        "per_subtype":  per_subtype,
    }
    with open(OUT_DIR / "balanced_subtype_report.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def run() -> None:
    # ── File existence checks ─────────────────────────────────────────────────
    required = {
        "val":                DATA_DIR / "val.csv",
        "test":               DATA_DIR / "test.csv",
        "hard_benign_test":   EVAL_DIR / "hard_benign_test.csv",
        "obfuscated_attacks": EVAL_DIR / "obfuscated_attacks.csv",
        "balanced_subtype":   EVAL_DIR / "balanced_subtype_test.csv",
        "domain_ood":         OOD_DIR  / "domain_ood.csv",
        "structural_ood":     OOD_DIR  / "structural_ood.csv",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    # ── No training step — rule is stateless, zero leakage risk ──────────────

    _, val_m   = _evaluate_split(required["val"],  "val")
    test_out, test_m = _evaluate_split(required["test"], "test")
    hb_res     = _evaluate_hard_benign(required["hard_benign_test"])
    obf_res    = _evaluate_obfuscated_attacks(required["obfuscated_attacks"])
    sub_res    = _evaluate_balanced_subtype(required["balanced_subtype"])
    _, d_m     = _evaluate_split(required["domain_ood"],     "domain_ood")
    _, s_m     = _evaluate_split(required["structural_ood"], "structural_ood")

    summary = {
        "model":          "rule_based",
        "threshold_note": "Rule is binary — no threshold tuning.",
        "val_results":    val_m,
        "test_results":   test_m,
        "hard_benign_test_fpr": {
            "fpr":       hb_res["fpr"],
            "n_flagged": hb_res["n_flagged"],
            "n":         hb_res["n"],
        },
        "obfuscated_attacks": {
            "overall_recall": obf_res["overall_recall"],
            "overall_fnr":    obf_res["overall_fnr"],
        },
        "balanced_subtype": {
            "f1":    sub_res["f1"],
            "recall": sub_res["recall"],
            "fpr":   sub_res["fpr"],
        },
        "ood_results": {
            "domain_ood":     d_m,
            "structural_ood": s_m,
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Rule-based baseline complete.")
    print(f"  test F1                  : {test_m['f1']:.4f}")
    print(f"  hard_benign FPR          : {hb_res['fpr']:.4f}  ({hb_res['n_flagged']}/{hb_res['n']})")
    print(f"  obfuscated recall        : {obf_res['overall_recall']:.4f}")
    print(f"  domain_ood F1            : {d_m['f1']:.4f}")
    print(f"  structural_ood F1        : {s_m['f1']:.4f}")
    print(f"  summary → {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    run()
