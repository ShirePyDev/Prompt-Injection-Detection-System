"""
scripts/consolidate_results.py

Reads summary.json from each completed baseline run and prints a single
comparison table across all failure-mode axes.

Usage (from project root):
    python scripts/consolidate_results.py
"""
from __future__ import annotations
import json
from pathlib import Path

MODELS = [
    ("Rule-based",  Path("outputs/rule_baseline/summary.json")),
    ("TF-IDF+LR",   Path("outputs/tfidf_baseline/summary.json")),
    ("DistilBERT",  Path("outputs/distilbert_baseline/summary.json")),
    ("DeBERTa-v3",  Path("outputs/deberta_v3/summary.json")),
]

# ── Extraction helpers ────────────────────────────────────────────────────────

def _get(d: dict, *keys, default="—"):
    """Safely traverse nested dict keys."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d if d != "" else default


def _fmt(val, default="—") -> str:
    if val == default or val is None:
        return default
    try:
        return f"{float(val):.4f}"
    except (ValueError, TypeError):
        return str(val)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    # Reject stale summary.json files that used the old schema (pre-rewrite).
    # New schema always has 'test_results' and 'hard_benign_test_fpr'.
    if "test_results" not in data or "hard_benign_test_fpr" not in data:
        return None
    return data


# ── Table printer ─────────────────────────────────────────────────────────────

def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    col_w = [max(len(h), max((len(r[i]) for r in rows), default=0))
             for i, h in enumerate(headers)]
    sep   = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt   = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print(f"\n{'─' * len(sep)}")
    print(f"  {title}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    summaries: dict[str, dict | None] = {}
    for name, path in MODELS:
        summaries[name] = _load(path)
        if not path.exists():
            status = "NOT FOUND — run the baseline first"
        elif summaries[name] is None:
            status = "STALE — re-run the baseline (old schema)"
        else:
            status = "OK"
        print(f"  [{status:^40}]  {path}")

    print()

    # ── Table 1: In-distribution test set ────────────────────────────────────
    rows = []
    for name, _ in MODELS:
        s = summaries[name]
        if s is None:
            rows.append([name, "—", "—", "—", "—", "—"])
            continue
        # tuned threshold results take priority; fall back to flat test_results
        t = s.get("test_results", {}).get("val_tuned_threshold")
        if not isinstance(t, dict):
            t = s.get("test_results", {})
        rows.append([
            name,
            _fmt(_get(t, "f1")),
            _fmt(_get(t, "precision")),
            _fmt(_get(t, "recall")),
            _fmt(_get(t, "fpr")),
            _fmt(_get(t, "fnr")),
        ])
    _print_table(
        "TABLE 1 — In-distribution test set  (val-tuned threshold)",
        ["Model", "F1", "Precision", "Recall", "FPR", "FNR"],
        rows,
    )

    # ── Table 2: Over-defense (hard_benign FPR) ───────────────────────────────
    rows = []
    for name, _ in MODELS:
        s = summaries[name]
        if s is None:
            rows.append([name, "—", "—", "—"])
            continue
        hb = _get(s, "hard_benign_test_fpr")
        if not isinstance(hb, dict):
            rows.append([name, "—", "—", "—"])
            continue
        fpr_tuned   = _fmt(_get(hb, "fpr_tuned",   default=_get(hb, "fpr")))
        fpr_default = _fmt(_get(hb, "fpr_default_0.5", default="—"))
        n_flagged   = str(_get(hb, "n_flagged_tuned", default=_get(hb, "n_flagged")))
        rows.append([name, fpr_default, fpr_tuned, n_flagged])
    _print_table(
        "TABLE 2 — Over-defense: hard_benign_test FPR  (all 1,472 rows are benign)",
        ["Model", "FPR @ 0.5", "FPR @ tuned", "# flagged (tuned)"],
        rows,
    )

    # ── Table 3: Obfuscation blindness ────────────────────────────────────────
    rows = []
    for name, _ in MODELS:
        s = summaries[name]
        if s is None:
            rows.append([name, "—", "—"])
            continue
        ob = _get(s, "obfuscated_attacks")
        if not isinstance(ob, dict):
            rows.append([name, "—", "—"])
            continue
        rows.append([
            name,
            _fmt(_get(ob, "overall_recall")),
            _fmt(_get(ob, "overall_fnr")),
        ])
    _print_table(
        "TABLE 3 — Obfuscation blindness: obfuscated_attacks recall  (405 injections; frozen eval set)",
        ["Model", "Recall (detect rate)", "FNR (miss rate)"],
        rows,
    )

    # ── Table 4: Domain OOD ───────────────────────────────────────────────────
    rows = []
    for name, _ in MODELS:
        s = summaries[name]
        if s is None:
            rows.append([name, "—", "—", "—"])
            continue
        d = _get(s, "ood_results", "domain_ood")
        if not isinstance(d, dict):
            rows.append([name, "—", "—", "—"])
            continue
        rows.append([name, _fmt(_get(d, "f1")), _fmt(_get(d, "fpr")), _fmt(_get(d, "fnr"))])
    _print_table(
        "TABLE 4 — Distribution shift: domain_ood  (2,000 rows, 4 new domains)",
        ["Model", "F1", "FPR", "FNR"],
        rows,
    )

    # ── Table 5: Structural OOD ───────────────────────────────────────────────
    rows = []
    for name, _ in MODELS:
        s = summaries[name]
        if s is None:
            rows.append([name, "—", "—", "—"])
            continue
        d = _get(s, "ood_results", "structural_ood")
        if not isinstance(d, dict):
            rows.append([name, "—", "—", "—"])
            continue
        rows.append([name, _fmt(_get(d, "f1")), _fmt(_get(d, "fpr")), _fmt(_get(d, "fnr"))])
    _print_table(
        "TABLE 5 — Distribution shift: structural_ood  (1,998 rows, 3 novel transforms)",
        ["Model", "F1", "FPR", "FNR"],
        rows,
    )

    # ── Table 6: Balanced subtype ─────────────────────────────────────────────
    rows = []
    for name, _ in MODELS:
        s = summaries[name]
        if s is None:
            rows.append([name, "—", "—", "—"])
            continue
        d = _get(s, "balanced_subtype")
        if not isinstance(d, dict):
            rows.append([name, "—", "—", "—"])
            continue
        rows.append([name, _fmt(_get(d, "f1")), _fmt(_get(d, "recall")), _fmt(_get(d, "fpr"))])
    _print_table(
        "TABLE 6 — Balanced subtype test  (2,297 rows, 6 attack subtypes)",
        ["Model", "F1", "Recall", "FPR"],
        rows,
    )

    print("\nNote: '—' means the baseline has not been run yet.")
    print("Run all baselines first, then re-run this script for the full table.\n")


if __name__ == "__main__":
    main()
