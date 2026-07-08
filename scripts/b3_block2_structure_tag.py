#!/usr/bin/env python3
"""
B3 / Block 2 -- Instruction-structure tagging of the externally-sourced
hard-benign set, and FPR measurement tagged-vs-untagged.

PURPOSE
-------
Block 1 showed over-defense is present across three independent corpora
(lmsys / oasst1 / dolly) and is HIGHER on the "plain" conversational slices
than on the topic-filtered "ai_adjacent" slices. That points away from
"security topic" as the trigger and toward instruction-like SURFACE STRUCTURE.

This script tests that directly and quantitatively: it tags each externally-
sourced benign row for the presence of instruction-like structure, then
measures FPR on tagged vs untagged rows -- holding corpus fixed and holding
topic-slice fixed -- for both fine-tuned detectors plus the rule baseline.

ANTI-CIRCULARITY (this is the methodological core)
--------------------------------------------------
The structure tag MUST be independent of every evaluated detector, or the
result is a tautology. Two independence guarantees:

  1. It is NOT derived from src/baselines/rule_based.py (that keyword list is
     itself an evaluated detector). A different, structure-oriented pattern set
     is used here.
  2. It reuses ONLY the specific, structurally-meaningful patterns from the
     PIDS-Bench subtype taxonomy (roleplay / contextual / exfiltration /
     tool / encoded). It deliberately EXCLUDES the direct_override catch-all
     (r".*"), which matches every string and would tag all rows.

The tag is therefore: "does this benign text exhibit one of the taxonomy's
explicit instruction-structured surface forms?" -- a linguistic property a
person can verify by eye, defined without reference to any model's scores.

The tag patterns are frozen below and printed at runtime for auditability.

OUTPUT
------
  outputs/resub_analysis/b3_block2_structure_fpr.csv      (the measurement)
  outputs/resub_analysis/b3_block2_tagged_rows.csv        (audit: which rows tagged)
  prints tables to stdout.

USAGE
  python scripts/b3_block2_structure_tag.py
"""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/M2026206/khalid/Prompt-Injection-Detector-System")
SEEDS = [13, 42, 123, 2024, 7777]
N_BOOT = 5000
CI = 95
RNG_SEED = 12345
OUT_DIR = REPO / "outputs" / "resub_analysis"
HB_FILE = REPO / "data" / "pids_bench_v3" / "eval_subsets" / "hard_benign_test.csv"

BASELINE_DIRS = {
    "DeBERTa-v3-FT": "outputs/multi_seed_runs/deberta_baseline_clean",
    "DistilBERT":    "outputs/multi_seed_runs/distilbert",
}

# ---------------------------------------------------------------------------
# INSTRUCTION-STRUCTURE TAG
# Drawn ONLY from the specific (non-catch-all) taxonomy patterns. Each entry is
# (taxonomy_origin, regex). No pattern here is taken from rule_based.py, and the
# direct_override ".*" catch-all is intentionally absent.
# ---------------------------------------------------------------------------
_TAG_PATTERNS: list[tuple[str, re.Pattern]] = []

def _p(origin: str, pattern: str) -> None:
    _TAG_PATTERNS.append((origin, re.compile(pattern, re.IGNORECASE | re.DOTALL)))

# roleplay / persona-assignment structure
_p("roleplay",  r"you\s+are\s+(?:now\s+)?(?:a|an|the)\s+\w")
_p("roleplay",  r"from\s+now\s+on\s+(?:you\s+are|you\s+will\s+be|act\s+as)")
_p("roleplay",  r"\bact\s+as\b")
_p("roleplay",  r"\bpretend\s+(?:to\s+be|you\s+are|that\s+you)\b")
_p("roleplay",  r"you\s+will\s+(?:now\s+)?(?:play|portray|embody|take\s+on\s+the\s+role)")
_p("roleplay",  r"(?:switch|change|adopt)\b.{0,30}\bpersona\b")

# contextual framing structure
_p("contextual", r"\bhypothetically\b")
_p("contextual", r"\bin\s+(?:a\s+)?(?:fictional|hypothetical|imaginary|simulated)\b")
_p("contextual", r"\bimagine\s+(?:you\s+(?:are|were|had)|if\s+you)\b")
_p("contextual", r"\bsuppose\s+(?:you\s+(?:are|were|could)|that\s+you)\b")
_p("contextual", r"\bwhat\s+would\s+(?:you\s+)?(?:say|do|respond|answer)\s+if\b")
_p("contextual", r"\bif\s+you\s+(?:were|had\s+no|could\s+ignore)\b.{0,40}\b(?:rule|restriction|limit|constraint|guideline|instruction)")

# exfiltration-style imperative directed at the system/instructions
_p("exfiltration", r"repeat\b.*\b(?:your\s+)?(?:instruction|rule|directive|prompt)")
_p("exfiltration", r"(?:reveal|expose|leak|output|print|show|share|tell me|give me)\b.{0,60}\b(?:system\s+prompt|hidden\s+prompt|original\s+prompt|your\s+instruction|your\s+rule|your\s+directive)")
_p("exfiltration", r"what\s+(?:are\s+)?your\s+(?:instruction|rule|directive|prompt|constraint)")
_p("exfiltration", r"(?:list|enumerate|describe|state)\s+(?:all\s+)?(?:your\s+)?(?:instruction|rule|directive|constraint|guideline)")

# tool/command-execution imperative structure
_p("tool", r"(?:call|invoke|trigger|execute|run)\b.{0,40}\b(?:api|endpoint|function|command|script|tool\s+call|plugin)")
_p("tool", r"\b(?:bash|shell|terminal|subprocess|exec\(|os\.system)\b")
_p("tool", r"(?:delete|modify|write|overwrite)\b.{0,40}\b(?:file|database|record|entry|table)")

# encoded-payload structure
_p("encoded", r"rot[-_\s]?13")
_p("encoded", r"\bbase[-_\s]?64\b")
_p("encoded", r"\b(?:encode|encrypt)\b.{0,40}\b(?:instruc|prompt|secret|password)\b")

# a separate, explicit imperative-override family (linguistic, not from rule_based.py)
_p("override", r"\b(?:ignore|disregard|forget)\b.{0,30}\b(?:previous|prior|above|earlier|the\s+following|all)\b")
_p("override", r"\bnew\s+instructions?\b\s*[:\-]")


def has_structure(text: str) -> tuple[int, list[str]]:
    """Return (1/0, list of matched taxonomy origins). Model-blind, rule-blind."""
    if not isinstance(text, str) or not text.strip():
        return 0, []
    hits = sorted({origin for origin, pat in _TAG_PATTERNS if pat.search(text)})
    return (1 if hits else 0), hits


def corpus_of(s): return s.replace("_ai_adjacent", "")
def slice_of(s):  return "ai_adjacent" if s.endswith("_ai_adjacent") else "plain"


def load_seed(pred_path: Path, hb: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(pred_path)
    pred = pred.drop(columns=[c for c in ("source", "source_type") if c in pred.columns])
    m = pred.merge(hb[["text", "source", "source_type"]], on="text",
                   how="left", validate="one_to_one")
    if m["source"].isna().any():
        raise SystemExit(f"JOIN FAILED on {pred_path}")
    m["corpus"] = m["source"].map(corpus_of)
    m["slice"]  = m["source"].map(slice_of)
    return m


def boot_ci(flags, rng):
    if len(flags) == 0:
        return (float("nan"), float("nan"))
    n = len(flags)
    b = np.array([flags[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(np.percentile(b, (100-CI)/2)), float(np.percentile(b, 100-(100-CI)/2))


def rate(per_seed_flags, rng):
    seed_rates = [f.mean() if len(f) else float("nan") for f in per_seed_flags]
    pooled = np.concatenate(per_seed_flags) if per_seed_flags else np.array([])
    lo, hi = boot_ci(pooled, rng)
    return {
        "fpr_mean": float(np.nanmean(seed_rates)),
        "fpr_std": float(np.nanstd(seed_rates, ddof=1)) if len(seed_rates) > 1 else 0.0,
        "ci_lo": lo, "ci_hi": hi,
        "n": int(len(per_seed_flags[0])) if per_seed_flags else 0,
    }


def main():
    hb = pd.read_csv(HB_FILE)
    ext = hb[hb["source_type"] == "real"].copy()      # the 872
    ext["corpus"] = ext["source"].map(corpus_of)
    ext["slice"]  = ext["source"].map(slice_of)
    ext["struct"], ext["struct_origins"] = zip(*ext["text"].map(has_structure))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    # ---- audit: how many rows tagged, by corpus/slice ----
    print(f"Structure-tag patterns in use: {len(_TAG_PATTERNS)} "
          f"(taxonomy-derived, rule_based.py-independent, no catch-all)")
    print(f"\nTagged externally-sourced rows: {int(ext['struct'].sum())} / {len(ext)} "
          f"({ext['struct'].mean()*100:.1f}%)")
    print("\nTag rate by corpus x slice:")
    print(pd.crosstab([ext["corpus"], ext["slice"]], ext["struct"],
                      margins=True))

    # save the tagged rows for human / second-annotator audit
    ext_audit = ext[["text", "source", "corpus", "slice", "struct",
                     "struct_origins"]].copy()
    ext_audit["struct_origins"] = ext_audit["struct_origins"].map(lambda x: "|".join(x))
    ext_audit.to_csv(OUT_DIR / "b3_block2_tagged_rows.csv", index=False)

    # ---- measurement: FPR tagged vs untagged, per model ----
    rows = []
    tag_lookup = ext.set_index("text")["struct"].to_dict()

    for model, rel in BASELINE_DIRS.items():
        bdir = REPO / rel
        frames = []
        for s in SEEDS:
            p = bdir / f"seed_{s}" / "hard_benign_predictions.csv"
            if p.exists():
                df = load_seed(p, hb)
                df = df[df["source_type"] == "real"].copy()
                df["struct"] = df["text"].map(tag_lookup)
                frames.append(df)
        if not frames:
            print(f"[skip] {model}: no baseline predictions")
            continue

        print(f"\n{'='*72}\n{model}\n{'='*72}")
        # overall tagged vs untagged
        for tag in (1, 0):
            flags = [d.loc[d["struct"] == tag, "flagged_default"].to_numpy() for d in frames]
            r = rate(flags, rng)
            label = "structured" if tag else "unstructured"
            print(f"  ALL-ext {label:<13} n={r['n']:>4}  FPR={r['fpr_mean']:.4f} "
                  f"±{r['fpr_std']:.4f}  [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")
            rows.append({"model": model, "corpus": "ALL", "slice": "ALL",
                         "struct": tag, **r})

        # within corpus x slice (the controlled contrast)
        print(f"\n  within corpus x slice (structured vs unstructured):")
        for corpus in ("lmsys", "oasst1", "dolly"):
            for sl in ("plain", "ai_adjacent"):
                for tag in (1, 0):
                    flags = [d.loc[(d["corpus"] == corpus) & (d["slice"] == sl)
                                   & (d["struct"] == tag),
                                   "flagged_default"].to_numpy() for d in frames]
                    if all(len(f) == 0 for f in flags):
                        continue
                    r = rate(flags, rng)
                    label = "struct" if tag else "unstruct"
                    print(f"    {corpus:<7} {sl:<11} {label:<9} "
                          f"n={r['n']:>3}  FPR={r['fpr_mean']:.4f} "
                          f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")
                    rows.append({"model": model, "corpus": corpus, "slice": sl,
                                 "struct": tag, **r})

    pd.DataFrame(rows).to_csv(OUT_DIR / "b3_block2_structure_fpr.csv", index=False)
    print(f"\nWrote:\n  {OUT_DIR/'b3_block2_structure_fpr.csv'}"
          f"\n  {OUT_DIR/'b3_block2_tagged_rows.csv'}  (audit these by eye)")


if __name__ == "__main__":
    main()
