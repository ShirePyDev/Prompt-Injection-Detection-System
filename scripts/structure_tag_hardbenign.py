#!/usr/bin/env python3
"""
Instruction-structure tagging of the externally-sourced hard-benign set.

Tests whether benign rows get over-flagged because they carry instruction-like
surface STRUCTURE -- not because of which corpus they came from, and not because
they discuss a security TOPIC. The contrast is measured inside each corpus and
inside each topic slice, so corpus and topic are both held fixed.

WHY THE TAG IS DEFENSIBLE
-------------------------
The tag is independent of every evaluated detector, so the measurement is not
circular:
  * It is NOT drawn from src/baselines/rule_based.py (that keyword list is an
    evaluated detector in its own right).
  * It is NOT the full subtype classifier either: that classifier ends in a
    direct_override catch-all (r".*") that matches every string. Running it on
    benign text would tag everything.
  * Instead it uses only the structurally specific, manually-verified patterns
    for persona assignment (roleplay), framing (contextual), and genuine
    command-execution phrasing (tool) -- each requiring an explicit imperative
    construction, not a bare keyword.

An earlier, looser version of this tag was audited by eye and three origins
(encoded, exfiltration, override) were found to misfire on benign text -- e.g.
a "convert this to nodejs" code request matched an encode pattern, and a benign
"judge whether this is prompt injection" task matched an override pattern. Those
origins are dropped here. The tool patterns are tightened to require a command
verb together with an execution object. The result is re-audited automatically
at the end of this run; only tag a claim if that audit is clean.

OUTPUT
  outputs/resub_analysis/structure_fpr_by_cell.csv     -- the FPR measurement
  outputs/resub_analysis/structure_tagged_rows.csv     -- every tagged row (audit)
  prints the tag rate, the FPR tables, and a fresh random audit sample.

USAGE
  python scripts/structure_tag_hardbenign.py
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
# INSTRUCTION-STRUCTURE TAG  (roleplay + contextual + tightened tool only)
# Each pattern requires an explicit instruction construction. No bare keywords,
# no catch-all, nothing copied from rule_based.py.
# ---------------------------------------------------------------------------
_TAG: list[tuple[str, re.Pattern]] = []

def _p(origin: str, pattern: str) -> None:
    _TAG.append((origin, re.compile(pattern, re.IGNORECASE | re.DOTALL)))

# --- persona assignment (roleplay): the prompt assigns the model an identity ---
_p("roleplay", r"\bact\s+as\s+(?:a|an|the)\b")
_p("roleplay", r"\bi\s+want\s+you\s+to\s+act\s+as\b")
_p("roleplay", r"\byou\s+are\s+(?:now\s+)?(?:a|an|the)\s+\w")
_p("roleplay", r"\bfrom\s+now\s+on\s+(?:you\s+are|you\s+will\s+be|act\s+as)\b")
_p("roleplay", r"\bpretend\s+(?:to\s+be|you\s+are|that\s+you)\b")
_p("roleplay", r"\byou\s+will\s+(?:now\s+)?(?:play|portray|embody|take\s+on\s+the\s+role)\b")
_p("roleplay", r"\b(?:switch|change|adopt)\b.{0,30}\bpersona\b")
_p("roleplay", r"\byour\s+(?:new\s+)?name\s+is\b")

# --- framing (contextual): hypothetical / fictional framing used as a directive ---
_p("contextual", r"\bhypothetically\b")
_p("contextual", r"\bin\s+(?:a\s+)?(?:fictional|hypothetical|imaginary|simulated)\s+(?:scenario|setting|world|context)\b")
_p("contextual", r"\bimagine\s+(?:you\s+(?:are|were|had)|that\s+you\s+are)\b")
_p("contextual", r"\bsuppose\s+(?:you\s+(?:are|were|could)|that\s+you\s+are)\b")
_p("contextual", r"\bif\s+you\s+(?:were|had\s+no|could\s+ignore)\b.{0,40}\b(?:rule|restriction|limit|constraint|guideline|instruction)")

# --- command execution (tool, tightened): imperative verb + execution object ---
_p("tool", r"\b(?:call|invoke|execute|run)\s+(?:the\s+|a\s+|this\s+)?(?:api|endpoint|function|command|script|tool|plugin)\b")
_p("tool", r"\b(?:execute|run)\s+(?:the\s+following\s+)?(?:bash|shell|terminal|code|command)\b")
_p("tool", r"\bos\.system\b|\bsubprocess\b|\bexec\(")

# patterns that previously misfired on benign text are intentionally NOT here:
#   - encoded   (matched "convert this to nodejs" code requests)
#   - exfiltration (overlapped benign "what are your ..." questions)
#   - override  (matched benign "the following" / "judge whether this is injection")


def has_structure(text: str) -> tuple[int, list[str]]:
    if not isinstance(text, str) or not text.strip():
        return 0, []
    hits = sorted({o for o, p in _TAG if p.search(text)})
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
    seed_rates = [f.mean() if len(f) else np.nan for f in per_seed_flags]
    pooled = np.concatenate(per_seed_flags) if per_seed_flags else np.array([])
    lo, hi = boot_ci(pooled, rng)
    return {"fpr_mean": float(np.nanmean(seed_rates)),
            "fpr_std": float(np.nanstd(seed_rates, ddof=1)) if len(seed_rates) > 1 else 0.0,
            "ci_lo": lo, "ci_hi": hi,
            "n": int(len(per_seed_flags[0])) if per_seed_flags else 0}


def main():
    hb = pd.read_csv(HB_FILE)
    ext = hb[hb["source_type"] == "real"].copy()       # the 872
    ext["corpus"] = ext["source"].map(corpus_of)
    ext["slice"]  = ext["source"].map(slice_of)
    ext["struct"], ext["origins"] = zip(*ext["text"].map(has_structure))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print(f"Tag patterns: {len(_TAG)}  (roleplay + contextual + tightened tool; "
          f"rule_based.py-independent; no catch-all)")
    print(f"Tagged: {int(ext['struct'].sum())} / {len(ext)} "
          f"({ext['struct'].mean()*100:.1f}%)")
    print("\nOrigins firing:")
    origins = ext.loc[ext['struct'] == 1, 'origins'].map(lambda x: "|".join(x))
    print(origins.value_counts().to_string())
    print("\nTag rate by corpus x slice:")
    print(pd.crosstab([ext["corpus"], ext["slice"]], ext["struct"], margins=True))

    audit = ext[["text", "source", "corpus", "slice", "struct", "origins"]].copy()
    audit["origins"] = audit["origins"].map(lambda x: "|".join(x))
    audit.to_csv(OUT_DIR / "structure_tagged_rows.csv", index=False)

    tag_lookup = ext.set_index("text")["struct"].to_dict()
    rows = []
    for model, rel in BASELINE_DIRS.items():
        bdir = REPO / rel
        frames = []
        for s in SEEDS:
            p = bdir / f"seed_{s}" / "hard_benign_predictions.csv"
            if p.exists():
                d = load_seed(p, hb)
                d = d[d["source_type"] == "real"].copy()
                d["struct"] = d["text"].map(tag_lookup)
                frames.append(d)
        if not frames:
            print(f"[skip] {model}")
            continue

        print(f"\n{'='*72}\n{model}\n{'='*72}")
        for tag in (1, 0):
            flags = [d.loc[d["struct"] == tag, "flagged_default"].to_numpy() for d in frames]
            r = rate(flags, rng)
            lbl = "structured" if tag else "unstructured"
            print(f"  ALL-ext {lbl:<13} n={r['n']:>4}  FPR={r['fpr_mean']:.4f} "
                  f"+/-{r['fpr_std']:.4f}  [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")
            rows.append({"model": model, "corpus": "ALL", "slice": "ALL", "struct": tag, **r})

        print("\n  within corpus x slice (structured vs unstructured):")
        for corpus in ("lmsys", "oasst1", "dolly"):
            for sl in ("plain", "ai_adjacent"):
                for tag in (1, 0):
                    flags = [d.loc[(d["corpus"] == corpus) & (d["slice"] == sl)
                                   & (d["struct"] == tag), "flagged_default"].to_numpy()
                             for d in frames]
                    if all(len(f) == 0 for f in flags):
                        continue
                    r = rate(flags, rng)
                    lbl = "struct" if tag else "unstruct"
                    print(f"    {corpus:<7} {sl:<11} {lbl:<9} n={r['n']:>3}  "
                          f"FPR={r['fpr_mean']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")
                    rows.append({"model": model, "corpus": corpus, "slice": sl,
                                 "struct": tag, **r})

    pd.DataFrame(rows).to_csv(OUT_DIR / "structure_fpr_by_cell.csv", index=False)

    # ---- automatic re-audit: a fresh random sample of tagged rows ----
    print(f"\n{'='*72}\nAUDIT: 15 random tagged rows -- check every one is "
          f"genuinely instruction-structured\n{'='*72}")
    tg = audit[audit["struct"] == 1]
    for _, r in tg.sample(min(15, len(tg)), random_state=7).iterrows():
        print(f"\n[{r['corpus']}/{r['slice']}] origins={r['origins']}")
        print("  ", str(r["text"])[:180].replace("\n", " "))

    print(f"\nWrote:\n  {OUT_DIR/'structure_fpr_by_cell.csv'}"
          f"\n  {OUT_DIR/'structure_tagged_rows.csv'}")


if __name__ == "__main__":
    main()
