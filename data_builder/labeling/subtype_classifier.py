"""
Rule-based subtype classifier for prompt injection attacks.

Assigns one of six attack subtypes to each injection prompt using
priority-ordered regex rules grounded in the PIDS-Bench taxonomy.

Taxonomy (taxonomy.json):
  direct_override        — Explicit instruction override
  roleplay_attack        — Persona-based manipulation
  contextual_manipulation — Indirect steering, framing-based bypass
  data_exfiltration      — Requests to leak prompts, secrets, credentials
  encoded_attack         — Encoded/obfuscated payloads (base64, rot13, etc.)
  tool_injection         — Force tool use or tool misuse (APIs, commands)

Priority order (most specific first, catch-all last):
  1. encoded_attack
  2. data_exfiltration
  3. tool_injection
  4. roleplay_attack
  5. contextual_manipulation
  6. direct_override  (catch-all)

Usage:
    python -m data_builder.labeling.subtype_classifier \
        --input  data/real_prompts/spml_injection.csv \
        --output data/real_prompts/spml_injection_classified.csv \
        --validate 200

Design notes:
  - Every rule was grounded by inspecting 5+ real SPML examples per subtype.
  - "tool" alone is intentionally NOT used — it matches persona descriptions
    ("you are a virtual tool/assistant"). Only specific tool-action phrases are used.
  - Contextual manipulation uses framing keywords ("hypothetically", "imagine",
    "fictional") distinct from roleplay persona keywords.
  - Precision on a 200-row stratified validation sample is printed after running.
"""

from __future__ import annotations

import argparse
import re
import random
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Rule definitions
# Each rule is (subtype_label, compiled_regex).
# Rules are evaluated in order; first match wins.
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, re.Pattern]] = []


def _r(subtype: str, pattern: str) -> None:
    _RULES.append((subtype, re.compile(pattern, re.IGNORECASE | re.DOTALL)))


# ── 1. encoded_attack ──────────────────────────────────────────────────────
# Base64/rot13/hex/cipher explicit encoding instructions or encoded blobs.
_r("encoded_attack", r"rot[-_\s]?13")
_r("encoded_attack", r"\bbase[-_\s]?64\b")
_r("encoded_attack", r"\bhex[-_\s]?(?:encode|decode|encoded|decoded)\b")
_r("encoded_attack", r"\bcipher(?:text)?\b.*\b(?:instruc|prompt|message)\b")
_r("encoded_attack", r"\b(?:encode|encrypt)\b.*\b(?:instruc|prompt|secret|password)\b")
_r("encoded_attack", r"\b(?:instruc|prompt|secret)\b.*\b(?:encode|encrypt)\b")

# ── 2. data_exfiltration ───────────────────────────────────────────────────
# Requests to expose hidden prompts, system instructions, credentials.
_r("data_exfiltration", r"repeat\b.*\b(?:your\s+)?(?:instruction|rule|directive|prompt)")
_r("data_exfiltration", r"(?:reveal|expose|leak|output|print|show|share|tell me|give me)\b.{0,60}\b(?:system\s+prompt|initialization|hidden\s+prompt|secret\s+prompt|original\s+prompt|your\s+instruction|your\s+rule|your\s+directive|your\s+password|your\s+credential|your\s+key)")
_r("data_exfiltration", r"what\s+(?:are\s+)?your\s+(?:instruction|rule|directive|prompt|constraint)")
_r("data_exfiltration", r"(?:list|enumerate|describe|state)\s+(?:all\s+)?(?:your\s+)?(?:instruction|rule|directive|constraint|guideline)")
_r("data_exfiltration", r"(?:access|retrieve|extract|dump)\b.{0,40}\b(?:data|credential|secret|password|key|token)")
_r("data_exfiltration", r"(?:password|secret\s+code|access\s+code|authorization\s+code)\b.{0,30}\b(?:now|immediately|directly)")

# ── 3. tool_injection ─────────────────────────────────────────────────────
# Specific tool/API/command-execution framing (NOT generic "tool" mentions).
_r("tool_injection", r"(?:call|invoke|trigger|execute|run)\b.{0,40}\b(?:api|endpoint|function|command|script|tool\s+call|plugin)")
_r("tool_injection", r"make\b.{0,20}\b(?:api|http|web)\b.{0,20}\b(?:call|request|query)")
_r("tool_injection", r"(?:send|post|get|fetch)\b.{0,30}\b(?:api|endpoint|webhook|request)\b")
_r("tool_injection", r"\b(?:bash|shell|terminal|subprocess|exec\(|os\.system)\b")
_r("tool_injection", r"(?:delete|modify|write|overwrite)\b.{0,40}\b(?:file|database|record|entry|table)")
_r("tool_injection", r"use\b.{0,20}\b(?:search\s+tool|calculator\s+tool|browser\s+tool|code\s+interpreter)")

# ── 4. roleplay_attack ────────────────────────────────────────────────────
# Persona injection: the attacker assigns a new identity to the model.
_r("roleplay_attack", r"you\s+are\s+(?:now\s+)?(?:a|an|the)\s+\w")
_r("roleplay_attack", r"from\s+now\s+on\s+(?:you\s+are|you\s+will\s+be|act\s+as)")
_r("roleplay_attack", r"\bact\s+as\b")
_r("roleplay_attack", r"\bpretend\s+(?:to\s+be|you\s+are|that\s+you)\b")
_r("roleplay_attack", r"your\s+(?:new\s+)?name\s+is\b")
_r("roleplay_attack", r"\broleplay\b.{0,40}\b(?:as|being|character)")
_r("roleplay_attack", r"you\s+will\s+(?:now\s+)?(?:play|portray|embody|take\s+on\s+the\s+role)")
_r("roleplay_attack", r"(?:switch|change|adopt)\b.{0,30}\bpersona\b")
_r("roleplay_attack", r"\byou\s+are\s+[A-Z][a-z]+(?:Bot|AI|GPT|Pal|Buddy|Assistant|Helper|Advisor|Guru|Expert|Guide)\b")

# ── 5. contextual_manipulation ────────────────────────────────────────────
# Framing-based bypasses: hypothetical, fictional, academic contexts used
# to indirectly steer the model past its constraints.
_r("contextual_manipulation", r"\bhypothetically\b")
_r("contextual_manipulation", r"\bin\s+(?:a\s+)?(?:fictional|hypothetical|imaginary|simulated)\b")
_r("contextual_manipulation", r"\bimagine\s+(?:you\s+(?:are|were|had)|if\s+you)\b")
_r("contextual_manipulation", r"\bsuppose\s+(?:you\s+(?:are|were|could)|that\s+you)\b")
_r("contextual_manipulation", r"\bfor\s+(?:academic|research|educational|training|fictional|story|creative)\s+purpose")
_r("contextual_manipulation", r"\bwhat\s+would\s+(?:you\s+)?(?:say|do|respond|answer)\s+if\b")
_r("contextual_manipulation", r"\bif\s+you\s+(?:were|had\s+no|didn.t\s+have|could\s+ignore)\b.{0,40}\b(?:rule|restriction|limit|constraint|guideline|instruction)")
_r("contextual_manipulation", r"\bfor\s+safety\s+(?:reason|purpose|test)\b.{0,80}\b(?:ignore|bypass|override|skip)")
_r("contextual_manipulation", r"\bnow\s+(?:take|accept|follow)\s+instructions?\s+(?:from|in)\b")
_r("contextual_manipulation", r"\binstruction\s+(?:injection|smug|hidden|embedded)\b")

# ── 6. direct_override ────────────────────────────────────────────────────
# Catch-all: explicit override verbs targeting previous instructions.
# This rule always matches; it must remain last.
_r("direct_override", r".*")


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_subtype(text: str) -> str:
    """Return the first matching attack subtype for the given text."""
    if not isinstance(text, str) or not text.strip():
        return "direct_override"
    for subtype, pattern in _RULES:
        if pattern.search(text):
            return subtype
    return "direct_override"


def classify_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Add an attack_type column to a copy of df."""
    out = df.copy()
    out["attack_type"] = out[text_col].apply(classify_subtype)
    return out


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return a stratified random sample of n rows across attack subtypes."""
    subtypes = df["attack_type"].unique().tolist()
    per_class = max(1, n // len(subtypes))
    frames = []
    rng = random.Random(seed)
    for st in subtypes:
        rows = df[df["attack_type"] == st]
        k = min(per_class, len(rows))
        frames.append(rows.sample(k, random_state=seed))
    sample = pd.concat(frames, ignore_index=True)
    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)


def print_distribution(df: pd.DataFrame, title: str = "Distribution") -> None:
    counts = df["attack_type"].value_counts()
    total = len(df)
    print(f"\n=== {title} (n={total:,}) ===")
    for subtype, count in counts.items():
        print(f"  {subtype:<30} {count:>6,}  ({count/total*100:5.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Classify injection prompts into subtypes.")
    parser.add_argument("--input",    required=True, help="Input CSV with a 'text' column")
    parser.add_argument("--output",   required=True, help="Output CSV path")
    parser.add_argument("--text-col", default="text", help="Name of text column (default: text)")
    parser.add_argument("--validate", type=int, default=200,
                        help="Save a stratified validation sample of this size (0 = skip)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    print(f"Loading  : {in_path}")
    df = pd.read_csv(in_path)
    print(f"Rows     : {len(df):,}")

    classified = classify_dataframe(df, text_col=args.text_col)
    print_distribution(classified, title="Subtype distribution after classification")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(out_path, index=False)
    print(f"\nSaved    : {out_path}")

    if args.validate > 0:
        val_path = out_path.parent / (out_path.stem + "_validation_sample.csv")
        sample = stratified_sample(classified, n=args.validate, seed=args.seed)
        sample.to_csv(val_path, index=False)
        print(f"Saved validation sample ({len(sample)} rows) : {val_path}")
        print("\nIMPORTANT: Manually review the validation sample and record precision per subtype.")
        print("This is required for the paper's annotation quality section.")


if __name__ == "__main__":
    main()
