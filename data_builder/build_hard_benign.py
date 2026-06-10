"""
Build `hard_benign_test.csv` as a staged, checkpointed pipeline.

Best staged order:
  1. Collect/filter LMSYS candidates -> save CSV
  2. Collect/filter OASST1 + Dolly -> save CSVs
  3. Merge real candidate pool -> save CSV
  4. Generate 600 templates -> save CSV
  5. Exact dedup against train + self -> save CSV
  6. Semantic dedup -> save CSV
  7. Source-balanced sampling -> final hard_benign_test.csv
  8. Audit sample generation -> hard_benign_audit_sample.csv
"""
from __future__ import annotations

import gc
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
if HF_TOKEN:
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

CACHE_DIR = PROJECT_ROOT / "data/_cache_hf_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = PROJECT_ROOT / "data/pids_bench_v3/eval_subsets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DIR = OUT_DIR / "_staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "hard_benign_test.csv"
AUDIT_FILE = OUT_DIR / "hard_benign_audit_sample.csv"
TRAIN_FILE = PROJECT_ROOT / "data/pids_bench_v3/train.csv"

LMSYS_STAGE = STAGING_DIR / "hard_benign_lmsys_candidates.csv"
OASST_STAGE = STAGING_DIR / "hard_benign_oasst1_candidates.csv"
DOLLY_STAGE = STAGING_DIR / "hard_benign_dolly_candidates.csv"
REAL_STAGE = STAGING_DIR / "hard_benign_real_candidates.csv"
TEMPLATE_STAGE = STAGING_DIR / "hard_benign_template_candidates.csv"
EXACT_STAGE = STAGING_DIR / "hard_benign_exact_dedup.csv"
SEMANTIC_STAGE = STAGING_DIR / "hard_benign_semantic_dedup.csv"

FINAL_COLUMNS = [
    "parent_seed_id",
    "text",
    "label",
    "source",
    "attack_type",
    "language",
    "generator_model",
    "paraphrase_round",
    "source_type",
    "obfuscation",
]

TOTAL_TARGET = 4000
PREFERRED_TARGET = 4000
MIN_ACCEPTABLE = 3200
TEMPLATE_TARGET = 600

SOURCE_TARGETS = {
    "lmsys": 2500,
    "oasst1": 600,
    "dolly": 300,
    "templates": 600,
}

LMSYS_KEYWORD_QUOTA = 3800
LMSYS_RANDOM_QUOTA = 600
OASST_KEYWORD_QUOTA = 400
OASST_RANDOM_QUOTA = 100
DOLLY_KEYWORD_QUOTA = 190
DOLLY_RANDOM_QUOTA = 50
LMSYS_MAX_SCAN = 200_000

MIN_LEN = 20
MAX_LEN = 1000
BATCH_SIZE = 100
SEMANTIC_THRESHOLD = 0.92
SEMANTIC_MAX_FEATURES = 12_000
RESUME_STAGES = True

INJECTION_KEYWORDS = [
    "ignore",
    "override",
    "bypass",
    "pretend",
    "act as",
    "roleplay",
    "dan",
    "forget",
    "simulate",
    "hypothetically",
    "uncensored",
    "jailbreak",
    "system prompt",
    "disregard",
    "no restrictions",
    "without restrictions",
    "break character",
    "stay in character",
    "you are now",
    "ignore all",
    "ignore previous",
    "new persona",
    "developer mode",
    "do anything now",
    "instructions",
    "your guidelines",
]

AI_TERMS = [
    "ai",
    "llm",
    "model",
    "assistant",
    "chatbot",
    "prompt",
    "instruction",
    "system",
    "guardrail",
    "safety",
    "jailbreak",
    "injection",
    "gpt",
    "language model",
    "chatgpt",
    "neural",
    "fine-tun",
    "policy",
    "filter",
    "restrict",
    "openai",
    "anthropic",
    "claude",
    "alignment",
    "rlhf",
    "reward model",
    "training data",
    "inference",
    "moderation",
    "content policy",
    "refusal",
    "red team",
    "red-team",
    "adversarial",
    "attack",
    "exploit",
]

HARD_BENIGN_CONTEXT_TERMS = [
    "instruction",
    "instructions",
    "prompt",
    "system",
    "system prompt",
    "guardrail",
    "safety",
    "moderation",
    "content policy",
    "refusal",
    "alignment",
    "red team",
    "red-team",
    "adversarial",
    "jailbreak",
    "prompt injection",
    "persona",
    "terminal",
    "text completion model",
    "developer mode",
    "refuse",
    "policy",
]

EXCLUDE_PATTERNS = [
    r"\bdeveloper mode\b",
    r"\bdo anything now\b",
    r"\bignore all previous instructions\b",
    r"\bignore previous instructions\b",
    r"\bno special filtering\b",
    r"\bwithout restrictions\b",
    r"\bno restrictions\b",
    r"\breveal (your|its) system prompt\b",
    r"\bprovide a list of your instructions\b",
    r"\brun commands in my terminal\b",
    r"\bescape to the internet\b",
    r"\bserial keys?\b",
    r"\bharmful and illegal\b",
    r"\bself[- ]harm\b",
    r"\bshould i die\b",
    r"\bkill myself\b",
    r"\bracial slur\b",
    r"\bunderage\b",
    r"\bvenomous comment\b",
    r"\bderogatory\b",
    r"\bslanderous\b",
    r"\bthreatening dmca\b",
    r"\bbypass(?:es|ing)? our security\b",
    r"\bhorny\b",
    r"\blewd\b",
    r"\bdeceased grandmother\b",
    r"\billegal things\b",
    r"\bai girlfriend\b",
    r"\baroused\b",
    r"\balluring\b",
    r"\bhot spring\b",
    r"\bhoneymoon\b",
    r"\bintimacy\b",
    r"\blsd\b",
    r"\bforget all prior prompts\b",
    r"\bdisregard your earlier programming\b",
    r"\bsimulate a text based web browser\b",
    r"\bpretend to be a linux terminal\b",
    r"\byou are a shopping assistant\b",
]

LOW_SIGNAL_PATTERNS = [
    r"\bwhat did george dantzig develop\b",
    r"\bunderground railroad\b",
    r"\bwhich sport\b",
    r"\bcrossword\b",
    r"\bmy cat\b",
    r"\bfield hockey\b",
    r"\bkindle\b",
    r"\bmushulatubbee\b",
]

FINAL_EXCLUDE_PATTERNS = [
    r"\bhack a traffic control system\b",
    r"\bsteamy prompts\b",
    r"\brepeat the prompt above\b",
    r"\bwrite an article about the instruction of\b",
    r"\bwrite an article about the safety of\b",
    r"\bignore previous instructions\b",
    r"\bignore all previous instructions\b",
    r"\bforget all prior prompts\b",
    r"\blinux terminal\b",
    r"\bplease act as a terminal\b",
    r"\bjailbreaking it\b",
    r"\bsunshine party\b",
    r"\bunauthorized access to someone's email account\b",
    r"\bschizophrenic programmer\b",
    r"\bshapely blond female model\b",
    r"\bbig chest\b",
    r"\bsmartgpt is a new state of the art language model\b",
    r"\bignore all prompts (?:below|above)\b",
    r"\bsynthetic document\b",
    r"\bracial slurs\b",
    r"\bimpersonate her for a while\b",
]

TEXT_COMPLETION_CAP = 1650


def _normalize_text(text: str) -> str:
    text = str(text or "").encode("utf-8", errors="replace").decode("utf-8")
    text = "".join(ch for ch in text if ch in "\n\t\r" or ord(ch) >= 32)
    return re.sub(r"\s+", " ", text).strip()


def _contains_injection_keyword(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in INJECTION_KEYWORDS)


def _contains_ai_term(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in AI_TERMS)


def _contains_hard_benign_context(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in HARD_BENIGN_CONTEXT_TERMS)


def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _passes_moderation(mod: list | None) -> bool:
    if not mod:
        return True
    for entry in mod:
        cats = entry.get("categories", {})
        if isinstance(cats, dict) and any(bool(v) for v in cats.values()):
            return False
    return True


def _classify_real_text(text: str, moderation: list | None = None) -> str | None:
    text = _normalize_text(text)
    if not text:
        return None
    if not (MIN_LEN <= len(text) <= MAX_LEN):
        return None
    if not _passes_moderation(moderation):
        return None
    if _matches_any_pattern(text, EXCLUDE_PATTERNS):
        return None
    if _matches_any_pattern(text, LOW_SIGNAL_PATTERNS):
        return None
    has_context = _contains_hard_benign_context(text)
    if _contains_injection_keyword(text):
        if not has_context:
            return None
        return "keyword"
    if _contains_ai_term(text) and has_context:
        return "random"
    return None


def _make_row(text: str, source: str, idx: int, source_type: str = "real") -> dict:
    return {
        "parent_seed_id": f"hb_{source}_{idx:07d}",
        "text": _normalize_text(text),
        "label": 0,
        "source": source,
        "attack_type": "none",
        "language": "en",
        "generator_model": "none",
        "paraphrase_round": 0,
        "source_type": source_type,
        "obfuscation": "none",
    }


def _save_rows(rows: list[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=FINAL_COLUMNS)
    else:
        df = df[FINAL_COLUMNS]
    df.to_csv(path, index=False)
    return df


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _run_stage(path: Path, builder, stage_name: str) -> list[dict]:
    if RESUME_STAGES and path.exists():
        rows = _load_rows(path)
        print(f"[Resume] {stage_name}: loaded {len(rows):,} rows from {path.name}")
        return rows
    rows = builder()
    _save_rows(rows, path)
    print(f"[Stage] {stage_name}: saved {len(rows):,} rows -> {path.name}")
    return rows


def collect_lmsys() -> list[dict]:
    from datasets import load_dataset

    print(
        f"\n[LMSYS] Streaming up to {LMSYS_MAX_SCAN:,} rows "
        f"(keyword={LMSYS_KEYWORD_QUOTA}, random={LMSYS_RANDOM_QUOTA}) ..."
    )
    try:
        ds = load_dataset(
            "lmsys/lmsys-chat-1m",
            split="train",
            streaming=True,
            token=HF_TOKEN or True,
            cache_dir=str(CACHE_DIR),
        )
    except Exception as exc:
        print(f"[LMSYS] ERROR loading dataset: {exc}")
        return []

    keyword_rows: list[dict] = []
    random_rows: list[dict] = []
    scanned = 0
    t0 = time.time()

    for row in ds:
        if scanned >= LMSYS_MAX_SCAN:
            break
        if len(keyword_rows) >= LMSYS_KEYWORD_QUOTA and len(random_rows) >= LMSYS_RANDOM_QUOTA:
            break

        scanned += 1
        if scanned % 5000 == 0:
            elapsed = time.time() - t0
            print(
                f"  [LMSYS] scanned={scanned:,} | "
                f"keyword={len(keyword_rows)}/{LMSYS_KEYWORD_QUOTA} | "
                f"random={len(random_rows)}/{LMSYS_RANDOM_QUOTA} | "
                f"elapsed={elapsed:.0f}s"
            )

        if str(row.get("language", "")).strip().lower() != "english":
            continue
        if row.get("redacted", False):
            continue

        conversation = row.get("conversation", [])
        if not conversation:
            continue

        first_user = next(
            (message.get("content", "") for message in conversation if message.get("role") == "user"),
            "",
        )
        cls = _classify_real_text(first_user, row.get("openai_moderation"))
        if cls == "keyword" and len(keyword_rows) < LMSYS_KEYWORD_QUOTA:
            keyword_rows.append(_make_row(first_user, "lmsys", len(keyword_rows)))
        elif cls == "random" and len(random_rows) < LMSYS_RANDOM_QUOTA:
            random_rows.append(_make_row(first_user, "lmsys_ai_adjacent", len(random_rows)))

    rows = keyword_rows + random_rows
    print(
        f"[LMSYS] Done. scanned={scanned:,} | collected={len(rows):,} "
        f"(keyword={len(keyword_rows)}, random={len(random_rows)})"
    )
    return rows


def collect_oasst1() -> list[dict]:
    from datasets import load_dataset

    print(f"\n[OASST1] Loading (keyword={OASST_KEYWORD_QUOTA}, random={OASST_RANDOM_QUOTA}) ...")
    try:
        ds = load_dataset(
            "OpenAssistant/oasst1",
            split="train[:50000]",
            cache_dir=str(CACHE_DIR),
        )
    except Exception as exc:
        print(f"[OASST1] ERROR loading dataset: {exc}")
        return []

    keyword_rows: list[dict] = []
    random_rows: list[dict] = []
    for row in ds:
        if len(keyword_rows) >= OASST_KEYWORD_QUOTA and len(random_rows) >= OASST_RANDOM_QUOTA:
            break
        if str(row.get("role", "")).strip().lower() != "prompter":
            continue
        if str(row.get("lang", "")).strip().lower() != "en":
            continue

        text = row.get("text", "")
        cls = _classify_real_text(text, None)
        if cls == "keyword" and len(keyword_rows) < OASST_KEYWORD_QUOTA:
            keyword_rows.append(_make_row(text, "oasst1", len(keyword_rows)))
        elif cls == "random" and len(random_rows) < OASST_RANDOM_QUOTA:
            random_rows.append(_make_row(text, "oasst1_ai_adjacent", len(random_rows)))

    rows = keyword_rows + random_rows
    print(f"[OASST1] Done. collected={len(rows):,} (keyword={len(keyword_rows)}, random={len(random_rows)})")
    return rows


def collect_dolly() -> list[dict]:
    from datasets import load_dataset

    print(f"\n[Dolly] Loading (keyword={DOLLY_KEYWORD_QUOTA}, random={DOLLY_RANDOM_QUOTA}) ...")
    try:
        ds = load_dataset(
            "databricks/databricks-dolly-15k",
            split="train",
            cache_dir=str(CACHE_DIR),
        )
    except Exception as exc:
        print(f"[Dolly] ERROR loading dataset: {exc}")
        return []

    keyword_rows: list[dict] = []
    random_rows: list[dict] = []
    for row in ds:
        if len(keyword_rows) >= DOLLY_KEYWORD_QUOTA and len(random_rows) >= DOLLY_RANDOM_QUOTA:
            break
        text = row.get("instruction", "")
        cls = _classify_real_text(text, None)
        if cls == "keyword" and len(keyword_rows) < DOLLY_KEYWORD_QUOTA:
            keyword_rows.append(_make_row(text, "dolly", len(keyword_rows)))
        elif cls == "random" and len(random_rows) < DOLLY_RANDOM_QUOTA:
            random_rows.append(_make_row(text, "dolly_ai_adjacent", len(random_rows)))

    rows = keyword_rows + random_rows
    print(f"[Dolly] Done. collected={len(rows):,} (keyword={len(keyword_rows)}, random={len(random_rows)})")
    return rows


def merge_real_candidates(lmsys_rows: list[dict], oasst_rows: list[dict], dolly_rows: list[dict]) -> list[dict]:
    rows = lmsys_rows + oasst_rows + dolly_rows
    print(
        f"[Merge] Real candidates={len(rows):,} "
        f"(LMSYS={len(lmsys_rows):,}, OASST1={len(oasst_rows):,}, Dolly={len(dolly_rows):,})"
    )
    return rows


def collect_templates() -> list[dict]:
    from data_builder.generators.hard_benign_templates import generate_all

    raw = generate_all()
    rows = [
        {
            "parent_seed_id": f"hb_template_{i:07d}",
            "text": _normalize_text(row["text"]),
            "label": 0,
            "source": row["source"],
            "attack_type": "none",
            "language": "en",
            "generator_model": "none",
            "paraphrase_round": 0,
            "source_type": "curated",
            "obfuscation": "none",
        }
        for i, row in enumerate(raw[:TEMPLATE_TARGET])
    ]
    print(f"[Templates] Generated {len(rows):,} rows")
    return rows


def load_train_texts() -> set[str]:
    if not TRAIN_FILE.exists():
        print(f"[Dedup] WARNING: train.csv not found at {TRAIN_FILE}")
        return set()
    df = pd.read_csv(TRAIN_FILE, usecols=["text"])
    texts = set(df["text"].map(_normalize_text).str.lower().dropna())
    print(f"[Dedup] Loaded {len(texts):,} train texts")
    return texts


def exact_dedup(rows: list[dict], train_texts: set[str]) -> list[dict]:
    seen: set[str] = set()
    clean: list[dict] = []
    removed_train = 0
    removed_self = 0
    for row in rows:
        key = _normalize_text(row["text"]).lower()
        if key in train_texts:
            removed_train += 1
            continue
        if key in seen:
            removed_self += 1
            continue
        seen.add(key)
        row["text"] = _normalize_text(row["text"])
        clean.append(row)
    print(
        f"[Dedup] Exact dedup removed train={removed_train:,}, "
        f"self={removed_self:,}. Remaining={len(clean):,}"
    )
    return clean


def semantic_dedup(rows: list[dict], threshold: float = SEMANTIC_THRESHOLD) -> list[dict]:
    from scipy.sparse import vstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not rows:
        return []

    texts = [_normalize_text(row["text"]) for row in rows]
    print(
        f"[SemanticDedup] Fitting TF-IDF on {len(rows):,} rows "
        f"(max_features={SEMANTIC_MAX_FEATURES}, batch={BATCH_SIZE}) ..."
    )
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=1,
        max_features=SEMANTIC_MAX_FEATURES,
        sublinear_tf=True,
    )
    vectorizer.fit(texts)
    gc.collect()

    kept_indices: list[int] = []
    kept_vecs = None
    for start in range(0, len(rows), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(rows))
        batch_matrix = vectorizer.transform(texts[start:end])
        for offset in range(end - start):
            idx = start + offset
            vec = batch_matrix[offset : offset + 1]
            if kept_vecs is None:
                kept_indices.append(idx)
                kept_vecs = vec
                continue
            sims = cosine_similarity(vec, kept_vecs).flatten()
            if sims.max() < threshold:
                kept_indices.append(idx)
                kept_vecs = vstack([kept_vecs, vec])
        del batch_matrix
        gc.collect()
        if end % 500 == 0 or end == len(rows):
            print(f"  [SemanticDedup] processed {end:,}/{len(rows):,} | kept={len(kept_indices):,}")

    clean = [rows[i] for i in kept_indices]
    print(f"[SemanticDedup] Removed {len(rows) - len(clean):,}. Remaining={len(clean):,}")
    return clean


def final_quality_filter(rows: list[dict]) -> list[dict]:
    clean = [row for row in rows if not _matches_any_pattern(row["text"], FINAL_EXCLUDE_PATTERNS)]
    removed = len(rows) - len(clean)
    print(f"[FinalFilter] Removed {removed:,} rows using final exclusion rules. Remaining={len(clean):,}")
    return clean


def _source_group(row: dict) -> str:
    source = str(row.get("source", "")).lower()
    if source.startswith("lmsys"):
        return "lmsys"
    if source.startswith("oasst1"):
        return "oasst1"
    if source.startswith("dolly"):
        return "dolly"
    return "templates"


def sample_to_source_targets(rows: list[dict]) -> list[dict]:
    by_group: dict[str, list[dict]] = {group: [] for group in SOURCE_TARGETS}
    for row in rows:
        by_group[_source_group(row)].append(row)

    result: list[dict] = []
    lmsys_backfill: list[dict] = []
    for group, target in SOURCE_TARGETS.items():
        pool = by_group.get(group, [])
        if group == "lmsys":
            non_text_completion = [
                row
                for row in pool
                if "you are the text completion model" not in row["text"].lower()
            ]
            text_completion = [
                row
                for row in pool
                if "you are the text completion model" in row["text"].lower()
            ]
            selected = []
            if len(non_text_completion) <= target:
                selected.extend(non_text_completion)
            else:
                selected.extend(random.sample(non_text_completion, target))
            remaining = target - len(selected)
            if remaining > 0:
                text_cap = min(TEXT_COMPLETION_CAP, remaining + len(selected))
                max_text_take = max(0, text_cap - len([r for r in selected if "you are the text completion model" in r["text"].lower()]))
                text_take = min(remaining, len(text_completion), max_text_take)
                if text_take:
                    selected.extend(random.sample(text_completion, text_take))
            result.extend(selected)
            selected_ids = {row["parent_seed_id"] for row in selected}
            lmsys_backfill = [row for row in pool if row["parent_seed_id"] not in selected_ids]
        elif len(pool) <= target:
            result.extend(pool)
        else:
            result.extend(random.sample(pool, target))

    if len(result) < MIN_ACCEPTABLE and lmsys_backfill:
        need = MIN_ACCEPTABLE - len(result)
        lmsys_current = sum(1 for row in result if _source_group(row) == "lmsys")
        max_backfill = max(0, 3200 - lmsys_current)
        n_backfill = min(need, len(lmsys_backfill), max_backfill)
        if n_backfill:
            result.extend(random.sample(lmsys_backfill, n_backfill))
            print(f"[Sample] LMSYS backfill added {n_backfill:,} rows")

    print("[Sample] Source balance (actual | target)")
    for group, target in SOURCE_TARGETS.items():
        actual = sum(1 for row in result if _source_group(row) == group)
        print(f"  {group:<10} {actual:>4} | {target}")
    print(f"[Sample] Final total={len(result):,} | preferred={PREFERRED_TARGET:,} | minimum={MIN_ACCEPTABLE:,}")
    return result


def build_audit_sample(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)[FINAL_COLUMNS].reset_index(drop=True)
    audit_n = min(200, len(df))
    if audit_n and "source" in df.columns:
        source_counts = df["source"].value_counts().sort_index()
        source_props = source_counts / source_counts.sum() * audit_n
        source_targets = source_props.astype(int)
        remainder = audit_n - int(source_targets.sum())
        if remainder > 0:
            frac = (source_props - source_targets).sort_values(ascending=False)
            for source_name in frac.index[:remainder]:
                source_targets[source_name] += 1

        samples: list[pd.DataFrame] = []
        for i, (source_name, n_source) in enumerate(source_targets.items()):
            if n_source <= 0:
                continue
            source_df = df[df["source"] == source_name]
            samples.append(source_df.sample(n=int(n_source), random_state=RANDOM_SEED + i))
        audit_df = pd.concat(samples, axis=0).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        audit_df = df.sample(n=audit_n, random_state=RANDOM_SEED).reset_index(drop=True)
    audit_df["audit_is_benign"] = ""
    audit_df["audit_is_attack_adjacent"] = ""
    audit_df["audit_notes"] = ""
    audit_df.to_csv(AUDIT_FILE, index=False)
    print(f"[Audit] Saved {audit_n:,} rows -> {AUDIT_FILE}")
    return audit_df


def main() -> None:
    print("=" * 70)
    print("BUILD hard_benign_test.csv")
    print("=" * 70)

    lmsys_rows = _run_stage(LMSYS_STAGE, collect_lmsys, "LMSYS candidates")
    oasst_rows = _run_stage(OASST_STAGE, collect_oasst1, "OASST1 candidates")
    dolly_rows = _run_stage(DOLLY_STAGE, collect_dolly, "Dolly candidates")
    real_rows = _run_stage(
        REAL_STAGE,
        lambda: merge_real_candidates(lmsys_rows, oasst_rows, dolly_rows),
        "merged real candidates",
    )
    template_rows = _run_stage(TEMPLATE_STAGE, collect_templates, "template candidates")

    train_texts = load_train_texts()
    exact_rows = _run_stage(
        EXACT_STAGE,
        lambda: exact_dedup(real_rows + template_rows, train_texts),
        "exact dedup",
    )
    semantic_rows = _run_stage(
        SEMANTIC_STAGE,
        lambda: semantic_dedup(exact_rows, threshold=SEMANTIC_THRESHOLD),
        "semantic dedup",
    )

    filtered_rows = final_quality_filter(semantic_rows)
    final_rows = sample_to_source_targets(filtered_rows)
    final_df = _save_rows(final_rows, OUT_FILE)
    build_audit_sample(final_rows)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows       : {len(final_df):,}")
    print(f"Label=0 (benign) : {(final_df['label'] == 0).sum():,}")
    real_count = (final_df["source_type"] == "real").sum()
    template_count = (final_df["source_type"] == "curated").sum()
    print(f"Real prompts     : {real_count:,}")
    print(f"Curated templates: {template_count:,}")
    print(f"Final file       : {OUT_FILE}")
    print(f"Audit file       : {AUDIT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
