"""Prompt ingestion helpers for the PIDS-Bench data builder."""
from __future__ import annotations

import base64
import logging
from pathlib import Path
import pandas as pd
import yaml

from data_builder.generators.templates_benign import BENIGN_TEMPLATES, TOPICS
from data_builder.generators.templates_injection import (
    ATTACK_TEMPLATES,
    CONTEXTS,
    ENCODING_TYPES,
    EXFIL_ACTIONS,
    GOALS,
    PERSONAS,
    TOOL_GOALS,
    TOOL_GOALS_ALL,
)
from data_builder.sources.real_loader import load_real_prompts as _load_real_source_rows
from data_builder.utils import dedup
from data_builder.utils.balance import apply_two_axis_balance

LOG = logging.getLogger(__name__)
CONFIG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CONFIG_DIR / "config.yaml"
SEED_COLUMNS = ["seed_id", "text", "label", "source", "attack_type", "language"]


def load_config(config_path: str | Path | None = None) -> dict:
    """Load config. Uses PIDS_BENCH_CONFIG env, else config_path, else config.yaml."""
    import os
    if config_path is not None:
        path = Path(config_path)
    elif os.environ.get("PIDS_BENCH_CONFIG"):
        path = Path(os.environ["PIDS_BENCH_CONFIG"])
    else:
        path = DEFAULT_CONFIG
    if not path.is_absolute():
        path = CONFIG_DIR / path.name
    return yaml.safe_load(path.read_text())


def _build_template_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for template in BENIGN_TEMPLATES:
        for topic in TOPICS:
            rows.append(
                {
                    "text": template.format(topic=topic),
                    "label": 0,
                    "source": "template",
                    "attack_type": "none",
                    "language": "en",
                }
            )

    for attack_type, templates in ATTACK_TEMPLATES.items():
        for tpl in templates:
            if "{persona}" in tpl:
                for persona in PERSONAS:
                    for goal in GOALS:
                        rows.append(
                            {
                                "text": tpl.format(persona=persona, goal=goal),
                                "label": 1,
                                "source": "template",
                                "attack_type": attack_type,
                                "language": "en",
                            }
                        )
            elif "{context}" in tpl:
                for context in CONTEXTS:
                    for goal in GOALS:
                        rows.append(
                            {
                                "text": tpl.format(context=context, goal=goal),
                                "label": 1,
                                "source": "template",
                                "attack_type": attack_type,
                                "language": "en",
                            }
                        )
            elif "{action}" in tpl:
                for action in EXFIL_ACTIONS:
                    for goal in GOALS:
                        rows.append(
                            {
                                "text": tpl.format(action=action, goal=goal),
                                "label": 1,
                                "source": "template",
                                "attack_type": attack_type,
                                "language": "en",
                            }
                        )
            elif "{encoding_type}" in tpl:
                for encoding_type in ENCODING_TYPES:
                    for goal in GOALS:
                        payload = base64.b64encode(goal.encode()).decode()
                        rows.append(
                            {
                                "text": tpl.format(encoding_type=encoding_type, payload=payload),
                                "label": 1,
                                "source": "template",
                                "attack_type": attack_type,
                                "language": "en",
                            }
                        )
            elif "{tool_goal}" in tpl:
                tool_goals = TOOL_GOALS_ALL if attack_type == "tool_injection" else TOOL_GOALS
                for goal in tool_goals:
                    rows.append(
                        {
                            "text": tpl.format(tool_goal=goal),
                            "label": 1,
                            "source": "template",
                            "attack_type": attack_type,
                            "language": "en",
                        }
                    )
            elif "{payload}" in tpl:
                for goal in GOALS:
                    payload = base64.b64encode(goal.encode()).decode()
                    rows.append(
                        {
                            "text": tpl.format(payload=payload),
                            "label": 1,
                            "source": "template",
                            "attack_type": attack_type,
                            "language": "en",
                        }
                    )
            else:
                for goal in GOALS:
                    rows.append(
                        {
                            "text": tpl.format(goal=goal),
                            "label": 1,
                            "source": "template",
                            "attack_type": attack_type,
                            "language": "en",
                        }
                    )
    return rows


def load_template_seeds(config_path: str | Path | None = None) -> pd.DataFrame:
    """Load template seeds. Only genuinely unique textual seeds; no duplicate text with different seed_id."""
    config = load_config(config_path)
    budget = config.get("seed_budget", 20000)
    base_rows = _build_template_rows()
    if not base_rows:
        return pd.DataFrame(columns=SEED_COLUMNS)

    # Deduplicate by text: keep first occurrence (no identical text with different seed_id)
    seen_texts: set[str] = set()
    unique_rows: list[dict[str, str | int]] = []
    for row in base_rows:
        text = row["text"]
        if text in seen_texts:
            continue
        seen_texts.add(text)
        unique_rows.append(row)

    # Shuffle unique rows so all attack subtypes are evenly represented in any prefix slice.
    # Without shuffling, template.head(N) would cut off subtypes that appear late in
    # generation order (encoded_attack, tool_injection) when N < total templates.
    rng = __import__("random").Random(config.get("seed", 42))
    rng.shuffle(unique_rows)

    if len(unique_rows) > budget:
        unique_rows = rng.sample(unique_rows, budget)
    elif len(unique_rows) < budget:
        LOG.info(
            "Template seeds: %d unique (budget %d); using all unique, no cycling",
            len(unique_rows),
            budget,
        )

    expanded: list[dict[str, str | int]] = []
    for idx, row in enumerate(unique_rows):
        entry = row.copy()
        seed_id = f"template_seed_{idx:07d}"
        entry.update(
            {
                "seed_id": seed_id,
                "id": f"template_row_{idx:07d}",
                "parent_seed_id": seed_id,
                "source_type": "template",
            }
        )
        expanded.append(entry)
    return pd.DataFrame(expanded)


def load_real_prompts(config_path: str | Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    rows = _load_real_source_rows(config)
    if not rows:
        return pd.DataFrame(columns=SEED_COLUMNS)
    return pd.DataFrame(rows)


def enforce_quota(df: pd.DataFrame, total: int, ratio: float) -> pd.DataFrame:
    target = int(round(total * ratio))
    if len(df) > target:
        return df.sample(target, random_state=42).reset_index(drop=True)
    return df.copy()


def _balance_injection_subtypes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Balance injection subtypes to match target distribution.
    - If config has subtype_distribution (v3): resample to hit target ratios.
    - Else if subtype_balance.enabled: cap direct_override at max_ratio * min(other).
    """
    inj = df[df["label"] == 1]
    benign = df[df["label"] == 0]
    if inj.empty:
        return df

    seed = config.get("seed", 42)
    rng = __import__("random").Random(seed)

    # v3: target distribution (30/15/20/12/10/13)
    dist = config.get("subtype_distribution", {})
    if dist:
        total_inj = len(inj)
        subtypes = [
            "direct_override",
            "roleplay_attack",
            "contextual_manipulation",
            "data_exfiltration",
            "encoded_attack",
            "tool_injection",
        ]
        target = {at: int(total_inj * float(dist.get(at, 0))) for at in subtypes}
        # Ensure sum equals total_inj (rounding)
        delta = total_inj - sum(target.values())
        if delta != 0 and "direct_override" in target:
            target["direct_override"] = target["direct_override"] + delta

        sampled = []
        for at in subtypes:
            sub = inj[inj["attack_type"] == at]
            n = target.get(at, 0)
            if n <= 0:
                continue
            if len(sub) >= n:
                sampled.append(sub.sample(n=n, random_state=seed))
            else:
                sampled.append(sub.sample(n=n, replace=True, random_state=seed))
        inj_balanced = pd.concat(sampled, ignore_index=True)
        result = pd.concat([benign, inj_balanced], ignore_index=True)
        result = result.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        LOG.info(
            "Subtype distribution applied: %s",
            inj_balanced["attack_type"].value_counts().to_dict(),
        )
        return result

    # Legacy: cap direct_override
    balance = config.get("subtype_balance", {})
    if not balance.get("enabled", False):
        return df
    max_ratio = float(balance.get("direct_override_max_ratio", 2.0))
    counts = inj["attack_type"].value_counts()
    others = [c for at, c in counts.items() if at != "direct_override"]
    if not others:
        return df
    min_other = min(others)
    cap = int(max_ratio * min_other)
    do_count = counts.get("direct_override", 0)
    if do_count <= cap:
        return df
    do_rows = inj[inj["attack_type"] == "direct_override"]
    keep = do_rows.sample(n=cap, random_state=seed)
    inj_rest = inj[inj["attack_type"] != "direct_override"]
    inj_balanced = pd.concat([inj_rest, keep], ignore_index=True)
    result = pd.concat([benign, inj_balanced], ignore_index=True)
    result = result.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    LOG.info("Subtype balance: capped direct_override %d -> %d", do_count, cap)
    return result


def collect_seeds(config_path: str | Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    budget = config.get("seed_budget", 20000)
    real_ratio = config.get("real_prompt_ratio", 0.15)

    real = load_real_prompts(config_path)
    # Cross-dataset dedup: remove exact duplicates across real sources.
    # Near-duplicate dedup on real seeds is opt-in via cross_dataset_near_dedup (default: false)
    # because it is O(n²) and sources are already from distinct non-overlapping domains.
    if not real.empty and config.get("cross_dataset_dedup", True):
        before = len(real)
        real = dedup.exact_dedup(real)
        if config.get("cross_dataset_near_dedup", False):
            real = dedup.near_duplicate_dedup(real)
        LOG.info("Cross-dataset dedup (real sources): %d -> %d rows", before, len(real))
    real = enforce_quota(real, budget, real_ratio)

    template = load_template_seeds(config_path)
    needed = budget - len(real)
    if needed > len(template):
        LOG.warning("Not enough template seeds (%d) to reach budget %d", len(template), budget)
        needed = len(template)
    template_subset = template.head(needed).copy()
    template_subset["source_type"] = "template"

    combined = pd.concat([real.assign(source_type="real"), template_subset], ignore_index=True)
    combined["seed_id"] = combined["seed_id"].fillna(
        pd.Series(combined.index.astype(int), index=combined.index)
    )
    combined = combined.reset_index(drop=True)

    # Deduplicate candidate pool before balancing (required for selected_texts set).
    # Near-duplicate dedup is O(n²) and skipped when cross_dataset_near_dedup=false
    # (the default for v3 with clean, non-overlapping sources).
    combined = dedup.exact_dedup(combined)
    if config.get("cross_dataset_near_dedup", False):
        combined = dedup.near_duplicate_dedup(combined)
    LOG.info("Candidate pool after dedup: %d rows", len(combined))

    # Two-axis balancing when subtype_distribution is present (v3)
    if config.get("subtype_distribution"):
        report_path = config.get("balance_report_path")
        if report_path:
            _root = Path(__file__).resolve().parents[2]
            report_path = _root / report_path if not Path(report_path).is_absolute() else Path(report_path)
        combined = apply_two_axis_balance(combined, config, report_path=report_path)
    else:
        combined = _balance_injection_subtypes(combined, config)
    return combined


def audit_seed_sample(df: pd.DataFrame, path: Path, sample_size: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        LOG.warning("No seeds to sample for audit")
        return
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    sample.to_csv(path, index=False)
    LOG.info("Wrote %d seed audit rows to %s", len(sample), path)
