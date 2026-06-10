"""Two-axis seed balancing: attack subtype distribution + global real/template mix."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)

INJECTION_SUBTYPES = (
    "direct_override",
    "roleplay_attack",
    "contextual_manipulation",
    "data_exfiltration",
    "encoded_attack",
    "tool_injection",
)


def apply_two_axis_balance(
    candidate_pool: pd.DataFrame,
    config: dict,
    report_path: Path | None = None,
) -> pd.DataFrame:
    """
    Two-axis seed balancing: subtype distribution (primary) + global real/template (secondary).

    - Primary: attack_type distribution (30/15/20/12/10/13)
    - Secondary: global source_type target (~60% real, ~40% template)
    - Per subtype: prefer real first, fill with template
    - If global real > target: swap real→template within same subtype

    Candidate pool must be deduplicated by text before calling.
    Uses a set for selected-text tracking (candidate pool is deduplicated).
    Switch to a counter if balancing ever runs on non-deduplicated rows.
    """
    if candidate_pool.empty:
        return candidate_pool

    seed = config.get("seed", 42)
    targets = _compute_targets(candidate_pool, config)
    selected, unselected = _select_per_subtype(candidate_pool, targets, seed)

    # Assemble pool and compute achieved global real ratio
    rows = []
    for st in INJECTION_SUBTYPES:
        rows.extend(selected["injection"][st]["real"])
        rows.extend(selected["injection"][st]["template"])
    rows.extend(selected["benign"]["real"])
    rows.extend(selected["benign"]["template"])

    total = len(rows)
    total_real = sum(
        len(selected["injection"][st]["real"]) for st in INJECTION_SUBTYPES
    ) + len(selected["benign"]["real"])
    total_template = total - total_real
    achieved_real_ratio = total_real / total if total > 0 else 0.0
    target_real_ratio = targets["source_type"]["real"]

    # If real > target, perform real→template swaps
    if achieved_real_ratio > target_real_ratio and total > 0:
        target_real_count = int(round(total * target_real_ratio))
        excess_real = total_real - target_real_count
        if excess_real > 0:
            _swap_real_to_template(
                selected, unselected, excess_real, seed
            )
            # Reassemble after swaps
            rows = []
            for st in INJECTION_SUBTYPES:
                rows.extend(selected["injection"][st]["real"])
                rows.extend(selected["injection"][st]["template"])
            rows.extend(selected["benign"]["real"])
            rows.extend(selected["benign"]["template"])

    result = pd.DataFrame(rows).reset_index(drop=True)
    result = result.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    _write_balance_report(selected, targets, report_path)

    return result


def _compute_targets(candidate_pool: pd.DataFrame, config: dict) -> dict:
    """Compute subtype targets and source_type target from config."""
    benign = candidate_pool[candidate_pool["label"] == 0]
    injection = candidate_pool[candidate_pool["label"] == 1]

    subtype_dist = config.get("subtype_distribution", {})
    source_dist = config.get("source_type_distribution", {"real": 0.60, "template": 0.40})

    # Total pool size: use seed_budget or full candidate size
    budget = config.get("seed_budget", len(candidate_pool))
    total_target = min(budget, len(candidate_pool))

    # Injection ratio: use explicit label_balance config when set (e.g. 0.50 for 50/50).
    # Falls back to inferring from candidate pool if not set.
    label_balance = config.get("label_balance")
    if label_balance is not None:
        inj_ratio = float(label_balance)
    elif len(candidate_pool) > 0:
        inj_ratio = len(injection) / len(candidate_pool)
    else:
        inj_ratio = 0.5
    total_inj_target = int(round(total_target * inj_ratio))
    total_benign_target = total_target - total_inj_target

    # Enforce true label balance: neither side can exceed what is available.
    # When label_balance is set explicitly (e.g. 0.50), both targets must be equal
    # and achievable — cap both to min(inj_available, benign_available).
    # Without this, if one class has fewer candidates than its target, the other
    # class fills its full target and the final label ratio is skewed.
    if label_balance is not None:
        inj_available = len(injection)
        ben_available = len(benign)
        capped_inj = min(total_inj_target, inj_available)
        capped_ben = min(total_benign_target, ben_available)
        balanced = min(capped_inj, capped_ben)
        total_inj_target = balanced
        total_benign_target = balanced
        LOG.info(
            "label_balance=%.2f: capped targets to %d injection / %d benign "
            "(pool: %d inj / %d ben)",
            label_balance, total_inj_target, total_benign_target,
            inj_available, ben_available,
        )

    targets = {}
    targets["subtype"] = {}
    for st in INJECTION_SUBTYPES:
        ratio = float(subtype_dist.get(st, 0))
        targets["subtype"][st] = max(0, int(round(total_inj_target * ratio)))
    delta = total_inj_target - sum(targets["subtype"].values())
    if delta != 0 and "direct_override" in targets["subtype"]:
        targets["subtype"]["direct_override"] += delta

    targets["benign"] = total_benign_target
    targets["source_type"] = {
        "real": float(source_dist.get("real", 0.60)),
        "template": float(source_dist.get("template", 0.40)),
    }
    targets["total"] = total_target
    return targets


def _select_per_subtype(
    candidate_pool: pd.DataFrame,
    targets: dict,
    seed: int,
) -> tuple[dict, dict]:
    """
    Per-subtype selection: prefer real, fill with template.
    Returns (selected, unselected) structures.
    """
    import random

    rng = random.Random(seed)
    benign = candidate_pool[candidate_pool["label"] == 0]
    injection = candidate_pool[candidate_pool["label"] == 1]

    selected = {
        "injection": {
            st: {"real": [], "template": []} for st in INJECTION_SUBTYPES
        },
        "benign": {"real": [], "template": []},
    }
    unselected = {
        "injection": {
            st: {"real": [], "template": []} for st in INJECTION_SUBTYPES
        },
        "benign": {"real": [], "template": []},
    }

    def _sample_rows(df: pd.DataFrame, n: int) -> tuple[list, list]:
        records = df.to_dict("records") if len(df) > 0 else []
        if n <= 0:
            return [], records
        if len(records) <= n:
            return records, []
        idx = rng.sample(range(len(records)), n)
        taken = [records[i] for i in idx]
        rest = [r for i, r in enumerate(records) if i not in set(idx)]
        return taken, rest

    # Injection: per subtype
    for st in INJECTION_SUBTYPES:
        sub = injection[injection["attack_type"] == st]
        real_df = sub[sub["source_type"] == "real"]
        template_df = sub[sub["source_type"] == "template"]
        target = targets["subtype"].get(st, 0)

        take_real = min(len(real_df), target)
        take_template = min(len(template_df), target - take_real)

        real_taken, real_rest = _sample_rows(real_df, take_real)
        template_taken, template_rest = _sample_rows(template_df, take_template)

        selected["injection"][st]["real"] = real_taken
        selected["injection"][st]["template"] = template_taken
        unselected["injection"][st]["real"] = real_rest
        unselected["injection"][st]["template"] = template_rest

    # Benign
    real_benign = benign[benign["source_type"] == "real"]
    template_benign = benign[benign["source_type"] == "template"]
    target_b = targets["benign"]

    take_real_b = min(len(real_benign), target_b)
    take_template_b = min(len(template_benign), target_b - take_real_b)

    real_taken_b, real_rest_b = _sample_rows(real_benign, take_real_b)
    template_taken_b, template_rest_b = _sample_rows(template_benign, take_template_b)

    selected["benign"]["real"] = real_taken_b
    selected["benign"]["template"] = template_taken_b
    unselected["benign"]["real"] = real_rest_b
    unselected["benign"]["template"] = template_rest_b

    return selected, unselected


def _swap_real_to_template(
    selected: dict,
    unselected: dict,
    excess_real: int,
    seed: int,
) -> None:
    """
    Swap real→template within same subtype.
    - Same attack_type only
    - Use only unselected template rows
    - Avoid duplicate text (selected_texts set; pool is deduplicated)
    """
    import random

    rng = random.Random(seed)

    # Candidate pool is deduplicated before balancing, so each text appears at most once.
    # Use a set for selected-text tracking. Switch to a counter if balancing ever runs
    # on non-deduplicated candidate rows.
    selected_texts = set()
    for st in INJECTION_SUBTYPES:
        for row in selected["injection"][st]["real"] + selected["injection"][st]["template"]:
            t = row.get("text")
            if t is not None and (isinstance(t, str) or hasattr(t, "__str__")):
                selected_texts.add(str(t).strip())
    for row in selected["benign"]["real"] + selected["benign"]["template"]:
        t = row.get("text")
        if t is not None and (isinstance(t, str) or hasattr(t, "__str__")):
            selected_texts.add(str(t).strip())

    swapped = 0
    for st in INJECTION_SUBTYPES:
        if swapped >= excess_real:
            break
        real_list = selected["injection"][st]["real"]
        unsel_template = unselected["injection"][st]["template"]
        if not real_list or not unsel_template:
            continue

        rng.shuffle(unsel_template)
        for _ in range(min(excess_real - swapped, len(real_list))):
            if not real_list or not unsel_template:
                break
            removed = real_list.pop()
            removed_text = str(removed.get("text", "")).strip()
            selected_texts.discard(removed_text)

            candidate = None
            for row in unsel_template:
                t = str(row.get("text", "")).strip()
                if t not in selected_texts:
                    candidate = row
                    break
            if candidate is None:
                selected_texts.add(removed_text)
                real_list.append(removed)
                break

            selected["injection"][st]["template"].append(candidate)
            unsel_template.remove(candidate)
            selected_texts.add(str(candidate.get("text", "")).strip())
            swapped += 1


def _write_balance_report(
    selected: dict,
    targets: dict,
    report_path: Path | None,
) -> None:
    """Write balance report: attack_type, source_type, attack_type×source_type, shortfalls."""
    # Achieved counts
    at_counts = {}
    for st in INJECTION_SUBTYPES:
        r = len(selected["injection"][st]["real"])
        t = len(selected["injection"][st]["template"])
        at_counts[st] = {"count": r + t, "real": r, "template": t}
    at_counts["none"] = {
        "count": len(selected["benign"]["real"]) + len(selected["benign"]["template"]),
        "real": len(selected["benign"]["real"]),
        "template": len(selected["benign"]["template"]),
    }

    total = sum(at_counts[st]["count"] for st in INJECTION_SUBTYPES) + at_counts["none"]["count"]
    total_real = sum(len(selected["injection"][st]["real"]) for st in INJECTION_SUBTYPES) + len(
        selected["benign"]["real"]
    )
    total_template = total - total_real
    achieved_real_ratio = total_real / total if total > 0 else 0
    target_real_ratio = targets["source_type"]["real"]

    report = {
        "targets": {
            "attack_type": targets["subtype"],
            "source_type": targets["source_type"],
            "benign": targets["benign"],
        },
        "achieved": {
            "attack_type": {
                st: {
                    "count": at_counts[st]["count"],
                    "ratio": round(at_counts[st]["count"] / total, 4) if total > 0 else 0,
                }
                for st in INJECTION_SUBTYPES
            },
            "attack_type_none": {
                "count": at_counts["none"]["count"],
                "ratio": round(at_counts["none"]["count"] / total, 4) if total > 0 else 0,
            },
            "source_type": {
                "real": {"count": total_real, "ratio": round(achieved_real_ratio, 4)},
                "template": {"count": total_template, "ratio": round(1 - achieved_real_ratio, 4)},
            },
            "attack_type_x_source_type": {
                st: {"real": at_counts[st]["real"], "template": at_counts[st]["template"]}
                for st in INJECTION_SUBTYPES
            },
            "attack_type_none_x_source_type": {
                "real": at_counts["none"]["real"],
                "template": at_counts["none"]["template"],
            },
        },
        "shortfalls": {},
        "total_seeds": total,
    }

    if achieved_real_ratio < target_real_ratio:
        report["shortfalls"]["source_type"] = {
            "real": {
                "target_ratio": target_real_ratio,
                "achieved_ratio": achieved_real_ratio,
                "message": "Insufficient real seeds to meet target",
            }
        }

    for st in INJECTION_SUBTYPES:
        target = targets["subtype"].get(st, 0)
        achieved = at_counts[st]["count"]
        if achieved < target:
            report["shortfalls"].setdefault("attack_type", {})[st] = {
                "target": target,
                "achieved": achieved,
                "shortfall": target - achieved,
            }

    LOG.info(
        "Balance report: attack_type=%s, source_type real=%d(%.2f) template=%d(%.2f)",
        {st: at_counts[st]["count"] for st in INJECTION_SUBTYPES},
        total_real,
        achieved_real_ratio,
        total_template,
        1 - achieved_real_ratio,
    )

    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        LOG.info("Wrote balance report to %s", report_path)
