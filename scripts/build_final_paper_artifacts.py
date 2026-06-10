#!/usr/bin/env python3
"""Build final paper tables and tradeoff figures from frozen result artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "paper_final_assets"
REPORT_PATH = ROOT / "reports" / "FINAL_RESULTS_TABLES.md"
CORE_CI_REPORT_PATH = ROOT / "reports" / "FAMILY_CI_CORE_RESULTS.md"
OOD_CI_REPORT_PATH = ROOT / "reports" / "FAMILY_CI_OOD_RESULTS.md"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    group: str
    kind: str
    summary_path: Path


MODEL_SPECS = [
    ModelSpec(
        model_id="rule_based",
        display_name="Rule-based",
        group="internal",
        kind="heuristic floor",
        summary_path=ROOT / "outputs" / "rule_baseline" / "summary.json",
    ),
    ModelSpec(
        model_id="tfidf_logreg",
        display_name="TF-IDF+LR",
        group="internal",
        kind="classical baseline",
        summary_path=ROOT / "outputs" / "tfidf_baseline" / "summary.json",
    ),
    ModelSpec(
        model_id="distilbert",
        display_name="DistilBERT",
        group="internal",
        kind="efficient neural baseline",
        summary_path=ROOT / "outputs" / "distilbert_baseline" / "summary.json",
    ),
    ModelSpec(
        model_id="deberta_v3",
        display_name="DeBERTa-v3",
        group="internal",
        kind="strong in-repo reference",
        summary_path=ROOT / "outputs" / "deberta_v3" / "summary.json",
    ),
    ModelSpec(
        model_id="protectai_small",
        display_name="ProtectAI small",
        group="external_pi",
        kind="external PI detector",
        summary_path=ROOT / "outputs" / "external_eval" / "multiaxis" / "protectai_small" / "summary.json",
    ),
    ModelSpec(
        model_id="deberta_pi",
        display_name="DeBERTa-PI",
        group="external_pi",
        kind="external PI detector",
        summary_path=ROOT / "outputs" / "external_eval" / "multiaxis" / "deberta_pi" / "summary.json",
    ),
    ModelSpec(
        model_id="promptguard2_86m",
        display_name="PromptGuard 2 (86M)",
        group="external_broad",
        kind="broad malicious-prompt comparator",
        summary_path=ROOT / "outputs" / "external_eval" / "multiaxis" / "promptguard2_86m" / "summary.json",
    ),
    ModelSpec(
        model_id="llamaguard",
        display_name="Llama Guard 3-1B",
        group="external_broad",
        kind="broad safety-policy comparator",
        summary_path=ROOT / "outputs" / "external_eval" / "multiaxis" / "llamaguard" / "summary.json",
    ),
    ModelSpec(
        model_id="gpt4o_mini",
        display_name="GPT-4o-mini (0-shot)",
        group="llm_zeroshot",
        kind="zero-shot LLM baseline",
        summary_path=ROOT / "outputs" / "external_eval" / "multiaxis" / "gpt4o_mini" / "summary.json",
    ),
]

DISPLAY_TO_ID = {spec.display_name: spec.model_id for spec in MODEL_SPECS}
DISPLAY_TO_ID.update(
    {
        "Rule-based": "rule_based",
        "TF-IDF+LR": "tfidf_logreg",
        "TF-IDF + LR": "tfidf_logreg",
        "DistilBERT": "distilbert",
        "DeBERTa-v3": "deberta_v3",
        "Llama Guard 3-1B": "llamaguard",
        "GPT-4o-mini (0-shot)": "gpt4o_mini",
    }
)

GROUP_COLORS = {
    "internal": "#1f77b4",
    "external_pi": "#d62728",
    "external_broad": "#ff7f0e",
    "llm_zeroshot": "#9467bd",
}

HB_LABEL_OFFSETS = {
    "rule_based": (8, -10),
    "tfidf_logreg": (-76, 18),
    "distilbert": (10, -6),
    "deberta_v3": (10, 16),
    "protectai_small": (-118, -4),
    "deberta_pi": (-66, 18),
    "promptguard2_86m": (10, -10),
    "llamaguard": (10, -10),
    "gpt4o_mini": (10, -10),
}

STRUCT_LABEL_OFFSETS = {
    "rule_based": (8, -10),
    "tfidf_logreg": (-74, -10),
    "distilbert": (-66, 18),
    "deberta_v3": (10, 16),
    "protectai_small": (-118, -8),
    "deberta_pi": (-66, 18),
    "promptguard2_86m": (10, -10),
    "llamaguard": (10, -10),
    "gpt4o_mini": (10, -10),
}


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def format_num(value: float) -> str:
    return f"{value:.4f}"


def _load_roc_pr_for_model(model_id: str) -> dict:
    """
    Attempt to load ROC-AUC and PR-AP from:
      1. outputs/<model_dir>/threshold_analysis/validation_tuned_test_metrics.csv
      2. outputs/roc_pr_curves/roc_pr_summary.json  (multi-model summary)
    Returns dict with keys roc_auc_test, pr_ap_test (or None if unavailable).
    """
    model_dir_map = {
        "rule_based":      None,  # rule-based has no probability scores
        "tfidf_logreg":    ROOT / "outputs" / "tfidf_baseline",
        "distilbert":      ROOT / "outputs" / "distilbert_baseline",
        "deberta_v3":      ROOT / "outputs" / "deberta_v3",
    }
    result = {"roc_auc_test": None, "pr_ap_test": None}

    # Path 1: per-model threshold_analysis output
    if model_id in model_dir_map and model_dir_map[model_id] is not None:
        ta_csv = model_dir_map[model_id] / "threshold_analysis" / "validation_tuned_test_metrics.csv"
        if ta_csv.exists():
            import pandas as _pd
            row = _pd.read_csv(ta_csv).iloc[0]
            result["roc_auc_test"] = float(row.get("roc_auc_test", float("nan"))) or None
            result["pr_ap_test"]   = float(row.get("pr_ap_test",   float("nan"))) or None
            if result["roc_auc_test"] is not None:
                return result

    # Path 2: multi-model ROC/PR summary
    roc_json = ROOT / "outputs" / "roc_pr_curves" / "roc_pr_summary.json"
    if roc_json.exists():
        import json as _json
        rows = _json.loads(roc_json.read_text())
        for r in rows:
            if r.get("model") == model_id and r.get("split") == "test":
                result["roc_auc_test"] = r.get("roc_auc")
                result["pr_ap_test"]   = r.get("pr_ap")
                return result

    return result


def _load_source_type_f1(model_id: str) -> dict:
    """
    Load real vs template F1 from outputs/paper_final_assets/table_source_type_breakdown.csv.
    Returns dict with keys real_f1, template_f1, gap (or None if unavailable).
    """
    st_csv = ROOT / "outputs" / "paper_final_assets" / "table_source_type_breakdown.csv"
    if not st_csv.exists():
        return {"real_f1": None, "template_f1": None, "gap_tmpl_minus_real": None}
    import pandas as _pd
    df = _pd.read_csv(st_csv)
    # Match on model_id — the CSV has display_name, not model_id, so we derive it
    from_id_to_display = {
        "rule_based": "Rule-based",
        "tfidf_logreg": "TF-IDF + LR",
        "distilbert": "DistilBERT",
        "deberta_v3": "DeBERTa-v3",
    }
    display = from_id_to_display.get(model_id)
    if display is None:
        return {"real_f1": None, "template_f1": None, "gap_tmpl_minus_real": None}
    row = df[df["display_name"] == display]
    if row.empty:
        return {"real_f1": None, "template_f1": None, "gap_tmpl_minus_real": None}
    r = row.iloc[0]
    return {
        "real_f1":             float(r["real_f1"]),
        "template_f1":         float(r["template_f1"]),
        "gap_tmpl_minus_real": float(r["gap_tmpl_minus_real"]),
    }


def build_model_record(spec: ModelSpec) -> dict:
    summary = read_json(spec.summary_path)

    if spec.model_id == "rule_based":
        test = summary["test_results"]
        hard_benign_fpr = summary["hard_benign_test_fpr"]["fpr"]
        obfuscated_recall = summary["obfuscated_attacks"]["overall_recall"]
        balanced_subtype_f1 = summary["balanced_subtype"]["f1"]
        domain = summary["ood_results"]["domain_ood"]
        structural = summary["ood_results"]["structural_ood"]
        note = "Binary rule set; no threshold tuning."
    elif spec.group == "internal":
        test = summary["test_results"]["val_tuned_threshold"]
        hard_benign_fpr = summary["hard_benign_test_fpr"]["fpr_tuned"]
        obfuscated_recall = summary["obfuscated_attacks"]["overall_recall"]
        balanced_subtype_f1 = summary["balanced_subtype"]["f1"]
        domain = summary["ood_results"]["domain_ood"]
        structural = summary["ood_results"]["structural_ood"]
        note = summary["threshold_note"]
    elif spec.model_id == "gpt4o_mini":
        splits = summary["splits"]
        test = splits["test"]
        hard_benign_fpr = splits["hard_benign"]["fpr"]
        obfuscated_recall = splits["obfuscated_attacks"]["recall"]
        balanced_subtype_f1 = splits["balanced_subtype"]["f1"]
        domain = splits["domain_ood"]
        structural = splits["structural_ood"]
        note = summary.get("framing_note", "Zero-shot GPT-4o-mini; threshold fixed at 0.5.")
    else:
        splits = summary["splits"]
        test = splits["test"]
        hard_benign_fpr = splits["hard_benign"]["fpr"]
        obfuscated_recall = splits["obfuscated_attacks"]["recall"]
        balanced_subtype_f1 = splits["balanced_subtype"]["f1"]
        domain = splits["domain_ood"]
        structural = splits["structural_ood"]
        note = summary["framing_note"]

    # Optional: ROC-AUC / PR-AP (only available after running eval/threshold_analysis.py
    # or eval/plot_roc_pr_curves.py; silently left as None otherwise)
    roc_pr = _load_roc_pr_for_model(spec.model_id)

    # Optional: real vs template F1 breakdown (only for internal models with source_type data)
    source_type = _load_source_type_f1(spec.model_id)

    return {
        "model_id":            spec.model_id,
        "display_name":        spec.display_name,
        "group":               spec.group,
        "kind":                spec.kind,
        "test_f1":             float(test["f1"]),
        "test_precision":      float(test["precision"]),
        "test_recall":         float(test["recall"]),
        "test_fpr":            float(test["fpr"]),
        "hard_benign_fpr":     float(hard_benign_fpr),
        "obfuscated_recall":   float(obfuscated_recall),
        "balanced_subtype_f1": float(balanced_subtype_f1),
        "domain_ood_f1":       float(domain["f1"]),
        "domain_ood_fpr":      float(domain["fpr"]),
        "structural_ood_f1":   float(structural["f1"]),
        "structural_ood_fpr":  float(structural["fpr"]),
        "note":                note,
        # Threshold-independent metrics (None until eval scripts are run)
        "roc_auc_test":        roc_pr["roc_auc_test"],
        "pr_ap_test":          roc_pr["pr_ap_test"],
        # Source-type breakdown (None for external models)
        "real_f1":             source_type.get("real_f1"),
        "template_f1":         source_type.get("template_f1"),
        "gap_tmpl_minus_real": source_type.get("gap_tmpl_minus_real"),
    }


def parse_markdown_table(md_text: str, heading: str) -> list[dict]:
    pattern = rf"### {re.escape(heading)}\n\n(?P<table>(?:\|.*\n)+)"
    match = re.search(pattern, md_text)
    if not match:
        raise RuntimeError(f"Could not find table for heading: {heading}")
    lines = [line.strip() for line in match.group("table").strip().splitlines()]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(dict(zip(header, cells)))
    return rows


def parse_core_ci() -> dict:
    if not CORE_CI_REPORT_PATH.exists():
        return {}
    text = CORE_CI_REPORT_PATH.read_text()
    hard_rows = parse_markdown_table(text, "Hard-benign FPR with 95% bootstrap CI")
    test_ci_rows = parse_markdown_table(text, "IID test F1 with 95% family-bootstrap CI")
    test_family_rows = parse_markdown_table(text, "IID test F1 with family-level sensitivity")

    data: dict[str, dict] = {}
    for row in hard_rows:
        model_id = DISPLAY_TO_ID[row["Model"]]
        low, high = parse_ci_brackets(row["95% CI"])
        data.setdefault(model_id, {})
        data[model_id]["hard_benign_fpr"] = float(row["FPR"])
        data[model_id]["hard_benign_fpr_ci_low"] = low
        data[model_id]["hard_benign_fpr_ci_high"] = high

    for row in test_ci_rows:
        model_id = DISPLAY_TO_ID[row["Model"]]
        low, high = parse_ci_brackets(row["95% CI"])
        data.setdefault(model_id, {})
        data[model_id]["test_f1"] = float(row["Row-level F1"])
        data[model_id]["test_f1_ci_low"] = low
        data[model_id]["test_f1_ci_high"] = high

    for row in test_family_rows:
        model_id = DISPLAY_TO_ID[row["Model"]]
        data.setdefault(model_id, {})
        data[model_id]["test_family_level"] = float(row["Family-level F1"])
        data[model_id]["test_family_delta"] = float(row["Delta"])

    return data


def parse_ci_brackets(text: str) -> tuple[float, float]:
    match = re.search(r"\[([0-9.]+),\s*([0-9.]+)\]", text)
    if not match:
        raise RuntimeError(f"Could not parse CI from: {text}")
    return float(match.group(1)), float(match.group(2))


def parse_ood_ci() -> dict:
    text = OOD_CI_REPORT_PATH.read_text()
    out: dict[str, dict] = {}

    domain_matches = re.findall(
        r"- ([A-Za-z0-9+ -]+): F1 `([0-9.]+)`, 95% CI `\[([0-9.]+), ([0-9.]+)\]`",
        text,
    )
    for label, value, low, high in domain_matches:
        model_id = DISPLAY_TO_ID[label.strip()]
        out.setdefault(model_id, {})
        out[model_id]["domain_ood_f1"] = float(value)
        out[model_id]["domain_ood_f1_ci_low"] = float(low)
        out[model_id]["domain_ood_f1_ci_high"] = float(high)

    structural_matches = re.findall(
        r"- ([A-Za-z0-9+ -]+): FPR `([0-9.]+)`, 95% CI `\[([0-9.]+), ([0-9.]+)\]`",
        text,
    )
    for label, value, low, high in structural_matches:
        model_id = DISPLAY_TO_ID[label.strip()]
        out.setdefault(model_id, {})
        out[model_id]["structural_ood_fpr"] = float(value)
        out[model_id]["structural_ood_fpr_ci_low"] = float(low)
        out[model_id]["structural_ood_fpr_ci_high"] = float(high)

    family_matches = re.findall(
        r"- ([A-Za-z0-9+ -]+): row FPR `([0-9.]+)` -> family FPR `([0-9.]+)`",
        text,
    )
    for label, row_value, family_value in family_matches:
        model_id = DISPLAY_TO_ID[label.strip()]
        out.setdefault(model_id, {})
        out[model_id]["structural_ood_row_fpr"] = float(row_value)
        out[model_id]["structural_ood_family_fpr"] = float(family_value)
        out[model_id]["structural_ood_family_delta"] = float(family_value) - float(row_value)

    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    align = "| " + " | ".join("---:" if key not in {"display_name", "kind", "group"} else "---" for key, _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                cells.append(format_num(value))
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, align, *body])


def build_svg_scatter(
    points: list[dict],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    subtitle: str,
    label_offsets: dict[str, tuple[int, int]],
    y_max: float,
    out_path: Path,
) -> None:
    width = 860
    height = 560
    left = 88
    right = 26
    top = 64
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = 0.0
    x_max = 1.02
    y_min = 0.0

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    elements: list[str] = []
    elements.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    elements.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    elements.append(
        f'<text x="{left}" y="28" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="700">{escape(title)}</text>'
    )
    elements.append(
        f'<text x="{left}" y="48" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#555">{escape(subtitle)}</text>'
    )

    x_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
    y_ticks = [round(y_max * i / 5, 2) for i in range(6)]

    for tick in x_ticks:
        x = sx(tick)
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e5e5e5" stroke-width="1"/>')
        elements.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#555">{tick:.2f}</text>'
        )
    for tick in y_ticks:
        y = sy(tick)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#555">{tick:.2f}</text>'
        )

    elements.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>')

    legend_x = width - 220
    legend_y = 28
    legend_items = [
        ("internal", "Internal baselines"),
        ("external_pi", "External PI detectors"),
        ("external_broad", "Broad prompt-security comparator"),
    ]
    for idx, (group, label) in enumerate(legend_items):
        y = legend_y + idx * 18
        color = GROUP_COLORS[group]
        elements.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}" stroke="#222" stroke-width="0.7"/>')
        elements.append(
            f'<text x="{legend_x + 12}" y="{y + 4}" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#333">{escape(label)}</text>'
        )

    for point in points:
        px = sx(point[x_key])
        py = sy(point[y_key])
        color = GROUP_COLORS[point["group"]]
        elements.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{color}" stroke="#222" stroke-width="0.8"/>')
        dx, dy = label_offsets.get(point["model_id"], (8, -8))
        tx = px + dx
        ty = py + dy
        anchor = "start" if dx >= 0 else "end"
        elements.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#111">{escape(point["display_name"])}</text>'
        )

    elements.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#222">{escape(x_label)}</text>'
    )
    elements.append(
        f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#222">{escape(y_label)}</text>'
    )
    elements.append("</svg>")
    out_path.write_text("\n".join(elements))


def _fmt_optional(value: float | None) -> str:
    """Format an optional float; return '—' if None."""
    if value is None:
        return "—"
    return format_num(value)


def build_report(
    focus_rows: list[dict],
    full_rows: list[dict],
    ci_rows: list[dict],
    family_rows: list[dict],
) -> str:
    # Table 1: focused comparison — include real/template F1 split for internal models,
    # and ROC-AUC where available.
    focus_columns = [
        ("display_name", "Model"),
        ("kind", "Kind"),
        ("test_f1", "Test F1"),
        ("real_f1_fmt", "Test F1 (real)"),
        ("template_f1_fmt", "Test F1 (tmpl)"),
        ("hard_benign_fpr", "Hard-benign FPR"),
        ("obfuscated_recall", "Obfuscated Recall"),
        ("domain_ood_f1", "Domain-OOD F1"),
        ("structural_ood_fpr", "Structural-OOD FPR"),
        ("roc_auc_fmt", "ROC-AUC (test)"),
    ]
    # Format optional columns before passing to markdown_table
    for r in focus_rows:
        r["real_f1_fmt"]    = _fmt_optional(r.get("real_f1"))
        r["template_f1_fmt"]= _fmt_optional(r.get("template_f1"))
        r["roc_auc_fmt"]    = _fmt_optional(r.get("roc_auc_test"))

    focus_table = markdown_table(focus_rows, focus_columns)
    full_table = markdown_table(
        full_rows,
        [
            ("display_name", "Model"),
            ("kind", "Kind"),
            ("test_f1", "Test F1"),
            ("test_fpr", "Test FPR"),
            ("balanced_subtype_f1", "Balanced-Subtype F1"),
            ("domain_ood_f1", "Domain-OOD F1"),
            ("structural_ood_f1", "Structural-OOD F1"),
            ("structural_ood_fpr", "Structural-OOD FPR"),
        ],
    )
    ci_table = markdown_table(
        ci_rows,
        [
            ("display_name", "Model"),
            ("test_f1", "Test F1"),
            ("test_f1_ci", "Test F1 95% CI"),
            ("hard_benign_fpr", "Hard-benign FPR"),
            ("hard_benign_fpr_ci", "Hard-benign FPR 95% CI"),
            ("domain_ood_f1", "Domain-OOD F1"),
            ("domain_ood_f1_ci", "Domain-OOD F1 95% CI"),
            ("structural_ood_fpr", "Structural-OOD FPR"),
            ("structural_ood_fpr_ci", "Structural-OOD FPR 95% CI"),
        ],
    )
    family_table = markdown_table(
        family_rows,
        [
            ("display_name", "Model"),
            ("test_row_f1", "Test Row F1"),
            ("test_family_f1", "Test Family F1"),
            ("test_family_delta", "Test Delta"),
            ("structural_row_fpr", "Structural Row FPR"),
            ("structural_family_fpr", "Structural Family FPR"),
            ("structural_family_delta", "Structural Delta"),
        ],
    )

    if ci_rows and family_rows:
        ci_sections = f"""
## Table 3. Internal statistical support

This table is the clean support table for the benchmark's main claim. It keeps the CI evidence focused on the internal baselines where family-level analysis was completed.

{ci_table}

## Table 4. Family-level sensitivity

This table shows that the main deployment-safety result is not explained away by repeated paraphrase rows.

{family_table}
"""
        ci_inputs_note = "- local family-bootstrap CI intermediates (used to refresh Tables 3 and 4)"
    else:
        ci_sections = """
## Internal statistical support

Tables 3 and 4 require local family-bootstrap CI intermediates that are not part of the public repo. The checked-in report may include those consolidated values, but they are optional for rebuilding the public artifacts.
"""
        ci_inputs_note = "- local family-bootstrap CI intermediates (optional; not tracked in the public repo)"

    return f"""# Final Results Tables

These tables and figures are built directly from the saved summaries under `outputs/`. Family-level support tables additionally use local CI intermediates during paper production.

- `outputs/rule_baseline/summary.json`
- `outputs/tfidf_baseline/summary.json`
- `outputs/distilbert_baseline/summary.json`
- `outputs/deberta_v3/summary.json`
- `outputs/external_eval/multiaxis/*/summary.json`
- `reports/FAMILY_CI_OOD_RESULTS.md`
{ci_inputs_note}

## Table 1. Focused detector comparison

This is the clearest main-table candidate for the paper because it keeps attention on the benchmark's central lesson: high IID performance does not guarantee benign-side safety or structural robustness.

{focus_table}

## Table 2. Extended multiaxis comparison

This table is useful as the fuller main-results table or as a compact appendix table.

{full_table}
{ci_sections}

## Figure outputs

- Main tradeoff figure: `outputs/paper_final_assets/figure_test_f1_vs_hard_benign_fpr.svg`
- Secondary tradeoff figure: `outputs/paper_final_assets/figure_test_f1_vs_structural_ood_fpr.svg`

## Paper-facing lesson

- The benchmark now teaches two distinct detector failures: over-defense on benign stress inputs and under-detection when the model becomes overly conservative.
- `DeBERTa-v3` is still the strongest IID model, but that does not protect it from severe hard-benign and structural benign false positives.
- `PromptGuard 2 (86M)` adds value because it exposes the opposite tradeoff: lower benign-side false positives, but heavy attack under-detection.
- The paper can now show that detector evaluation is not a one-number leaderboard problem; it is a tradeoff and failure-mode characterization problem.
"""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for spec in MODEL_SPECS:
        if not spec.summary_path.exists():
            print(f"[SKIP] {spec.model_id}: summary not found at "
                  f"{spec.summary_path.relative_to(ROOT)} — run the baseline first.")
            continue
        records.append(build_model_record(spec))
    core_ci = parse_core_ci()
    ood_ci = parse_ood_ci()

    focus_rows = [
        {
            "display_name":    r["display_name"],
            "kind":            r["kind"],
            "group":           r["group"],
            "model_id":        r["model_id"],
            "test_f1":         r["test_f1"],
            "real_f1":         r.get("real_f1"),
            "template_f1":     r.get("template_f1"),
            "gap_tmpl_minus_real": r.get("gap_tmpl_minus_real"),
            "hard_benign_fpr": r["hard_benign_fpr"],
            "obfuscated_recall": r["obfuscated_recall"],
            "domain_ood_f1":   r["domain_ood_f1"],
            "structural_ood_fpr": r["structural_ood_fpr"],
            "roc_auc_test":    r.get("roc_auc_test"),
            "pr_ap_test":      r.get("pr_ap_test"),
        }
        for r in records
    ]
    full_rows = [
        {
            "display_name": r["display_name"],
            "kind": r["kind"],
            "test_f1": r["test_f1"],
            "test_fpr": r["test_fpr"],
            "balanced_subtype_f1": r["balanced_subtype_f1"],
            "domain_ood_f1": r["domain_ood_f1"],
            "structural_ood_f1": r["structural_ood_f1"],
            "structural_ood_fpr": r["structural_ood_fpr"],
        }
        for r in records
    ]

    ci_rows = []
    family_rows = []
    if core_ci:
        for spec in MODEL_SPECS[:4]:
            ci = core_ci[spec.model_id]
            ood = ood_ci[spec.model_id]
            ci_rows.append(
                {
                    "display_name": spec.display_name,
                    "test_f1": ci["test_f1"],
                    "test_f1_ci": f"[{format_num(ci['test_f1_ci_low'])}, {format_num(ci['test_f1_ci_high'])}]",
                    "hard_benign_fpr": ci["hard_benign_fpr"],
                    "hard_benign_fpr_ci": f"[{format_num(ci['hard_benign_fpr_ci_low'])}, {format_num(ci['hard_benign_fpr_ci_high'])}]",
                    "domain_ood_f1": ood["domain_ood_f1"],
                    "domain_ood_f1_ci": f"[{format_num(ood['domain_ood_f1_ci_low'])}, {format_num(ood['domain_ood_f1_ci_high'])}]",
                    "structural_ood_fpr": ood["structural_ood_fpr"],
                    "structural_ood_fpr_ci": f"[{format_num(ood['structural_ood_fpr_ci_low'])}, {format_num(ood['structural_ood_fpr_ci_high'])}]",
                }
            )
            family_rows.append(
                {
                    "display_name": spec.display_name,
                    "test_row_f1": ci["test_f1"],
                    "test_family_f1": ci["test_family_level"],
                    "test_family_delta": ci["test_family_delta"],
                    "structural_row_fpr": ood["structural_ood_row_fpr"],
                    "structural_family_fpr": ood["structural_ood_family_fpr"],
                    "structural_family_delta": ood["structural_ood_family_delta"],
                }
            )

    write_csv(
        OUT_DIR / "table_1_detector_focus.csv",
        focus_rows,
        [
            "display_name", "kind", "group", "model_id", "test_f1",
            "real_f1", "template_f1", "gap_tmpl_minus_real",
            "hard_benign_fpr", "obfuscated_recall",
            "domain_ood_f1", "structural_ood_fpr",
            "roc_auc_test", "pr_ap_test",
        ],
    )
    write_csv(
        OUT_DIR / "table_2_multiaxis_extended.csv",
        full_rows,
        ["display_name", "kind", "test_f1", "test_fpr", "balanced_subtype_f1", "domain_ood_f1", "structural_ood_f1", "structural_ood_fpr"],
    )
    if ci_rows:
        write_csv(
            OUT_DIR / "table_3_internal_ci.csv",
            ci_rows,
            ["display_name", "test_f1", "test_f1_ci", "hard_benign_fpr", "hard_benign_fpr_ci", "domain_ood_f1", "domain_ood_f1_ci", "structural_ood_fpr", "structural_ood_fpr_ci"],
        )
    if family_rows:
        write_csv(
            OUT_DIR / "table_4_family_sensitivity.csv",
            family_rows,
            ["display_name", "test_row_f1", "test_family_f1", "test_family_delta", "structural_row_fpr", "structural_family_fpr", "structural_family_delta"],
        )
    write_csv(
        OUT_DIR / "tradeoff_points.csv",
        records,
        [
            "model_id",
            "display_name",
            "group",
            "kind",
            "test_f1",
            "test_precision",
            "test_recall",
            "test_fpr",
            "real_f1",
            "template_f1",
            "gap_tmpl_minus_real",
            "hard_benign_fpr",
            "obfuscated_recall",
            "balanced_subtype_f1",
            "domain_ood_f1",
            "domain_ood_fpr",
            "structural_ood_f1",
            "structural_ood_fpr",
            "roc_auc_test",
            "pr_ap_test",
            "note",
        ],
    )

    build_svg_scatter(
        records,
        x_key="test_f1",
        y_key="hard_benign_fpr",
        x_label="IID test F1",
        y_label="Hard-benign false-positive rate",
        title="Detector Tradeoff: IID Test F1 vs Hard-benign FPR",
        subtitle="Higher F1 is better; lower hard-benign FPR is better. This is the clearest single-figure statement of the benchmark claim.",
        label_offsets=HB_LABEL_OFFSETS,
        y_max=0.50,
        out_path=OUT_DIR / "figure_test_f1_vs_hard_benign_fpr.svg",
    )
    build_svg_scatter(
        records,
        x_key="test_f1",
        y_key="structural_ood_fpr",
        x_label="IID test F1",
        y_label="Structural-OOD benign false-positive rate",
        title="Detector Tradeoff: IID Test F1 vs Structural-OOD Benign FPR",
        subtitle="Higher F1 is better; lower structural benign FPR is better. This figure is useful as the secondary deployment-shift tradeoff plot.",
        label_offsets=STRUCT_LABEL_OFFSETS,
        y_max=0.90,
        out_path=OUT_DIR / "figure_test_f1_vs_structural_ood_fpr.svg",
    )

    report_text = build_report(focus_rows, full_rows, ci_rows, family_rows)
    REPORT_PATH.write_text(report_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
