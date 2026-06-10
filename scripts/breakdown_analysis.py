#!/usr/bin/env python3
"""
Breakdown analysis for PIDS-Bench v3.

Purpose:
  - show where detector failures concentrate, not just whether aggregate metrics are high
  - keep the same frozen splits, thresholds, and model backends used in the main benchmark

Outputs under outputs/breakdown_analysis/:
  - test_source_type_metrics.csv
  - test_attack_type_recall.csv
  - test_attack_type_source_mix.csv
  - domain_ood_by_domain.csv
  - structural_ood_by_transform.csv
  - obfuscated_attacks_by_obfuscation.csv
  - BREAKDOWN_ANALYSIS_REPORT.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "breakdown_analysis"
CACHE_DIR = OUT_DIR / "score_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.family_ci_analysis import (  # noqa: E402
    MODEL_SPECS,
    infer_hf_classifier,
    infer_rule,
    infer_tfidf,
    load_hf_model,
    load_saved_scores,
    metrics_for_kind,
    pick_device,
)


SPLIT_PATHS: dict[str, Path] = {
    "test": ROOT / "data" / "pids_bench_v3" / "test.csv",
    "domain_ood": ROOT / "data" / "pids_bench_v3" / "ood" / "domain_ood.csv",
    "structural_ood": ROOT / "data" / "pids_bench_v3" / "ood" / "structural_ood.csv",
    "obfuscated_attacks": ROOT / "data" / "pids_bench_v3" / "eval_subsets" / "obfuscated_attacks.csv",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Breakdown analysis for PIDS-Bench v3")
    ap.add_argument(
        "--models",
        default="all",
        help="Comma-separated model ids or 'all'",
    )
    return ap.parse_args()


def load_split(split_id: str) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_PATHS[split_id]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def cache_path(model_name: str, split_id: str) -> Path:
    return CACHE_DIR / f"{model_name}__{split_id}_scores.csv"


def load_cached_scores(model_name: str, split_id: str, df: pd.DataFrame) -> np.ndarray | None:
    path = cache_path(model_name, split_id)
    if not path.exists():
        return None
    cached = pd.read_csv(path)
    if "score" not in cached.columns or len(cached) != len(df):
        return None
    if cached["text"].astype(str).tolist() != df["text"].astype(str).tolist():
        return None
    return cached["score"].to_numpy(dtype=float)


def save_cached_scores(model_name: str, split_id: str, df: pd.DataFrame, scores: np.ndarray) -> None:
    pd.DataFrame(
        {
            "text": df["text"].astype(str),
            "label": df["label"].astype(int),
            "score": scores.astype(float),
        }
    ).to_csv(cache_path(model_name, split_id), index=False)


def get_scores(
    model_name: str,
    split_id: str,
    df: pd.DataFrame,
    device: torch.device,
    tfidf_cache: dict[str, object],
    hf_cache: dict[str, tuple[object, object]],
) -> np.ndarray:
    saved = load_saved_scores(model_name, split_id, df)
    if saved is not None:
        return saved

    cached = load_cached_scores(model_name, split_id, df)
    if cached is not None:
        return cached

    spec = MODEL_SPECS[model_name]
    if spec.backend == "rule":
        scores = infer_rule(df)
    elif spec.backend == "tfidf":
        if "pipe" not in tfidf_cache:
            assert spec.model_path is not None
            tfidf_cache["pipe"] = joblib.load(spec.model_path)
        scores = infer_tfidf(df, tfidf_cache["pipe"])
    else:
        if model_name not in hf_cache:
            hf_cache[model_name] = load_hf_model(spec, device)
        tok, model = hf_cache[model_name]
        scores = infer_hf_classifier(
            df,
            tok,
            model,
            device,
            batch_size=spec.batch_size or 16,
            max_length=spec.max_length or 512,
        )

    save_cached_scores(model_name, split_id, df, scores)
    return scores


def add_scores(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["score"] = scores.astype(float)
    return out


def binary_group_rows(df: pd.DataFrame, group_col: str, threshold: float) -> list[dict]:
    rows: list[dict] = []
    for group_value, group in df.groupby(group_col, dropna=False):
        group = group.reset_index(drop=True)
        metrics = metrics_for_kind(
            group["label"].to_numpy(),
            group["score"].to_numpy(),
            threshold,
            "binary",
        )
        benign_mask = group["label"].to_numpy() == 0
        attack_mask = group["label"].to_numpy() == 1
        flagged = group["score"].to_numpy() >= threshold
        benign_flagged = int(flagged[benign_mask].sum()) if benign_mask.any() else 0
        attack_detected = int(flagged[attack_mask].sum()) if attack_mask.any() else 0

        rows.append(
            {
                "group": str(group_value),
                "n": int(len(group)),
                "n_benign": int(benign_mask.sum()),
                "n_injection": int(attack_mask.sum()),
                "positive_rate": float(group["label"].mean()),
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "fpr": float(metrics["fpr"]),
                "fnr": float(metrics["fnr"]),
                "benign_flagged": benign_flagged,
                "attack_detected": attack_detected,
            }
        )
    return rows


def positive_group_rows(df: pd.DataFrame, group_col: str, threshold: float) -> list[dict]:
    attack_df = df[df["label"] == 1].copy()
    rows: list[dict] = []
    for group_value, group in attack_df.groupby(group_col, dropna=False):
        detected = int((group["score"].to_numpy() >= threshold).sum())
        n = int(len(group))
        rows.append(
            {
                "group": str(group_value),
                "n": n,
                "n_detected": detected,
                "recall": float(detected / n) if n else 0.0,
                "fnr": float(1.0 - (detected / n)) if n else 0.0,
            }
        )
    return rows


def attack_type_source_mix_rows(df: pd.DataFrame) -> list[dict]:
    attack_df = df[df["label"] == 1].copy()
    rows: list[dict] = []
    for attack_type, group in attack_df.groupby("attack_type", dropna=False):
        source_counts = group["source_type"].value_counts()
        real_rows = int(source_counts.get("real", 0))
        template_rows = int(source_counts.get("template", 0))
        total = int(len(group))
        rows.append(
            {
                "attack_type": str(attack_type),
                "n": total,
                "real_rows": real_rows,
                "template_rows": template_rows,
                "real_share": float(real_rows / total) if total else 0.0,
                "template_share": float(template_rows / total) if total else 0.0,
            }
        )
    return rows


def write_report(
    source_df: pd.DataFrame,
    attack_recall_df: pd.DataFrame,
    attack_mix_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    structural_df: pd.DataFrame,
    obf_df: pd.DataFrame,
) -> None:
    def top_row(df: pd.DataFrame, *, model: str, sort_col: str, ascending: bool) -> pd.Series:
        sub = df[df["model"] == model].sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        return sub.iloc[0]

    lines: list[str] = [
        "# Breakdown Analysis: Core Findings",
        "",
        "This report decomposes aggregate benchmark performance into source-type, domain, structural-transform, subtype, and obfuscation views.",
        "The purpose is to show where errors concentrate, not to add another aggregate headline metric.",
        "",
        "## Source-Type Gap On The IID Test Split",
        "",
    ]

    for model in source_df["model"].drop_duplicates().tolist():
        sub = source_df[source_df["model"] == model].set_index("source_type")
        if {"real", "template"}.issubset(sub.index):
            real_f1 = float(sub.loc["real", "f1"])
            template_f1 = float(sub.loc["template", "f1"])
            lines.append(
                f"- `{model}`: real-source F1 `{real_f1:.4f}` vs template-source F1 `{template_f1:.4f}`"
            )

    lines.extend(
        [
            "",
            "## Domain OOD Heterogeneity",
            "",
        ]
    )
    for model in domain_df["model"].drop_duplicates().tolist():
        worst = top_row(domain_df, model=model, sort_col="f1", ascending=True)
        lines.append(
            f"- `{model}` weakest domain: `{worst['domain']}` with F1 `{worst['f1']:.4f}`, benign FPR `{worst['fpr']:.4f}`, attack recall `{worst['recall']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## Structural OOD Concentration",
            "",
        ]
    )
    for model in structural_df["model"].drop_duplicates().tolist():
        worst = top_row(structural_df, model=model, sort_col="fpr", ascending=False)
        lines.append(
            f"- `{model}` highest structural benign FPR: `{worst['transform_type']}` at `{worst['fpr']:.4f}` with attack recall `{worst['recall']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## Attack-Type Composition",
            "",
            "Real/template mix on attack rows from the IID test split:",
            "",
        ]
    )
    for _, row in attack_mix_df.sort_values("real_share").iterrows():
        lines.append(
            f"- `{row['attack_type']}`: {int(row['real_rows'])} real / {int(row['template_rows'])} template (real share `{row['real_share']:.3f}`)"
        )

    lines.extend(
        [
            "",
            "## Obfuscation Diagnostic",
            "",
        ]
    )
    for model in obf_df["model"].drop_duplicates().tolist():
        worst = top_row(obf_df, model=model, sort_col="recall", ascending=True)
        best = top_row(obf_df, model=model, sort_col="recall", ascending=False)
        lines.append(
            f"- `{model}` recall ranges from `{worst['recall']:.4f}` on `{worst['obfuscation']}` to `{best['recall']:.4f}` on `{best['obfuscation']}`"
        )

    lines.extend(
        [
            "",
            "Machine-readable tables:",
            "- `test_source_type_metrics.csv`",
            "- `test_attack_type_recall.csv`",
            "- `test_attack_type_source_mix.csv`",
            "- `domain_ood_by_domain.csv`",
            "- `structural_ood_by_transform.csv`",
            "- `obfuscated_attacks_by_obfuscation.csv`",
        ]
    )

    (OUT_DIR / "BREAKDOWN_ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    wanted_models = list(MODEL_SPECS) if args.models == "all" else [x.strip() for x in args.models.split(",") if x.strip()]
    device = pick_device()
    tfidf_cache: dict[str, object] = {}
    hf_cache: dict[str, tuple[object, object]] = {}

    frames = {split_id: load_split(split_id) for split_id in SPLIT_PATHS}

    test_source_rows: list[dict] = []
    test_attack_recall_rows: list[dict] = []
    domain_rows: list[dict] = []
    structural_rows: list[dict] = []
    obf_rows: list[dict] = []

    test_attack_mix_df = pd.DataFrame(attack_type_source_mix_rows(frames["test"])).sort_values("attack_type").reset_index(drop=True)
    test_attack_mix_df.to_csv(OUT_DIR / "test_attack_type_source_mix.csv", index=False)

    for model_name in wanted_models:
        spec = MODEL_SPECS[model_name]

        test_scores = get_scores(model_name, "test", frames["test"], device, tfidf_cache, hf_cache)
        test_df = add_scores(frames["test"], test_scores)
        for row in binary_group_rows(test_df, "source_type", spec.score_threshold):
            row["model"] = model_name
            row["source_type"] = row.pop("group")
            test_source_rows.append(row)
        for row in positive_group_rows(test_df, "attack_type", spec.score_threshold):
            row["model"] = model_name
            row["attack_type"] = row.pop("group")
            test_attack_recall_rows.append(row)

        domain_scores = get_scores(model_name, "domain_ood", frames["domain_ood"], device, tfidf_cache, hf_cache)
        domain_df = add_scores(frames["domain_ood"], domain_scores)
        for row in binary_group_rows(domain_df, "domain", spec.score_threshold):
            row["model"] = model_name
            row["domain"] = row.pop("group")
            domain_rows.append(row)

        structural_scores = get_scores(model_name, "structural_ood", frames["structural_ood"], device, tfidf_cache, hf_cache)
        structural_df = add_scores(frames["structural_ood"], structural_scores)
        for row in binary_group_rows(structural_df, "transform_type", spec.score_threshold):
            row["model"] = model_name
            row["transform_type"] = row.pop("group")
            structural_rows.append(row)

        obf_scores = get_scores(model_name, "obfuscated_attacks", frames["obfuscated_attacks"], device, tfidf_cache, hf_cache)
        obf_df = add_scores(frames["obfuscated_attacks"], obf_scores)
        for row in positive_group_rows(obf_df, "obfuscation", spec.score_threshold):
            row["model"] = model_name
            row["obfuscation"] = row.pop("group")
            obf_rows.append(row)

    test_source_df = pd.DataFrame(test_source_rows).sort_values(["model", "source_type"]).reset_index(drop=True)
    test_source_df.to_csv(OUT_DIR / "test_source_type_metrics.csv", index=False)

    test_attack_recall_df = pd.DataFrame(test_attack_recall_rows).sort_values(["model", "attack_type"]).reset_index(drop=True)
    test_attack_recall_df.to_csv(OUT_DIR / "test_attack_type_recall.csv", index=False)

    domain_breakdown_df = pd.DataFrame(domain_rows).sort_values(["model", "domain"]).reset_index(drop=True)
    domain_breakdown_df.to_csv(OUT_DIR / "domain_ood_by_domain.csv", index=False)

    structural_breakdown_df = pd.DataFrame(structural_rows).sort_values(["model", "transform_type"]).reset_index(drop=True)
    structural_breakdown_df.to_csv(OUT_DIR / "structural_ood_by_transform.csv", index=False)

    obf_breakdown_df = pd.DataFrame(obf_rows).sort_values(["model", "obfuscation"]).reset_index(drop=True)
    obf_breakdown_df.to_csv(OUT_DIR / "obfuscated_attacks_by_obfuscation.csv", index=False)

    write_report(
        test_source_df,
        test_attack_recall_df,
        test_attack_mix_df,
        domain_breakdown_df,
        structural_breakdown_df,
        obf_breakdown_df,
    )


if __name__ == "__main__":
    main()
