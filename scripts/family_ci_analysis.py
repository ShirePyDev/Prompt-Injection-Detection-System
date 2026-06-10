#!/usr/bin/env python3
"""
Family-level sensitivity and confidence-interval analysis for PIDS-Bench v3.

Purpose:
  - quantify how much repeated paraphrase / variant families affect reported metrics
  - add bootstrap confidence intervals without changing the frozen benchmark

Outputs under outputs/family_ci_analysis/:
  - family_ci_metrics.json
  - primary_metric_table.csv
  - family_effect_table.csv
  - FAMILY_CI_REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "family_ci_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.rule_based import rule_score

BOOTSTRAP_REPS = 1000
RANDOM_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    backend: str
    score_threshold: float
    model_path: Path | None = None
    max_length: int | None = None
    batch_size: int | None = None


@dataclass(frozen=True)
class SplitSpec:
    split_id: str
    csv_path: Path
    kind: str
    primary_metric: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "rule_based": ModelSpec(
        model_id="rule_based",
        backend="rule",
        score_threshold=0.5,
    ),
    "tfidf_logreg": ModelSpec(
        model_id="tfidf_logreg",
        backend="tfidf",
        score_threshold=0.518,
        model_path=ROOT / "models" / "tfidf_logreg" / "tfidf_logreg.joblib",
    ),
    "distilbert": ModelSpec(
        model_id="distilbert",
        backend="hf_classifier",
        score_threshold=0.45,
        model_path=ROOT / "models" / "distilbert",
        max_length=256,
        batch_size=32,
    ),
    "deberta_v3": ModelSpec(
        model_id="deberta_v3",
        backend="hf_classifier",
        score_threshold=0.502,
        model_path=ROOT / "models" / "deberta_v3",
        max_length=512,
        batch_size=8,
    ),
}


SPLIT_SPECS: list[SplitSpec] = [
    SplitSpec("test", ROOT / "data" / "pids_bench_v3" / "test.csv", "binary", "f1"),
    SplitSpec(
        "hard_benign",
        ROOT / "data" / "pids_bench_v3" / "eval_subsets" / "hard_benign_test.csv",
        "benign_only",
        "fpr",
    ),
    SplitSpec(
        "balanced_subtype",
        ROOT / "data" / "pids_bench_v3" / "eval_subsets" / "balanced_subtype_test.csv",
        "binary",
        "f1",
    ),
    SplitSpec(
        "obfuscated_attacks",
        ROOT / "data" / "pids_bench_v3" / "eval_subsets" / "obfuscated_attacks.csv",
        "injection_only",
        "recall",
    ),
    SplitSpec(
        "domain_ood",
        ROOT / "data" / "pids_bench_v3" / "ood" / "domain_ood.csv",
        "binary",
        "f1",
    ),
    SplitSpec(
        "structural_ood",
        ROOT / "data" / "pids_bench_v3" / "ood" / "structural_ood.csv",
        "binary",
        "fpr",
    ),
]

EXTRA_METRICS = ("f1", "fpr", "fnr", "recall")

SAVED_PREDICTION_PATHS: dict[tuple[str, str], Path] = {
    ("distilbert", "test"): ROOT / "outputs" / "distilbert_baseline" / "test_predictions.csv",
    ("distilbert", "hard_benign"): ROOT / "outputs" / "distilbert_baseline" / "hard_benign_predictions.csv",
    ("deberta_v3", "test"): ROOT / "outputs" / "deberta_v3" / "test_predictions.csv",
    ("deberta_v3", "hard_benign"): ROOT / "outputs" / "deberta_v3" / "hard_benign_predictions.csv",
}

BREAKDOWN_SCORE_CACHE_DIR = ROOT / "outputs" / "breakdown_analysis" / "score_cache"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def metrics_for_kind(y_true: np.ndarray, scores: np.ndarray, threshold: float, kind: str) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (scores >= threshold).astype(int)

    if kind == "benign_only":
        fp = int((y_pred == 1).sum())
        n = len(y_pred)
        return {
            "n": n,
            "fpr": safe_div(fp, n),
            "n_flagged": fp,
        }

    if kind == "injection_only":
        tp = int((y_pred == 1).sum())
        n = len(y_pred)
        recall = safe_div(tp, n)
        return {
            "n": n,
            "recall": recall,
            "fnr": 1.0 - recall,
            "n_detected": tp,
        }

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = len(y_true)
    return {
        "n": n,
        "accuracy": safe_div(tp + tn, n),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
    }


def family_key_for_split(df: pd.DataFrame, split_id: str) -> pd.Series | None:
    if "parent_seed_id" in df.columns:
        return df["parent_seed_id"].astype(str)
    if split_id == "structural_ood" and "original_seed_idx" in df.columns:
        return df["label"].astype(str) + ":" + df["original_seed_idx"].astype(str)
    return None


def bootstrap_ci(
    frame: pd.DataFrame,
    threshold: float,
    kind: str,
    metric: str,
    family_col: str | None,
    reps: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    stats: list[float] = []

    if family_col is not None:
        groups = list(frame.groupby(family_col).indices.values())
        n_units = len(groups)
        for _ in range(reps):
            chosen = rng.integers(0, n_units, size=n_units)
            idx = np.concatenate([np.asarray(groups[i], dtype=int) for i in chosen])
            sample = frame.iloc[idx]
            m = metrics_for_kind(
                sample["label"].to_numpy(),
                sample["score"].to_numpy(),
                threshold,
                kind,
            )
            stats.append(float(m[metric]))
        unit_type = "family"
        n_boot_units = n_units
    else:
        n_rows = len(frame)
        for _ in range(reps):
            idx = rng.integers(0, n_rows, size=n_rows)
            sample = frame.iloc[idx]
            m = metrics_for_kind(
                sample["label"].to_numpy(),
                sample["score"].to_numpy(),
                threshold,
                kind,
            )
            stats.append(float(m[metric]))
        unit_type = "row"
        n_boot_units = n_rows

    arr = np.asarray(stats, dtype=float)
    return {
        "metric": metric,
        "unit_type": unit_type,
        "n_boot_units": n_boot_units,
        "mean": float(arr.mean()),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
    }


def family_collapsed_frame(frame: pd.DataFrame, family_col: str) -> pd.DataFrame:
    agg = (
        frame.groupby(family_col, as_index=False)
        .agg(
            label=("label", "first"),
            score=("score", "mean"),
            n_rows=("text", "size"),
        )
        .rename(columns={family_col: "family_id"})
    )
    return agg


def infer_rule(df: pd.DataFrame) -> np.ndarray:
    return np.asarray([float(rule_score(t)) for t in df["text"].astype(str).tolist()], dtype=float)


def infer_tfidf(df: pd.DataFrame, pipe) -> np.ndarray:
    return pipe.predict_proba(df["text"].astype(str))[:, 1].astype(float)


@torch.inference_mode()
def infer_hf_classifier(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    texts = df["text"].astype(str).tolist()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        scores.append(probs.astype(float))
    return np.concatenate(scores, axis=0)


def load_hf_model(spec: ModelSpec, device: torch.device):
    assert spec.model_path is not None
    tokenizer = AutoTokenizer.from_pretrained(spec.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(spec.model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def load_saved_scores(model_name: str, split_id: str, df: pd.DataFrame) -> np.ndarray | None:
    path = SAVED_PREDICTION_PATHS.get((model_name, split_id))
    if path is not None and path.exists():
        pred_df = pd.read_csv(path)
        if "probability_score" in pred_df.columns and len(pred_df) == len(df):
            if "text" not in pred_df.columns or pred_df["text"].astype(str).tolist() == df["text"].astype(str).tolist():
                return pred_df["probability_score"].to_numpy(dtype=float)

    cache_path = BREAKDOWN_SCORE_CACHE_DIR / f"{model_name}__{split_id}_scores.csv"
    if not cache_path.exists():
        return None
    cached_df = pd.read_csv(cache_path)
    if "score" not in cached_df.columns or len(cached_df) != len(df):
        return None
    if "text" in cached_df.columns:
        if cached_df["text"].astype(str).tolist() != df["text"].astype(str).tolist():
            return None
    return cached_df["score"].to_numpy(dtype=float)


def primary_metric_sentence(split_id: str) -> str:
    if split_id == "test":
        return "IID held-out performance (F1)"
    if split_id == "hard_benign":
        return "over-defense on benign-only stress prompts (FPR)"
    if split_id == "balanced_subtype":
        return "subtype-balance diagnostic (F1)"
    if split_id == "obfuscated_attacks":
        return "detection rate on obfuscated injections (recall)"
    if split_id == "domain_ood":
        return "domain-shift performance (F1)"
    if split_id == "structural_ood":
        return "benign false-positive stress under structural shift (FPR)"
    return split_id


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Family-level sensitivity and CI analysis")
    ap.add_argument(
        "--splits",
        default="all",
        help="Comma-separated split ids or 'all'",
    )
    ap.add_argument(
        "--models",
        default="all",
        help="Comma-separated model ids or 'all'",
    )
    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=BOOTSTRAP_REPS,
        help=f"Bootstrap repetitions (default: {BOOTSTRAP_REPS})",
    )
    return ap.parse_args()


def run() -> None:
    args = parse_args()
    device = pick_device()
    tfidf_pipe = None
    hf_cache: dict[str, tuple[AutoTokenizer, AutoModelForSequenceClassification]] = {}
    split_index = {spec.split_id: i for i, spec in enumerate(SPLIT_SPECS)}
    model_index = {name: i for i, name in enumerate(MODEL_SPECS)}

    wanted_models = list(MODEL_SPECS) if args.models == "all" else [x.strip() for x in args.models.split(",") if x.strip()]
    wanted_splits = {spec.split_id for spec in SPLIT_SPECS} if args.splits == "all" else {x.strip() for x in args.splits.split(",") if x.strip()}
    split_specs = [spec for spec in SPLIT_SPECS if spec.split_id in wanted_splits]

    metrics_blob: dict[str, dict] = {}
    primary_rows: list[dict] = []
    family_effect_rows: list[dict] = []

    for model_name, spec in MODEL_SPECS.items():
        if model_name not in wanted_models:
            continue
        metrics_blob[model_name] = {
            "threshold": spec.score_threshold,
            "splits": {},
        }

        if spec.backend == "tfidf":
            assert spec.model_path is not None
            tfidf_pipe = joblib.load(spec.model_path)

        for split in split_specs:
            df = pd.read_csv(split.csv_path).copy()
            df["text"] = df["text"].astype(str)
            df["label"] = df["label"].astype(int)

            saved_scores = load_saved_scores(model_name, split.split_id, df)
            if saved_scores is not None:
                scores = saved_scores
            elif spec.backend == "rule":
                scores = infer_rule(df)
            elif spec.backend == "tfidf":
                assert tfidf_pipe is not None
                scores = infer_tfidf(df, tfidf_pipe)
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

            work = df.copy()
            work["score"] = scores
            fam_key = family_key_for_split(work, split.split_id)
            if fam_key is not None:
                work["family_id"] = fam_key

            row_metrics = metrics_for_kind(
                work["label"].to_numpy(),
                work["score"].to_numpy(),
                spec.score_threshold,
                split.kind,
            )

            primary_ci = bootstrap_ci(
                work,
                spec.score_threshold,
                split.kind,
                split.primary_metric,
                "family_id" if "family_id" in work.columns else None,
                reps=args.bootstrap_reps,
                seed=RANDOM_SEED,
            )

            extra_ci: dict[str, dict] = {}
            if split.kind == "binary":
                for metric_idx, extra_metric in enumerate(EXTRA_METRICS):
                    extra_ci[extra_metric] = bootstrap_ci(
                        work,
                        spec.score_threshold,
                        split.kind,
                        extra_metric,
                        "family_id" if "family_id" in work.columns else None,
                        reps=args.bootstrap_reps,
                        seed=RANDOM_SEED + 100 * model_index[model_name] + 10 * split_index[split.split_id] + metric_idx,
                    )

            family_metrics = None
            family_ci = None
            n_families = None
            if "family_id" in work.columns:
                fam = family_collapsed_frame(work, "family_id")
                n_families = int(len(fam))
                family_metrics = metrics_for_kind(
                    fam["label"].to_numpy(),
                    fam["score"].to_numpy(),
                    spec.score_threshold,
                    split.kind,
                )
                family_ci = bootstrap_ci(
                    fam,
                    spec.score_threshold,
                    split.kind,
                    split.primary_metric,
                    family_col=None,
                    reps=args.bootstrap_reps,
                    seed=RANDOM_SEED,
                )

            metrics_blob[model_name]["splits"][split.split_id] = {
                "row_level": row_metrics,
                "row_level_primary_ci": primary_ci,
                "row_level_extra_ci": extra_ci,
                "family_level": family_metrics,
                "family_level_primary_ci": family_ci,
                "n_rows": int(len(work)),
                "n_families": n_families,
                "bootstrap_unit": primary_ci["unit_type"],
            }

            primary_rows.append(
                {
                    "model": model_name,
                    "split": split.split_id,
                    "primary_metric": split.primary_metric,
                    "primary_metric_role": primary_metric_sentence(split.split_id),
                    "row_level": row_metrics.get(split.primary_metric),
                    "row_ci95_low": primary_ci["ci95_low"],
                    "row_ci95_high": primary_ci["ci95_high"],
                    "bootstrap_unit": primary_ci["unit_type"],
                    "n_rows": len(work),
                    "n_families": n_families,
                    "family_level": None if family_metrics is None else family_metrics.get(split.primary_metric),
                    "family_ci95_low": None if family_ci is None else family_ci["ci95_low"],
                    "family_ci95_high": None if family_ci is None else family_ci["ci95_high"],
                }
            )

            if family_metrics is not None:
                family_effect_rows.append(
                    {
                        "model": model_name,
                        "split": split.split_id,
                        "metric": split.primary_metric,
                        "row_level": row_metrics.get(split.primary_metric),
                        "family_level": family_metrics.get(split.primary_metric),
                        "delta_family_minus_row": family_metrics.get(split.primary_metric) - row_metrics.get(split.primary_metric),  # type: ignore
                        "n_rows": len(work),
                        "n_families": n_families,
                    }
                )

    with open(OUT_DIR / "family_ci_metrics.json", "w") as f:
        json.dump(metrics_blob, f, indent=2)

    primary_df = pd.DataFrame(primary_rows).sort_values(["split", "model"]).reset_index(drop=True)
    primary_df.to_csv(OUT_DIR / "primary_metric_table.csv", index=False)

    family_effect_df = pd.DataFrame(family_effect_rows).sort_values(["split", "model"]).reset_index(drop=True)
    family_effect_df.to_csv(OUT_DIR / "family_effect_table.csv", index=False)

    report_lines = [
        "# Family-Level Sensitivity and Confidence Intervals",
        "",
        "Generated from the frozen benchmark and saved in-repo models/checkpoints.",
        "",
        "Method:",
        "- Row-level point metrics use the same val-tuned thresholds as the baseline reports.",
        "- Confidence intervals use bootstrap resampling.",
        "- Where family identifiers exist, bootstrap resamples at the family level rather than the row level.",
        "- Family-level sensitivity collapses each family to the mean attack score and thresholds once per family.",
        "- `structural_ood` uses `label:original_seed_idx` as the family key because `original_seed_idx` is shared across both classes.",
        "- `domain_ood` has no family identifier in the released CSV, so its CI uses row-level bootstrap.",
        "- Where saved transformer prediction files already exist (`test`, `hard_benign`), the script reuses them instead of rerunning inference.",
        "",
        "Primary metrics by split:",
        "- `test`: F1",
        "- `hard_benign`: FPR",
        "- `balanced_subtype`: F1",
        "- `obfuscated_attacks`: recall",
        "- `domain_ood`: F1",
        "- `structural_ood`: FPR",
        "",
        "See `primary_metric_table.csv` and `family_effect_table.csv` for the machine-readable outputs.",
    ]
    (OUT_DIR / "FAMILY_CI_REPORT.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    run()
