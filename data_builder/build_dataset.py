"""Build pipeline entrypoint for PIDS-Bench."""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY from .env before paraphrase stage

from data_builder.build_eval_subsets import run as run_build_eval_subsets
from data_builder.generators.realize import paraphrase_seeds
from data_builder.sources import collect_seeds, audit_seed_sample, load_config
from data_builder.transforms import obfuscate
from data_builder.utils import dedup, stats
from data_builder.utils.split import stratified_split, split_by_source_composition

LOG = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project root (parent of data_builder)
AUDIT_PATH = _PROJECT_ROOT / "data_builder/audit/seed_audit.csv"
PARAPHRASE_AUDIT_DIR = _PROJECT_ROOT / "data_builder/audit"
PARAPHRASE_SUBDIR = "_paraphrases"


def run_seed_pipeline(config_path: str | Path | None = None) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO)
    seeds = collect_seeds(config_path)
    LOG.info("Collected %d seeds before dedup", len(seeds))

    seeds = dedup.exact_dedup(seeds)
    seeds = dedup.near_duplicate_dedup(seeds)

    audit_seed_sample(seeds, AUDIT_PATH)
    LOG.info("Seed pool ready with %d rows", len(seeds))

    return seeds


def run_paraphrase_pipeline(
    seeds: pd.DataFrame,
    output_name: str,
    audit_name: str,
    split_name: str = "default",
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)

    models = config.get("paraphrase_models", [])
    per_seed = config.get("paraphrases_per_seed", 3)
    template = config.get("paraphrase_prompt_template", "{text}")
    stage = config.get("paraphrase_stage", {})
    cache_dir = stage.get("cache_dir")
    resume = stage.get("resume", True)
    delay_seconds = float(stage.get("delay_seconds", 0.5))

    if not models:
        raise RuntimeError("No paraphrase models configured in config.yaml")

    output_dir = Path(config.get("output_dir", "data/pids_bench_v1"))
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    paraphrase_dir = output_dir / PARAPHRASE_SUBDIR
    expected_rows = len(seeds) * per_seed
    output_path = paraphrase_dir / output_name

    if resume and output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            if "text" not in existing.columns or "label" not in existing.columns:
                paraphrases = None
                skip_write = False
            elif len(existing) >= expected_rows:
                paraphrases = existing.iloc[:expected_rows].copy()
                # Do not overwrite file when we took a subset (keeps larger file intact)
                skip_write = len(existing) > expected_rows
                LOG.info(
                    "Resuming: using first %d rows from %s (file has %d, skipping paraphrase API)%s",
                    expected_rows,
                    output_path,
                    len(existing),
                    " (not overwriting file)" if skip_write else "",
                )
            else:
                paraphrases = None
                skip_write = False
        except Exception:
            paraphrases = None
            skip_write = False
    else:
        paraphrases = None
        skip_write = False

    if paraphrases is None:
        _cache_dir = None
        if cache_dir:
            _cache_dir = Path(cache_dir)
            if not _cache_dir.is_absolute():
                _cache_dir = _PROJECT_ROOT / _cache_dir
        paraphrases = paraphrase_seeds(
            seeds,
            per_seed,
            models,
            template,
            cache_dir=_cache_dir,
            resume=resume,
            split_name=split_name,
            delay_seconds=delay_seconds,
        )
        skip_write = False

    PARAPHRASE_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = PARAPHRASE_AUDIT_DIR / audit_name
    audit_seed_sample(paraphrases, audit_path)

    LOG.info("Paraphrased %d rows", len(paraphrases))

    paraphrase_dir.mkdir(parents=True, exist_ok=True)
    if not skip_write:
        paraphrases.to_csv(output_path, index=False)

    return paraphrases


def apply_obfuscation(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    obf_cfg = config.get("obfuscation", {})
    ratio = obf_cfg.get("overall_target", 0.0)

    if ratio <= 0.0:
        df["obfuscation"] = df.get("obfuscation", "none")
        return df

    transform_mix = list(obf_cfg.get("transform_mix", {}).items())
    if not transform_mix:
        df["obfuscation"] = df.get("obfuscation", "none")
        return df

    names, weights = zip(*transform_mix)

    rng = random.Random(config.get("seed", 42))

    df = df.copy()
    df["obfuscation"] = df.get("obfuscation", "none")

    n_apply = int(len(df) * ratio)
    if n_apply <= 0:
        return df

    n_apply = min(n_apply, len(df))
    indices = rng.sample(range(len(df)), k=n_apply)

    for idx in indices:
        transform = rng.choices(names, weights=weights, k=1)[0]
        df.at[idx, "text"] = obfuscate.TRANSFORMS[transform](df.at[idx, "text"], rng)   #type: ignore
        df.at[idx, "obfuscation"] = transform

    return df


def trim_splits_to_target(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trim splits to target counts. Supports v3 absolute counts or v2 ratio-based."""
    splits = config.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})
    seed = config.get("seed", 42)

    # v3: absolute counts (e.g. train: 60000, val: 7500, standard_test: 7500)
    train_cfg = splits.get("train", 0.8)
    val_cfg = splits.get("val", 0.1)
    test_cfg = splits.get("standard_test", splits.get("test", 0.1))

    if isinstance(train_cfg, (int, float)) and train_cfg > 1 and isinstance(val_cfg, (int, float)):
        target_train = int(train_cfg)
        target_val = int(val_cfg)
        target_test = int(test_cfg) if isinstance(test_cfg, (int, float)) else 0
    else:
        target_total = int(config.get("target_total", config.get("target_total_main_corpus", 0)))
        if target_total <= 0:
            return train_df, val_df, test_df
        target_train = int(target_total * float(train_cfg))
        target_val = int(target_total * float(val_cfg))
        target_test = target_total - target_train - target_val

    if len(train_df) > target_train:
        train_df = train_df.sample(n=target_train, random_state=seed).reset_index(drop=True)
    if len(val_df) > target_val:
        val_df = val_df.sample(n=target_val, random_state=seed).reset_index(drop=True)
    if len(test_df) > target_test:
        test_df = test_df.sample(n=target_test, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def log_obfuscation_counts(df: pd.DataFrame) -> None:
    overall, by_label, by_type = stats.obfuscation_table(df.to_dict("records"))

    LOG.info("Obfuscation distribution overall: %s", overall)
    LOG.info("Obfuscation by label: %s", dict(by_label))
    LOG.info("Obfuscation by attack_type: %s", dict(by_type))


def run(config_path: str | Path | None = None) -> None:
    config = load_config(config_path)

    output_dir = Path(config.get("output_dir", "data/pids_bench_v1"))
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = run_seed_pipeline(config_path)

    splits = config.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})
    seed = config.get("seed", 42)
    source_comp = config.get("source_composition", {})
    train_cfg = splits.get("train", 0.8)
    val_cfg = splits.get("val", 0.1)
    test_cfg = splits.get("standard_test", splits.get("test", 0.1))

    if isinstance(train_cfg, (int, float)) and train_cfg > 1:
        train_n, val_n, test_n = int(train_cfg), int(val_cfg), int(test_cfg)
        if source_comp and "train" in source_comp and "eval" in source_comp:
            # Split by source composition: train 55-60% real, eval 70% real
            seed_df = seeds
            real_records = seed_df[seed_df["source_type"] == "real"].to_dict("records")
            template_records = seed_df[seed_df["source_type"] == "template"].to_dict("records")
            train_seeds, val_seeds, test_seeds = split_by_source_composition(
                real_records,
                template_records,
                train_n,
                val_n,
                test_n,
                train_real_ratio=float(source_comp["train"].get("real", 0.57)),
                eval_real_ratio=float(source_comp["eval"].get("real", 0.70)),
                paraphrases_per_seed=config.get("paraphrases_per_seed", 3),
                seed=seed,
            )
            LOG.info(
                "Source composition: train real=%d template=%d, val real=%d template=%d, test real=%d template=%d",
                sum(1 for r in train_seeds if r.get("source_type") == "real"),
                sum(1 for r in train_seeds if r.get("source_type") == "template"),
                sum(1 for r in val_seeds if r.get("source_type") == "real"),
                sum(1 for r in val_seeds if r.get("source_type") == "template"),
                sum(1 for r in test_seeds if r.get("source_type") == "real"),
                sum(1 for r in test_seeds if r.get("source_type") == "template"),
            )
        else:
            total = train_n + val_n + test_n
            train_ratio = train_n / total
            val_ratio = val_n / total
            train_seeds, val_seeds, test_seeds = stratified_split(
                seeds.to_dict("records"), train_ratio, val_ratio, seed
            )
    else:
        train_seeds, val_seeds, test_seeds = stratified_split(
            seeds.to_dict("records"), float(train_cfg), float(val_cfg), seed
        )

    train_df = run_paraphrase_pipeline(
        pd.DataFrame(train_seeds),
        "train_paraphrases.csv",
        "train_paraphrase_sample.csv",
        split_name="train",
        config_path=config_path,
    )
    val_df = run_paraphrase_pipeline(
        pd.DataFrame(val_seeds),
        "val_paraphrases.csv",
        "val_paraphrase_sample.csv",
        split_name="val",
        config_path=config_path,
    )
    test_df = run_paraphrase_pipeline(
        pd.DataFrame(test_seeds),
        "test_paraphrases.csv",
        "test_paraphrase_sample.csv",
        split_name="test",
        config_path=config_path,
    )

    # Post-paraphrase dedup to remove repetitive variants
    if config.get("post_paraphrase_dedup", {}).get("enabled", True):
        before = len(train_df) + len(val_df) + len(test_df)
        train_df = dedup.exact_dedup(train_df)
        val_df = dedup.exact_dedup(val_df)
        test_df = dedup.exact_dedup(test_df)
        after = len(train_df) + len(val_df) + len(test_df)
        LOG.info("Post-paraphrase dedup: %d -> %d rows", before, after)

    train_df = apply_obfuscation(train_df, config)
    val_df = apply_obfuscation(val_df, config)
    test_df = apply_obfuscation(test_df, config)

    train_df, val_df, test_df = trim_splits_to_target(
        train_df,
        val_df,
        test_df,
        config,
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    log_obfuscation_counts(combined_df)

    LOG.info(
        "Final split counts (train/val/test): %d/%d/%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # Build eval subsets (balanced_subtype_test, obfuscated_attacks) when config has eval_splits
    eval_splits = config.get("eval_splits", {})
    if eval_splits:
        LOG.info("Building evaluation subsets...")
        run_build_eval_subsets(output_dir, eval_splits)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build PIDS-Bench dataset.")
    ap.add_argument(
        "--config",
        default=None,
        help="Config file (e.g. config_v3.yaml). Default: config.yaml",
    )
    args = ap.parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()