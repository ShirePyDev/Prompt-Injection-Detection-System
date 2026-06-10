from __future__ import annotations
import random
from collections import defaultdict


def stratified_split(rows: list[dict], train_p: float, val_p: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        k = (r["label"], r.get("attack_type", "none"), r.get("obfuscation", "none"))
        buckets[k].append(r)

    train, val, test = [], [], []
    for k, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_p)
        n_val = int(n * val_p)
        train += items[:n_train]
        val += items[n_train : n_train + n_val]
        test += items[n_train + n_val :]
    return train, val, test


def stratified_split_by_counts(
    rows: list[dict],
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split by absolute counts, stratified by label + attack_type + obfuscation."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        k = (r["label"], r.get("attack_type", "none"), r.get("obfuscation", "none"))
        buckets[k].append(r)

    total = len(rows)
    requested = train_n + val_n + test_n
    train, val, test = [], [], []
    for k, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        if total <= 0:
            continue
        # Distribute proportionally to requested counts (handles total < requested)
        n_train = max(0, min(n, int(round(n * train_n / requested))))
        n_val = max(0, min(n - n_train, int(round(n * val_n / requested))))
        n_test = n - n_train - n_val
        train += items[:n_train]
        val += items[n_train : n_train + n_val]
        test += items[n_train + n_val :]

    # Trim to exact counts (may lose some rows if total > train_n+val_n+test_n)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train[:train_n], val[:val_n], test[:test_n]


def split_by_source_composition(
    real_rows: list[dict],
    template_rows: list[dict],
    train_n: int,
    val_n: int,
    test_n: int,
    train_real_ratio: float,
    eval_real_ratio: float,
    paraphrases_per_seed: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split real and template seeds separately to hit target source composition.
    - Train: train_real_ratio real (e.g. 0.57)
    - Val/test: eval_real_ratio real (e.g. 0.70)
    Returns (train_seeds, val_seeds, test_seeds).
    """
    rng = random.Random(seed)

    # Target rows per split (after paraphrase). Seeds = rows / paraphrases_per_seed.
    train_real_rows = int(train_n * train_real_ratio)
    train_template_rows = train_n - train_real_rows
    val_real_rows = int(val_n * eval_real_ratio)
    val_template_rows = val_n - val_real_rows
    test_real_rows = int(test_n * eval_real_ratio)
    test_template_rows = test_n - test_real_rows

    train_real_seeds = max(0, train_real_rows // paraphrases_per_seed)
    train_template_seeds = max(0, train_template_rows // paraphrases_per_seed)
    val_real_seeds = max(0, val_real_rows // paraphrases_per_seed)
    val_template_seeds = max(0, val_template_rows // paraphrases_per_seed)
    test_real_seeds = max(0, test_real_rows // paraphrases_per_seed)
    test_template_seeds = max(0, test_template_rows // paraphrases_per_seed)

    real_train, real_val, real_test = stratified_split_by_counts(
        real_rows, train_real_seeds, val_real_seeds, test_real_seeds, seed
    )
    template_train, template_val, template_test = stratified_split_by_counts(
        template_rows, train_template_seeds, val_template_seeds, test_template_seeds, seed
    )

    train_seeds = real_train + template_train
    val_seeds = real_val + template_val
    test_seeds = real_test + template_test
    rng.shuffle(train_seeds)
    rng.shuffle(val_seeds)
    rng.shuffle(test_seeds)
    return train_seeds, val_seeds, test_seeds