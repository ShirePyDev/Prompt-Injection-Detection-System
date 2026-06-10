"""Deduplication helpers for seeds and paraphrases."""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from data_builder.utils.normalize import normalize_text

LOG = logging.getLogger(__name__)


def exact_dedup(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df["_normalized"] = df[text_col].apply(normalize_text)
    before = len(df)
    df = df.drop_duplicates(subset="_normalized").reset_index(drop=True)
    after = len(df)
    LOG.info("Exact dedup removed %d rows", before - after)
    return df.drop(columns=["_normalized"])


def near_duplicate_dedup(
    df: pd.DataFrame,
    text_col: str = "text",
    threshold: float = 0.95,
    batch_size: int = 512,
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if len(df) < 2:
        return df

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    X = vectorizer.fit_transform(df[text_col].fillna(""))

    keep = []
    removed = set()
    for i in range(X.shape[0]):
        if i in removed:
            continue
        window = X[:i]
        if window.shape[0] == 0:
            keep.append(i)
            continue
        sims = X[i].dot(window.T).toarray().ravel()
        if sims.size and np.max(sims) > threshold:
            removed.add(i)
            continue
        keep.append(i)
    LOG.info("Near-duplicate dedup removed %d rows", len(df) - len(keep))
    return df.iloc[keep].reset_index(drop=True)
