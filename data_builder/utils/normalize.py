"""Normalization helpers for dataset text."""
from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Minimal normalization: lowercase, collapse whitespace, strip."""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\"", "'")
    return text
