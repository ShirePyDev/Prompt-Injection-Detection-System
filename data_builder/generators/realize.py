"""Paraphrase expansion for seeds."""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
from openai import OpenAI
from openai import BadRequestError

LOG = logging.getLogger(__name__)
_client: OpenAI | None = None


def configure_openai() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run paraphrasing")
    _client = OpenAI(api_key=api_key)
    return _client


def _safe_filename(seed_id: str) -> str:
    """Sanitize seed_id for use in cache filename."""
    return re.sub(r"[^\w\-.]", "_", str(seed_id))


def _sanitize_text(text: str) -> str:
    """Sanitize text for safe JSON/API payload. Removes chars that break OpenAI requests."""
    if not isinstance(text, str):
        text = str(text)
    # Fix encoding
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    # Remove null bytes and other control chars
    text = text.replace("\x00", "")
    text = "".join(c for c in text if c in "\n\t\r" or (ord(c) >= 32 and ord(c) != 0x7F))
    # Normalize unpaired surrogates (can break JSON)
    text = unicodedata.normalize("NFKC", text)
    # Ensure result is JSON-serializable
    try:
        json.dumps(text)
    except (TypeError, ValueError):
        text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    return text[:8000]  # Cap length to avoid API limits


def paraphrase_prompt(model: str, prompt: str, max_tokens: int = 512) -> str:
    client = configure_openai()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()  # type: ignore


def make_prompt(template: str, text: str) -> str:
    return template.format(text=text)


def paraphrase_seeds(
    seeds: pd.DataFrame,
    paraphrases_per_seed: int,
    models: Sequence[str],
    template: str,
    *,
    cache_dir: str | Path | None = None,
    resume: bool = True,
    split_name: str = "default",
    delay_seconds: float = 0.5,
    **kwargs,
) -> pd.DataFrame:
    """Paraphrase each seed. Writes per-seed cache for resume after rate limits."""
    configure_openai()
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total_seeds = len(seeds)
    total_calls = total_seeds * paraphrases_per_seed
    done_calls = 0

    for idx, (_, seed) in enumerate(seeds.iterrows()):
        seed_id = seed["seed_id"]
        cache_file = cache_path / f"{split_name}_{_safe_filename(seed_id)}.json" if cache_path else None

        # Resume: load from cache if complete
        if resume and cache_file and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(cached, list) and len(cached) == paraphrases_per_seed:
                    rows.extend(cached)
                    done_calls += paraphrases_per_seed
                    if (idx + 1) % 500 == 0 or idx == total_seeds - 1:
                        LOG.info("Paraphrase %s: %d/%d seeds (%d/%d calls)", split_name, idx + 1, total_seeds, done_calls, total_calls)
                    continue
            except Exception:
                pass

        # Generate paraphrases (with fallback on API error)
        seed_rows: list[dict] = []
        raw_text = _sanitize_text(seed["text"])
        used_fallback = False

        for index in range(paraphrases_per_seed):
            model = models[index % len(models)]
            try:
                prompt = make_prompt(template, raw_text)
                text = paraphrase_prompt(model, prompt)
            except BadRequestError as e:
                if not used_fallback:
                    LOG.warning(
                        "Paraphrase API error for seed %s (idx %d): %s. Using original text as fallback.",
                        seed_id, idx, str(e)[:80],
                    )
                    used_fallback = True
                text = raw_text

            seed_rows.append({
                "parent_seed_id": seed_id,
                "text": text,
                "label": int(seed["label"]),
                "source": str(seed["source"]),
                "attack_type": str(seed["attack_type"]),
                "language": str(seed["language"]),
                "generator_model": model,
                "paraphrase_round": index + 1,
                "source_type": str(seed.get("source_type", "template")),
            })
            done_calls += 1
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        rows.extend(seed_rows)

        # Write cache immediately so resume works after rate limit
        if cache_file:
            tmp = cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(seed_rows, ensure_ascii=False, indent=0), encoding="utf-8")
            tmp.replace(cache_file)

        if (idx + 1) % 500 == 0 or idx == total_seeds - 1:
            LOG.info("Paraphrase %s: %d/%d seeds (%d/%d calls)", split_name, idx + 1, total_seeds, done_calls, total_calls)

    return pd.DataFrame(rows)
