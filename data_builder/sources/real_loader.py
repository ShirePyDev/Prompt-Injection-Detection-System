from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_OUTPUT_COLUMNS = [
    "id",
    "seed_id",
    "parent_seed_id",
    "text",
    "label",
    "source",
    "source_type",
    "source_name",
    "attack_type",
    "obfuscation",
    "language",
]


def _validate_source_entry(src: dict[str, Any]) -> None:
    required = ["name", "path", "quota"]
    missing = [k for k in required if k not in src]
    if missing:
        raise ValueError(f"Missing required real_sources fields: {missing} in {src}")


def _safe_get(row: pd.Series, key: str, default: Any = None) -> Any:
    return row[key] if key in row and pd.notna(row[key]) else default


def _normalize_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    return str(text).strip()


def _normalize_language(value: Any, fallback: str = "en") -> str:
    if value is None or pd.isna(value):
        return fallback
    v = str(value).strip().lower()

    # small normalization
    mapping = {
        "english": "en",
        "eng": "en",
        "deutsch": "de",
        "german": "de",
        "korean": "ko",
        "kr": "ko",
        "kor": "ko",
    }
    return mapping.get(v, v)


def _normalize_attack_type(value: Any, fallback: str = "none") -> str:
    if value is None or pd.isna(value):
        return fallback
    return str(value).strip().lower()


def _make_ids(prefix: str, idx: int) -> tuple[str, str, str]:
    seed_id = f"{prefix}_seed_{idx:07d}"
    row_id = f"{prefix}_row_{idx:07d}"
    parent_seed_id = seed_id
    return row_id, seed_id, parent_seed_id


def _prepare_rows_from_df(
    df: pd.DataFrame,
    *,
    source_name: str,
    source_type: str,
    quota: int,
    default_label: int | None = None,
    default_attack_type: str = "none",
    default_language: str = "en",
    text_column: str = "text",
    label_column: str | None = "label",
    attack_type_column: str | None = "attack_type",
    language_column: str | None = "language",
) -> list[dict[str, Any]]:
    if text_column not in df.columns:
        raise ValueError(
            f"Expected text column '{text_column}' in source '{source_name}', "
            f"but columns are: {df.columns.tolist()}"
        )

    rows: list[dict[str, Any]] = []
    df = df.copy()

    # remove empty texts first
    df[text_column] = df[text_column].apply(_normalize_text)
    df = df[df[text_column] != ""]

    if quota > 0:
        df = df.head(quota)

    for i, (_, row) in enumerate(df.iterrows()):
        text = _normalize_text(row[text_column])
        if not text:
            continue

        # label
        if label_column is not None and label_column in df.columns:
            label = int(_safe_get(row, label_column, default_label if default_label is not None else 0))
        elif default_label is not None:
            label = int(default_label)
        else:
            raise ValueError(
                f"Source '{source_name}' has no label column '{label_column}' "
                f"and no default_label was provided."
            )

        # attack type
        if attack_type_column is not None and attack_type_column in df.columns:
            attack_type = _normalize_attack_type(_safe_get(row, attack_type_column, default_attack_type))
        else:
            attack_type = default_attack_type

        # benign should always map to none
        if label == 0:
            attack_type = "none"

        # language
        if language_column is not None and language_column in df.columns:
            language = _normalize_language(_safe_get(row, language_column, default_language), default_language)
        else:
            language = default_language

        row_id, seed_id, parent_seed_id = _make_ids(source_name, i)

        rows.append(
            {
                "id": row_id,
                "seed_id": seed_id,
                "parent_seed_id": parent_seed_id,
                "text": text,
                "label": label,
                "source": source_name,
                "source_type": source_type,
                "source_name": source_name,
                "attack_type": attack_type,
                "obfuscation": "none",
                "language": language,
            }
        )

    return rows


def load_real_prompts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Load all real prompt sources defined in config.yaml.

    Expected config shape:

    real_sources:
      - name: qualifire
        path: data/real_prompts/qualifire.csv
        quota: 1500
        source_type: real
        default_label: 1
        default_attack_type: direct_override
        default_language: en
        text_column: text
        label_column: label
        attack_type_column: attack_type
        language_column: language
    """
    real_sources = cfg.get("real_sources", [])
    if not real_sources:
        return []

    all_rows: list[dict[str, Any]] = []

    for src in real_sources:
        _validate_source_entry(src)

        source_name = str(src["name"]).strip()
        source_type = str(src.get("source_type", "real")).strip()
        path = Path(src["path"])
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            path = project_root / path
        quota = int(src["quota"])

        if not path.exists():
            raise FileNotFoundError(f"Real source file not found: {path}")

        # column mapping / defaults
        text_column = str(src.get("text_column", "text"))
        label_column = src.get("label_column", "label")
        attack_type_column = src.get("attack_type_column", "attack_type")
        language_column = src.get("language_column", "language")

        default_label = src.get("default_label", None)
        default_attack_type = str(src.get("default_attack_type", "none"))
        default_language = str(src.get("default_language", "en"))

        # load
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".jsonl", ".json"}:
            df = pd.read_json(path, lines=(path.suffix.lower() == ".jsonl"))
        else:
            raise ValueError(f"Unsupported real source format: {path.suffix} for {path}")

        rows = _prepare_rows_from_df(
            df,
            source_name=source_name,
            source_type=source_type,
            quota=quota,
            default_label=default_label,
            default_attack_type=default_attack_type,
            default_language=default_language,
            text_column=text_column,
            label_column=label_column,
            attack_type_column=attack_type_column,
            language_column=language_column,
        )

        all_rows.extend(rows)

    return all_rows