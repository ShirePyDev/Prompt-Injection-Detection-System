"""
Shared DeBERTa surrogate scoring for PIDS-Bench v3 adaptive pilot (max_len=512, class-1 prob).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 512


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float64)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tau(summary_path: Path) -> float:
    with open(summary_path) as f:
        data = json.load(f)
    if "val_tuned_threshold" not in data:
        raise KeyError(f"{summary_path} missing 'val_tuned_threshold'")
    return float(data["val_tuned_threshold"])


class DeBERTaSurrogate:
    """Loads once; reuse for many scoring calls (adaptive loop)."""

    def __init__(
        self,
        model_dir: Path,
        device: torch.device | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device or pick_device()
        self.batch_size = batch_size
        # Local export has tokenizer.json only (no spm.model); fast tokenizer is required.
        # tokenizer_config.json must not use legacy list-shaped extra_special_tokens (transformers 4.57+).
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score_texts(self, texts: list[str]) -> np.ndarray:
        all_probs: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tok(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs = _softmax(logits.detach().float().cpu().numpy())[:, 1]
            all_probs.extend(probs.tolist())
        return np.asarray(all_probs, dtype=np.float64)


@torch.inference_mode()
def score_injections(
    texts: list[str],
    model_dir: Path,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """One-shot scoring (e.g. seed selection) — loads model then discards."""
    s = DeBERTaSurrogate(model_dir, device, batch_size)
    return s.score_texts(texts)
