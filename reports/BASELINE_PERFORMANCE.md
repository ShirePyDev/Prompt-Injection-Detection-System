# PIDS-Bench v3 — In-repo baseline performance

This document summarizes how each **in-repository** detector performs on the **frozen** benchmark (`data/pids_bench_v3/`, checksums in `data/pids_bench_v3/FREEZE_MANIFEST.txt`). Numbers are taken directly from `outputs/*/summary.json` (synchronized with this file).

**Protocol:** Probabilistic models (TF-IDF+LR, DistilBERT, DeBERTa-v3) tune a **single threshold on `val.csv` only** (F1-maximising grid), then apply it to test and all stress sets. The rule-based detector is **binary** (no threshold tuning).

**Obfuscation set:** `eval_subsets/obfuscated_attacks.csv` contains **405** injection rows (rebuilt 2026-03-20; see manifest note on deduplication vs. the deprecated larger file).

---

## 1. Rule-based (regex + heuristics)

**Role:** Lightweight guardrail; not trained on the benchmark.

| Axis | Performance |
|------|-------------|
| **In-distribution test** | Weak overall (F1 **0.067**): precision **1.0** when it fires, recall **≈ 3.5%**. |
| **Over-defense (hard_benign_test)** | FPR **≈ 22.9%** (**337 / 1,472** flagged). |
| **Obfuscation (405 injections)** | Recall **≈ 2.5%** — rules do not track encoded/perturbed attacks. |
| **Domain OOD** | F1 **≈ 0.133**. |
| **Structural OOD** | F1 **≈ 0.338** (benign FPR **≈ 32.2%** per `summary.json`). |
| **Balanced subtype** | F1 **≈ 0.067**. |

**Takeaway:** Useful as a **floor**; pattern lists alone do not match modern injection diversity.

---

## 2. TF-IDF + Logistic Regression

**Role:** Strong non-neural baseline; bag-of-words + linear classifier. **Val-tuned τ = 0.518.**

| Axis | Performance |
|------|-------------|
| **In-distribution test** | F1 **≈ 0.967** @ val-tuned threshold. |
| **Over-defense** | FPR **≈ 39.0%** tuned (**574 / 1,472** flagged). |
| **Obfuscation (405)** | Recall **≈ 96.5%**. |
| **Domain OOD** | F1 **≈ 0.951**. |
| **Structural OOD** | F1 **≈ 0.689**; **FPR ≈ 79.2%** on benigns — heavy false alarming on structural benigns. |
| **Balanced subtype** | F1 **≈ 0.967**. |

---

## 3. DistilBERT-base-uncased (fine-tuned)

**Role:** Smaller transformer. **Val-tuned τ = 0.45.**

| Axis | Performance |
|------|-------------|
| **In-distribution test** | F1 **≈ 0.982** @ tuned threshold. |
| **Over-defense** | FPR **≈ 44.8%** tuned (**660 / 1,472**) — highest among these four on this axis in current summaries. |
| **Obfuscation (405)** | Recall **≈ 98.5%**. |
| **Domain OOD** | F1 **≈ 0.906**. |
| **Structural OOD** | F1 **≈ 0.612**; **FPR ≈ 73.2%** on benigns. |
| **Balanced subtype** | F1 **≈ 0.979**. |

---

## 4. DeBERTa-v3-base (fine-tuned)

**Role:** Larger transformer reference. **Val-tuned τ = 0.502** (`outputs/deberta_v3/summary.json` → `val_tuned_threshold`).

| Axis | Performance |
|------|-------------|
| **In-distribution test** | F1 **≈ 0.989** @ tuned threshold; precision **≈ 0.989**, recall **≈ 0.988**. |
| **Over-defense** | FPR **≈ 34.9%** tuned (**514 / 1,472**). |
| **Obfuscation (405)** | Recall **≈ 99.3%**. |
| **Domain OOD** | F1 **≈ 0.983**; FPR **≈ 1.0%**. |
| **Structural OOD** | F1 **≈ 0.703**; **FPR ≈ 82.5%** on benigns — severe over-flagging on benign structural-OOD prompts despite high attack recall. |
| **Balanced subtype** | F1 **≈ 0.988**. |

**Takeaway:** Strong on IID, obfuscation (this set), and domain OOD under this benchmark; **structural OOD** remains a **failure mode** for benign FPR, not a “success” axis.

---

## 5. Side-by-side tables (consolidated)

### Table A — In-distribution test (`test.csv`, val-tuned threshold)

| Model | F1 | Precision | Recall | FPR | FNR |
|-------|-----|-----------|--------|-----|-----|
| Rule-based | 0.0672 | 1.0000 | 0.0348 | 0.0000 | 0.9652 |
| TF-IDF+LR | 0.9672 | 0.9587 | 0.9758 | 0.0471 | 0.0242 |
| DistilBERT | 0.9820 | 0.9780 | 0.9860 | 0.0249 | 0.0140 |
| DeBERTa-v3 | 0.9889 | 0.9894 | 0.9884 | 0.0119 | 0.0116 |

### Table B — Hard benign over-defense (`hard_benign_test.csv`, 1,472 benign)

| Model | FPR @ tuned | # flagged (tuned) | τ |
|-------|-------------|---------------------|---|
| Rule-based | 0.2289 | 337 | — |
| TF-IDF+LR | 0.3899 | 574 | 0.518 |
| DistilBERT | 0.4484 | 660 | 0.45 |
| DeBERTa-v3 | 0.3492 | 514 | 0.502 |

### Table C — Obfuscation (`obfuscated_attacks.csv`, **405** injections)

| Model | Recall | FNR |
|-------|--------|-----|
| Rule-based | 0.0247 | 0.9753 |
| TF-IDF+LR | 0.9654 | 0.0346 |
| DistilBERT | 0.9852 | 0.0148 |
| DeBERTa-v3 | 0.9926 | 0.0074 |

### Table D — Domain OOD (`domain_ood.csv`, 2,000 rows)

| Model | F1 | FPR | FNR |
|-------|-----|-----|-----|
| Rule-based | 0.1326 | 0.0000 | 0.9290 |
| TF-IDF+LR | 0.9508 | 0.0670 | 0.0330 |
| DistilBERT | 0.9055 | 0.0420 | 0.1380 |
| DeBERTa-v3 | 0.9834 | 0.0100 | 0.0230 |

### Table E — Structural OOD (`structural_ood.csv`, 1,998 rows)

| Model | F1 | FPR | FNR |
|-------|-----|-----|-----|
| Rule-based | 0.3384 | 0.3223 | 0.7307 |
| TF-IDF+LR | 0.6891 | 0.7918 | 0.0581 |
| DistilBERT | 0.6121 | 0.7317 | 0.2362 |
| DeBERTa-v3 | 0.7034 | 0.8248 | 0.0100 |

### Table F — Balanced subtype (`balanced_subtype_test.csv`, 2,297 rows)

| Model | F1 | Recall | FPR |
|-------|-----|--------|-----|
| Rule-based | 0.0670 | 0.0346 | 0.0000 |
| TF-IDF+LR | 0.9668 | 0.9809 | 0.0442 |
| DistilBERT | 0.9788 | 0.9872 | 0.0275 |
| DeBERTa-v3 | 0.9882 | 0.9900 | 0.0125 |

---

## 6. Regenerating tables

```bash
cd Prompt-Injection-Detector-System
python scripts/consolidate_results.py
```

After any new training run, refresh the corresponding `outputs/<baseline>/summary.json` and update this file if prose must stay in sync.

---

## 7. Source files

| Baseline | Summary path |
|----------|--------------|
| Rule-based | `outputs/rule_baseline/summary.json` |
| TF-IDF+LR | `outputs/tfidf_baseline/summary.json` |
| DistilBERT | `outputs/distilbert_baseline/summary.json` |
| DeBERTa-v3 | `outputs/deberta_v3/summary.json` |
