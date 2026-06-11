# PIDS-Bench: A Multi-Axis Evaluation Framework for Prompt Injection Detectors

> **Paper:** *PIDS-Bench: A Multi-Axis Evaluation Framework for Characterizing Failure Modes of Prompt Injection Detectors Under Over-Defense, Obfuscation, and Distribution Shift*
> *(under review, IEEE Access)*

This repository contains the frozen benchmark, the evaluation code, and the dataset-construction pipeline behind the paper. Everything here is intended to make the results reproducible from a clean checkout.

---

## 📖 Overview

Prompt-injection detectors are increasingly deployed as gateway classifiers in front of LLM systems, yet most published evaluations summarize a detector with a single F1 score on a single in-distribution test set. That practice hides the errors that matter once a detector is deployed — in particular, how often it flags *benign* inputs as attacks.

PIDS-Bench evaluates a detector along multiple axes at once: in-distribution accuracy, benign false-positive behaviour on inputs that superficially resemble attacks, robustness to obfuscated attacks, and behaviour under domain and structural distribution shift. We evaluate seven detectors spanning classical, fine-tuned, external, and broad-safety designs, with a rule-based heuristic reported separately as a lower-bound reference.

The central finding is that a detector's in-distribution F1 does not predict its benign-side safety. The strongest internal detector reaches an in-distribution F1 of 0.9882 while still flagging roughly **38%** of an adversarially constructed benign set, and we show this over-defense is not removed by threshold calibration. Decomposing the benign errors by the *provenance* of the benign inputs reveals an asymmetry we call **provenance-sensitive over-defense**: curated hard-negative augmentation nearly eliminates over-defense on curated benign inputs but leaves it largely intact on externally-sourced ones.

---

## 🎯  What the benchmark measures

A detector with an in-distribution F1 of 0.9882 sounds deployable. PIDS-Bench shows that the same model flags **38.25%** of an adversarially constructed benign set as attacks at a fixed threshold, and that this benign false-positive rate rises further on a curated subset (**48.13%**). A single F1 score reveals neither.

The benchmark targets three failure modes that single-metric evaluation systematically hides:

| Failure mode | What it captures | Evaluation set |
|---|---|---|
| **Over-defense** | The detector flags benign prompts that pattern-match attacks | `hard_benign_test` — 1,472 benign inputs that mimic injection structure |
| **Obfuscation blindness** | Detection degrades when attacks are encoded or surface-perturbed | `obfuscated_attacks` — 405 injections across six transform types |
| **Distribution shift** | Performance changes outside the training domain or surface form | `domain_ood` (specialist domains) and `structural_ood` (structural transforms) |

A fourth axis, `balanced_subtype_test`, is a per-subtype diagnostic disaggregation of the in-distribution split rather than a separate stress condition; it is used to check per-attack-type fairness, not to define a failure mode.

---

## 📊 Benchmark Specification — PIDS-Bench v3

The benchmark is frozen under SHA-256 checksums recorded in:

```text
data/pids_bench_v3/FREEZE_MANIFEST.txt
```

(frozen 2026-03-24).

The manifest is the authoritative source of truth for row counts and hashes. Evaluation sets are static and must not be modified.

### Dataset Splits

| Split                   |       Rows | Benign | Injection | Purpose                       |
| ----------------------- | ---------: | -----: | --------: | ----------------------------- |
| `train`                 |     27,093 | 12,846 |    14,247 | Model training                |
| `val`                   |      3,989 |  1,943 |     2,046 | Threshold tuning              |
| `test`                  |      3,918 |  1,848 |     2,070 | In-distribution evaluation    |
| `hard_benign_test`      |      1,472 |  1,472 |         0 | Over-defense stress test      |
| `balanced_subtype_test` |      2,297 |  1,200 |     1,097 | Per-subtype diagnostics       |
| `obfuscated_attacks`    |        405 |      0 |       405 | Obfuscation robustness        |
| `domain_ood`            |      2,000 |  1,000 |     1,000 | Domain distribution shift     |
| `structural_ood`        |      1,998 |    999 |       999 | Structural distribution shift |
| **Total**               | **43,172** |        |           |                               |

---

## 🏗 Dataset construction

The dataset is built by the pipeline in `data_builder/`:

- **Seed collection** draws benign and injection examples from several public sources.
- **Deduplication** applies exact-match normalization followed by TF-IDF character n-gram near-duplicate filtering (cosine threshold 0.95) across the full seed pool **before** the split is applied.
- **Seed-family splitting** assigns every paraphrase and obfuscated descendant of a seed to the same partition before train/validation/test are separated, so structurally similar examples cannot straddle training and evaluation.
- **Paraphrase expansion** generates surface variants for the two scarce attack subtypes (`encoded_attack`, `tool_injection`) using GPT-4o-mini as a paraphrase generator at construction time. This is data-construction tooling only; GPT-4o-mini is **not** an evaluated detector.
- **Obfuscation augmentation** applies six transform types to injection seeds: Base64 encoding, ROT13, whitespace perturbation, homoglyph substitution, prompt dilution, and instruction-prefix mimicry.
- **Hard-negative augmentation** (optional, `data_builder/build_hard_negative.py`) adds curated benign examples that pattern-match injection surface features, cross-deduplicated against all training and evaluation files.

---

## 📈 Results

All learned models follow the same protocol: train on `train.csv`, tune a single classification threshold on `val.csv` (F1-maximizing grid search), and apply that threshold once to the test split and every stress set. Threshold selection never sees test data. The two fine-tuned models are reported as **five-seed mean ± standard deviation** over seeds {13, 42, 123, 2024, 7777}. TF-IDF + LR converges to identical solutions across seeds and is reported without variance.

### Internal baselines

Five-seed mean ± standard deviation at fixed threshold τ = 0.5. Hard-benign FPR is the aggregate over the full set (n = 1,472). ↑ higher is better; ↓ lower is better.

| Model | IID F1 ↑ | Real-source F1 ↑ | IID Ben-FPR ↓ | Hard-ben. FPR ↓ | Obf. Recall ↑ | Domain-shift F1 ↑ | Struct.-shift FPR ↓ | ROC-AUC ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR | 0.9656 | 0.9467 | 0.0514 | 0.4090 | 0.9654 | 0.9508 | 0.7918 | 0.9941 |
| DistilBERT | 0.9831 ± 0.0017 | 0.9738 ± 0.0026 | 0.0249 ± 0.0013 | 0.4702 ± 0.0677 | 0.9877 ± 0.0058 | 0.8886 ± 0.0214 | 0.7826 ± 0.0963 | 0.9974 ± 0.0002 |
| DeBERTa-v3-FT | 0.9882 ± 0.0008 | 0.9825 ± 0.0009 | 0.0124 ± 0.0013 | 0.3825 ± 0.0474 | 0.9867 ± 0.0079 | 0.9604 ± 0.0099 | 0.8108 ± 0.0617 | 0.9983 ± 0.0004 |

### Hard-benign over-defense by source origin

False-positive rate on the hard-benign set at τ = 0.5, decomposed by the provenance of the benign input. This decomposition is the basis for the provenance-sensitive over-defense finding.

| Model | Aggregate ↓ | Externally-sourced (n = 872) ↓ | Curated (n = 600) ↓ |
|---|---:|---:|---:|
| TF-IDF + LR | 0.4090 | 0.4622 | 0.3317 |
| DistilBERT | 0.4702 ± 0.0677 | 0.4606 ± 0.0639 | 0.4843 ± 0.0793 |
| DeBERTa-v3-FT | 0.3825 ± 0.0474 | 0.3144 ± 0.0525 | 0.4813 ± 0.0855 |

> **Note on direction.** DeBERTa-v3-FT shows the strongest provenance dependence (0.3144 externally-sourced vs. 0.4813 curated). DistilBERT over-blocks both subsets at comparable rates, and TF-IDF over-blocks the externally-sourced subset *more* than the curated one — so the curated-vs-externally-sourced asymmetry is specific to the fine-tuned detectors, not universal across all models.

### External and broad-safety detectors

<!--
  TODO (fill from the submitted paper's Table 5 — external/broad-safety detectors):
  ProtectAI DeBERTa-v3, DeBERTa-PI (deepset), PromptGuard 2 (86M), Llama Guard 3-1B.
  Paste those rows and they go here, at their reported thresholds (native and swept).
  Left blank deliberately rather than reproduced from an older draft, to avoid
  publishing unverified numbers next to the paper.
-->

*Results for the external prompt-injection detectors (ProtectAI DeBERTa-v3, DeBERTa-PI) and broad-safety comparators (PromptGuard 2 (86M), Llama Guard 3-1B) are reported in the paper's external-detector table. External detectors are evaluated at their native decision boundaries and, where a continuous score is available, at a swept threshold; broad-safety classifiers do not expose a continuous score and are excluded from ROC-AUC comparison.*

### Rule-based lower-bound reference

The rule-based heuristic is reported separately as a lower-bound reference and is outside the seven-detector comparison. It achieves high precision but very low recall (F1 ≈ 0.07), which confirms that pattern matching alone cannot handle modern injection diversity.

---

## 🔍 Key Findings

**In-distribution F1 does not predict benign-side safety.** DeBERTa-v3-FT posts the highest in-distribution F1 (0.9882) yet still flags 38.25% of the hard-benign set at τ = 0.5, rising to 48.13% on the curated subset. Across a full threshold sweep and five seeds, no internal detector reaches an operating point that simultaneously satisfies F1 ≥ 0.95 and externally-sourced hard-benign FPR ≤ 0.10 on this security-adjacent stress distribution.

**Over-defense is not a threshold artifact.** The benign inputs that are over-blocked are scored deep in the injection region, not marginally above the decision boundary, so raising the threshold does not separate them from true positives without destroying recall.

**Over-defense is provenance-sensitive.** Curated hard-negative augmentation nearly eliminates over-defense on the curated subset (DeBERTa-v3-FT 0.4813 → 0.0010; DistilBERT 0.4843 → 0.0123) but leaves over-defense on the externally-sourced subset largely intact (DeBERTa-v3-FT 0.3144 → 0.3101; DistilBERT 0.4606 → 0.3899), with no measurable in-distribution cost. The augmentation was matched to the curated subset; whether augmentation matched to the externally-sourced distribution would close the gap is untested.

**Obfuscation is largely handled by fine-tuned models.** TF-IDF + LR, DistilBERT, and DeBERTa-v3-FT all exceed 96% recall on the obfuscated set, while the rule-based heuristic collapses. One model-specific vulnerability stands out: TF-IDF + LR drops to zero recall on homoglyph attacks, whereas the transformer models remain robust.

**Distribution shift exposes model-specific weaknesses.** Under domain shift, DeBERTa-v3-FT retains strong F1 (0.9604) while DistilBERT degrades (0.8886), with the code domain specifically reducing DistilBERT recall to 0.435 — a within-class content shift, not generalization to an unseen attack distribution. Under structural shift, all fine-tuned models experience severe benign false-positive pressure, with DeBERTa-v3-FT's structural FPR averaging 0.8108 across transforms and `instruction_prefix_mimic` the worst single transform.

---

## 📂 Repository structure

```
data/pids_bench_v3/               Frozen benchmark dataset
  train.csv                       27,093 rows — model training
  val.csv                         3,989 rows — threshold tuning only
  test.csv                        3,918 rows — in-distribution evaluation
  eval_subsets/
    hard_benign_test.csv          1,472 benign inputs (over-defense; has source_type)
    balanced_subtype_test.csv     2,297 rows balanced across attack subtypes
    obfuscated_attacks.csv        405 obfuscated injections (six transform types)
  ood/
    domain_ood.csv                2,000 rows across specialist domains
    structural_ood.csv            1,998 rows across structural transforms
  FREEZE_MANIFEST.txt             SHA-256 checksums — authoritative source of truth

src/baselines/
  rule_based.py                   Regex + heuristic detector (no training)
  tfidf_logreg.py                 TF-IDF + Logistic Regression
  distilbert_baseline.py          DistilBERT fine-tuned on PIDS-Bench v3
  deberta_v3.py                   DeBERTa-v3 fine-tuned on PIDS-Bench v3
  deberta_v3_hardneg.py           DeBERTa-v3 with hard-negative augmentation

src/external_eval/                ProtectAI and related external detector evaluation
src/external_eval_security/       PromptGuard 2, LLM-Guard, DeBERTa-PI evaluation

data_builder/
  build_dataset.py                Main dataset construction pipeline
  build_domain_ood.py             Domain OOD set construction
  build_structural_ood.py         Structural OOD set construction
  build_hard_benign.py            Hard-benign stress set construction
  build_hard_negative.py          Optional hard-negative augmentation pipeline
  config_v3.yaml                  Dataset construction configuration
  utils/                          Deduplication, normalization, split utilities
  generators/                     Seed templates and paraphrase generation
  transforms/                     Obfuscation transform implementations

eval/                             Threshold analysis, ROC/PR plots, breakdowns
scripts/                          Result aggregation, multi-seed runs, external eval
reports/                          Per-axis result write-ups

outputs/                          Experiment results (git-ignored; generated locally)
models/                           Trained checkpoints (git-ignored; generated locally)
```

---

## ⚙️ Setup

**Requirements:** Python 3.10 or later. A CUDA GPU or Apple Silicon (MPS) is recommended for the transformer baselines; the rule-based and TF-IDF baselines run on CPU.

```bash
git clone https://github.com/ShirePyDev/Prompt-Injection-Detection-System.git
cd Prompt-Injection-Detection-System
pip install -r requirements.txt
```

For Colab or Kaggle, use `requirements_colab.txt`. The training notebooks (`colab_deberta_training.ipynb`, `kaggle_deberta_training.ipynb`) are self-contained.

If you plan to rebuild the OOD sets or run anything that calls a hosted API or a gated HuggingFace model, create a `.env` file in the project root with your own credentials:

```
OPENAI_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

The `.env` file is git-ignored and must never be committed.

---

## 🚀 Running the baselines

Each learned baseline trains on `train.csv`, tunes its threshold on `val.csv`, and applies it once to all evaluation sets.

```bash
# Rule-based reference (no training)
python -m src.baselines.rule_based

# TF-IDF + Logistic Regression
python -m src.baselines.tfidf_logreg

# DistilBERT
python -m src.baselines.distilbert_baseline

# DeBERTa-v3
python -m src.baselines.deberta_v3
```

To include hard-negative augmentation during fine-tuning:

```bash
python -m src.baselines.deberta_v3_hardneg
```

The paper's headline numbers are five-seed aggregates. To reproduce the multi-seed runs, use the scripts under `scripts/` (for example, the DeBERTa and DistilBERT multi-seed runners) with seeds {13, 42, 123, 2024, 7777}, then aggregate.

---

## 📋 Evaluation

```bash
# Consolidate per-model summaries into the main comparison tables
python scripts/consolidate_results.py

# Per-domain and per-transform breakdowns
python scripts/breakdown_analysis.py

# External and broad-safety detectors (needs HF token; OpenAI key only if used)
python scripts/run_external_multiaxis.py
```

Threshold tuning and ROC/PR analysis live in `eval/` (for example, `eval/threshold_analysis.py` and the plotting scripts).

---

## 🔄 Reproducibility

- The fine-tuned results are five-seed means over seeds {13, 42, 123, 2024, 7777}, reported with standard deviations.
- All evaluation sets are frozen under SHA-256 checksums in `data/pids_bench_v3/FREEZE_MANIFEST.txt`; the obfuscation rebuild and its zero-overlap checks are documented there.
- Threshold tuning operates exclusively on `val.csv` and is applied once to every evaluation set without re-tuning.
- Hyperparameters are specified in each baseline script and the construction config; the pinned software environment is recorded in `requirements.txt`.
- The deduplication, seed-family split, and paraphrase-generation procedures are implemented in `data_builder/utils/` and `data_builder/build_dataset.py`.

A note on scope: the `domain_ood` injection side is drawn from held-out injection data that shares provenance with the training injection class, so its attack-side numbers measure content-level transfer within the injection class rather than generalization to an unseen attack distribution. The hard-benign audit was performed by a single annotator. Both points are stated in the paper's limitations.
---

## 📝 Citation

If you use PIDS-Bench in your work, please cite:

```bibtex
@article{pids_bench_2026,
  title   = {PIDS-Bench: A Multi-Axis Evaluation Framework for Characterizing
             Failure Modes of Prompt Injection Detectors Under Over-Defense,
             Obfuscation, and Distribution Shift},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

---

## 📄 License

Released under the MIT License (see `LICENSE`). The benchmark dataset in `data/pids_bench_v3/` is released for research use; source datasets incorporated into the benchmark retain their original licenses, with provenance documented under `data_builder/sources/`.
