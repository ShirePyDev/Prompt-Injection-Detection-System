# PIDS-Bench: A Multi-Axis Evaluation Framework for Prompt Injection Detectors

**Paper:** *PIDS-Bench: A Multi-Axis Evaluation Framework for Characterizing Failure Modes of Prompt Injection Detectors Under Over-Defense, Obfuscation, and Distribution Shift*

---

## Overview

Prompt injection detectors are widely deployed as safety guardrails for LLM-based systems, yet their evaluation remains largely one-dimensional. Most published evaluations report a single F1 score on a single in-distribution test set. This practice obscures three distinct failure modes that matter in deployment: a detector that aggressively flags benign inputs, a detector that fails when attacks are encoded or rephrased, and a detector that collapses under domain or surface-form shift.

This repository introduces **PIDS-Bench** (Prompt Injection Detection System Benchmark), a multi-axis evaluation framework that exposes these failure modes through five orthogonal evaluation axes. We build the benchmark, train and evaluate four in-repository baselines spanning the range from lightweight heuristics to large fine-tuned transformers, and extend the comparison to four external detectors. The central finding is that no model holds its performance profile across all five axes — high IID F1 is not predictive of benign-side safety or structural robustness.

---

## Motivation

Consider a detector with IID F1 of 0.989. Is it deployable? PIDS-Bench shows that same model flags **34.9%** of adversarially curated benign inputs as injections, and raises its benign false positive rate to **82.5%** under structural distribution shift. A single F1 score would not reveal either failure.

We identify three failure modes that single-metric evaluation systematically hides:

| Failure Mode | Definition | Evaluation Axis |
|---|---|---|
| **Over-defense** | Detector flags benign prompts that pattern-match attacks | FPR on `hard_benign_test` — 1,472 curated adversarial benign inputs |
| **Obfuscation blindness** | Detection degrades when attacks are encoded or surface-perturbed | Recall on `obfuscated_attacks` — 405 injection examples with six transform types |
| **Distribution shift** | Performance collapses outside the training domain or surface form | F1 and FPR on `domain_ood` (4 specialist domains) and `structural_ood` (3 structural transforms) |

---

## Benchmark — PIDS-Bench v3

The benchmark is frozen under SHA-256 checksums (see `data/pids_bench_v3/FREEZE_MANIFEST.txt`, frozen 2026-03-24). All evaluation sets are static and must not be modified. Authoritative row counts and hashes are in the manifest; the table below is for reference only.

### Dataset Splits

| Split | Rows | Benign | Injection | Purpose |
|---|---|---|---|---|
| `train` | 27,093 | 12,846 (47.4%) | 14,247 (52.6%) | Model training |
| `val` | 3,989 | 1,943 (48.7%) | 2,046 (51.3%) | Threshold tuning and early stopping |
| `test` | 3,918 | 1,848 (47.2%) | 2,070 (52.8%) | Standard in-distribution evaluation |
| `hard_benign_test` | 1,472 | 1,472 (100%) | 0 | Over-defense stress test |
| `balanced_subtype_test` | 2,297 | 1,200 | 1,097 | Attack-subtype fairness |
| `obfuscated_attacks` | 405 | 0 | 405 (100%) | Obfuscation robustness (no train overlap — see manifest) |
| `domain_ood` | 2,000 | 1,000 | 1,000 | Domain distribution shift (Medical, Legal, Finance, Code) |
| `structural_ood` | 1,998 | 999 | 999 | Structural distribution shift (3 surface transforms) |
| **Total (all manifest sets)** | **43,172** | | | |

`balanced_subtype_test` and `obfuscated_attacks` are derived diagnostic slices drawn exclusively from `test.csv`. The OOD stress sets are independently constructed and held completely separate from the main corpus.

### Dataset Construction

The dataset is built through a multi-stage pipeline implemented in `data_builder/`:

- **Seed collection** draws from seven public sources covering both real and template-generated benign and injection examples.
- **Deduplication** applies exact-match normalization followed by TF-IDF character n-gram near-duplicate filtering (cosine threshold 0.95) across the full seed pool before splitting.
- **Paraphrase expansion** generates three surface variants per seed using an LLM, after splitting, ensuring all paraphrases of a seed land in the same split and cannot leak across train and test.
- **Obfuscation augmentation** applies six transform types to injection seeds — Base64 encoding, ROT13, whitespace perturbation, homoglyph substitution, prompt dilution, and instruction-prefix mimicry.
- **Hard-negative augmentation** (optional, see `data_builder/build_hard_negative.py`) adds curated benign examples that pattern-match injection surface features, cross-deduplicated against all training and evaluation files.

---

## Results

### Table 1 — Main Comparison (all models)

| Model | Kind | Test F1 | Hard-Benign FPR | Obfuscated Recall | Domain-OOD F1 | Structural-OOD FPR | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Rule-based | heuristic floor | 0.067 | 0.229 | 0.025 | 0.133 | 0.322 | — |
| TF-IDF + LR | classical baseline | 0.967 | 0.390 | 0.965 | 0.951 | 0.792 | 0.994 |
| DistilBERT | efficient neural | 0.982 | 0.448 | 0.985 | 0.906 | 0.732 | 0.998 |
| DeBERTa-v3 | strong in-repo reference | 0.989 | 0.349 | 0.993 | 0.983 | 0.825 | 0.999 |
| ProtectAI small | external PI detector | 0.884 | 0.164 | 0.844 | 0.891 | 0.443 | — |
| DeBERTa-PI | external PI detector | 0.902 | 0.216 | 0.896 | 0.910 | 0.453 | — |
| PromptGuard 2 (86M) | broad malicious-prompt comparator | 0.539 | 0.152 | 0.324 | 0.190 | 0.332 | — |
| GPT-4o-mini (0-shot) | zero-shot LLM baseline | 0.744 | 0.174 | 0.689 | 0.942 | 0.304 | 0.942 |

*Hard-benign FPR: lower is better. Obfuscated Recall: higher is better. External detectors use binary outputs without per-input probability scores and are excluded from ROC-AUC comparison.*

### Table 2 — In-Distribution Performance with 95% Confidence Intervals

| Model | Test F1 | 95% CI | Hard-Benign FPR | 95% CI | Domain-OOD F1 | 95% CI | Structural-OOD FPR | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rule-based | 0.067 | [0.047, 0.088] | 0.229 | [0.208, 0.250] | 0.133 | [0.105, 0.159] | 0.322 | [0.280, 0.368] |
| TF-IDF + LR | 0.967 | [0.960, 0.975] | 0.390 | [0.366, 0.415] | 0.951 | [0.941, 0.961] | 0.792 | [0.752, 0.830] |
| DistilBERT | 0.982 | [0.976, 0.987] | 0.448 | [0.424, 0.474] | 0.906 | [0.891, 0.919] | 0.732 | [0.707, 0.756] |
| DeBERTa-v3 | 0.989 | [0.984, 0.993] | 0.349 | [0.325, 0.373] | 0.983 | [0.978, 0.989] | 0.825 | [0.789, 0.854] |

### Table 3 — Family-Level Sensitivity Analysis

To confirm that the paraphrase expansion design does not inflate reported metrics, we re-evaluate treating all paraphrases of the same seed as a single family unit.

| Model | Test F1 (row) | Test F1 (family) | Δ | Structural FPR (row) | Structural FPR (family) | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Rule-based | 0.067 | 0.061 | −0.006 | 0.322 | 0.231 | −0.091 |
| TF-IDF + LR | 0.967 | 0.978 | +0.011 | 0.792 | 0.826 | +0.034 |
| DistilBERT | 0.982 | 0.987 | +0.005 | 0.732 | 0.838 | +0.106 |
| DeBERTa-v3 | 0.989 | 0.990 | +0.001 | 0.825 | 0.850 | +0.025 |

Family-level deltas are small in the IID direction and, notably, slightly larger for structural-OOD FPR — confirming that the over-defense finding is not a paraphrase artifact.

### Key Findings

**Over-defense is universal.** Every model in the study raises its false positive rate substantially on the hard-benign stress set relative to the standard test set. DeBERTa-v3 produces the best IID F1 (0.989) but still flags 34.9% of adversarially curated benign prompts. The structural-OOD benign FPR for all fine-tuned models exceeds 73%, indicating that models associate structural surface features with injections as a training artifact.

**Obfuscation is largely solved by fine-tuned models.** TF-IDF+LR, DistilBERT, and DeBERTa-v3 all achieve above 96.5% recall on the 405-injection obfuscated set. The rule-based system collapses to 2.5%, confirming that pattern matching alone cannot handle modern injection diversity.

**Distribution shift exposes model-specific weaknesses.** Under domain shift, DeBERTa-v3 retains strong F1 (0.983) while DistilBERT degrades to 0.906 and the Code domain specifically reduces DistilBERT recall to 44.8%. Under structural shift, all models experience severe benign false-positive pressure — DeBERTa-v3's benign FPR on the instruction-prefix-mimic transform reaches 84.7%.

**Four qualitatively distinct failure profiles emerge.** The eight-model study reveals that detector evaluation is not a leaderboard problem. The four profiles are: (1) over-defense, where fine-tuned models achieve high attack recall at the cost of benign precision; (2) attack–defense tradeoff, where stronger detectors are also more over-defensive; (3) under-detection, where conservative models suppress false positives by missing most attacks; and (4) obfuscation-blind generalism, where a zero-shot LLM handles structural shift better than any fine-tuned model but fails on obfuscated injections.

---

## Repository Structure

```
data/pids_bench_v3/               Frozen benchmark dataset
  train.csv                       27,093 rows — model training
  val.csv                         3,989 rows — threshold tuning only
  test.csv                        3,918 rows — in-distribution evaluation
  eval_subsets/
    hard_benign_test.csv          1,472 adversarial benign inputs (over-defense)
    balanced_subtype_test.csv     2,297 rows balanced across attack subtypes
    obfuscated_attacks.csv        405 obfuscated injections (6 transform types)
  ood/
    domain_ood.csv                2,000 rows across Medical, Legal, Finance, Code
    structural_ood.csv            1,998 rows across 3 structural transforms
  FREEZE_MANIFEST.txt             SHA-256 checksums — authoritative source of truth

src/baselines/
  rule_based.py                   Regex + heuristic detector (no training)
  tfidf_logreg.py                 TF-IDF + Logistic Regression
  distilbert_baseline.py          DistilBERT-base-uncased fine-tuned on PIDS-Bench v3
  deberta_v3.py                   DeBERTa-v3-base fine-tuned on PIDS-Bench v3
  llm_zeroshot_baseline.py        GPT-4o-mini zero-shot baseline (OpenAI API)

src/external_eval/                ProtectAI and LlamaGuard evaluation scripts
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

eval/
  threshold_analysis.py           Val-set threshold tuning and ROC/PR curve generation
  plot_roc_pr_curves.py           Multi-model ROC and precision-recall plots
  generate_probability_scores.py  Probability score generation for TF-IDF and others
  threshold_fpr_tradeoff.py       Threshold vs. FPR tradeoff analysis
  adaptive_attack_loop.py         Adaptive attack evaluation loop
  source_breakdown.py             Performance breakdown by source type
  surrogate_deberta.py            Surrogate model for adaptive attack transfer

scripts/
  consolidate_results.py          Aggregate outputs/*/summary.json into summary tables
  breakdown_analysis.py           Per-domain and per-transform breakdown analysis
  family_ci_analysis.py           Family-level bootstrap confidence intervals
  build_final_paper_artifacts.py  Generate all paper-ready tables and figures
  run_external_multiaxis.py       Run all external detector evaluations

reports/
  BASELINE_PERFORMANCE.md         Full per-axis results for all in-repo baselines
  EXTERNAL_BASELINES_CORE_RESULTS.md  External detector results and failure profile analysis
  FINAL_RESULTS_TABLES.md         Paper-facing consolidated tables (Tables 1–4)
  BREAKDOWN_ANALYSIS_CORE_RESULTS.md  Per-domain and per-transform breakdowns
  FAMILY_CI_OOD_RESULTS.md        Family-level sensitivity analysis results
  PIDS_BENCH_V3_REPORT.md         Full benchmark specification and checksums

outputs/                          Experiment results (git-ignored; generated locally)
models/                           Trained model checkpoints (git-ignored; generated locally)
```

---

## Setup

**Requirements:** Python 3.10 or later. A GPU or Apple Silicon MPS is recommended for the transformer baselines; the rule-based and TF-IDF baselines run efficiently on CPU.

```bash
git clone https://github.com/ShirePyDev/Prompt-Injection-Detection-System.git
cd Prompt-Injection-Detection-System
pip install -r requirements.txt
```

For Colab or Kaggle environments, use `requirements_colab.txt` instead, which is stripped of environment-specific dependencies. The training notebooks `colab_deberta_training.ipynb` and `kaggle_deberta_training.ipynb` are self-contained.

If you plan to rebuild the OOD sets or run the LLM zero-shot baseline, create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

## Running the Baselines

All four in-repository baselines follow the same protocol: train on `train.csv`, tune a classification threshold on `val.csv` only (F1-maximising grid search), and apply that threshold once to test and all stress sets. Threshold selection is never informed by test data.

### Rule-Based Detector

```bash
python -m src.baselines.rule_based
```

No training required. Outputs a classification report and summary to `outputs/rule_baseline/`.

### TF-IDF + Logistic Regression

```bash
python -m src.baselines.tfidf_logreg
```

Trains in under a minute on CPU. Val-tuned threshold: τ = 0.518. Outputs to `outputs/tfidf_baseline/`.

### DistilBERT

```bash
python -m src.baselines.distilbert_baseline
```

Fine-tunes `distilbert-base-uncased` for 2 epochs (batch size 16, lr 5e-5, seed 42). Estimated runtime: approximately 1.5 hours on Apple Silicon MPS. Val-tuned threshold: τ = 0.45. Model saved to `models/distilbert/`, outputs to `outputs/distilbert_baseline/`.

To include hard-negative augmentation during training:

```bash
python -m src.baselines.distilbert_baseline --hard-neg
```

### DeBERTa-v3-base

```bash
python -m src.baselines.deberta_v3
```

Fine-tunes `microsoft/deberta-v3-base` for 3 epochs (batch size 16, lr 2e-5, seed 42). Estimated runtime: approximately 3–5 hours on Apple Silicon MPS or a Colab T4. Val-tuned threshold: τ = 0.502. Outputs to `outputs/deberta_v3/`, including OOD evaluation, obfuscation recall, and threshold analysis automatically.

For Colab, use the provided notebook:

```
colab_deberta_training.ipynb
```

---

## Threshold Tuning and ROC Analysis

All probabilistic detectors support threshold analysis. The threshold is always selected on `val.csv` and applied once without further adjustment:

```bash
python eval/threshold_analysis.py \
  --val_csv outputs/distilbert_baseline/val_predictions.csv \
  --test_csv outputs/distilbert_baseline/test_predictions.csv \
  --out_dir outputs/distilbert_baseline/threshold_analysis
```

To generate multi-model ROC and precision-recall curves on the structural-OOD split:

```bash
python eval/plot_roc_pr_curves.py --splits structural_ood
```

---

## Evaluation

### Reproducing the Main Results Table

After running all four baselines, consolidate the summary files:

```bash
python scripts/consolidate_results.py
```

This reads `outputs/*/summary.json` and regenerates the main comparison tables in `reports/`.

### Per-Domain and Per-Transform Breakdowns

```bash
python scripts/breakdown_analysis.py
```

Produces per-domain FPR and recall for `domain_ood`, and per-transform breakdown for `structural_ood`, written to `outputs/breakdown_analysis/`.

### Family-Level Confidence Intervals

```bash
python scripts/family_ci_analysis.py
```

Bootstrap resampling at the seed-family level (1,000 iterations, seed 42) to confirm that confidence intervals are not deflated by correlated paraphrase rows.

### External Detectors

```bash
python scripts/run_external_multiaxis.py
```

Requires HuggingFace token for ProtectAI, DeBERTa-PI, and PromptGuard 2, and an OpenAI API key for the GPT-4o-mini baseline. Results are written to `outputs/external_eval/multiaxis/`.

---

## Reproducibility

All results in the paper are fully reproducible from this repository:

- Training uses fixed random seed (`seed=42`) throughout all baselines.
- All evaluation sets are frozen under SHA-256 checksums in `data/pids_bench_v3/FREEZE_MANIFEST.txt`.
- Threshold tuning operates exclusively on `val.csv` and is applied once to all evaluation sets without re-tuning.
- All hyperparameters (learning rate, batch size, epochs, threshold grid) are hardcoded in each baseline script with no external configuration required.
- LLM prompts used to construct the OOD sets are documented in `data_builder/build_domain_ood.py` and `data_builder/build_structural_ood.py`.
- The complete deduplication pipeline, split logic, and paraphrase generation procedure are implemented in `data_builder/utils/` and `data_builder/build_dataset.py`.

---

## Citation

If you use PIDS-Bench or any component of this repository in your work, please cite:

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

## License

This repository is released under the MIT License. See `LICENSE` for details.
The benchmark dataset (`data/pids_bench_v3/`) is released for research use. Source datasets incorporated into the benchmark retain their original licenses; see `data_builder/sources/` for provenance documentation.
