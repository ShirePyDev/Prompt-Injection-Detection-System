# PIDS-Bench: Evaluating Prompt-Injection Detectors Under Over-Defense, Obfuscation, and Distribution Shift

> **Paper:** Yusuf Khalid Shire and Sang-Chul Kim. "PIDS-Bench: Evaluating Prompt-Injection Detectors Under Over-Defense, Obfuscation, and Distribution Shift." *IEEE Access*, vol. 14, 2026. [DOI: 10.1109/ACCESS.2026.3728186](https://doi.org/10.1109/ACCESS.2026.3728186) · [arXiv: 2609.15017](https://arxiv.org/abs/2609.15017)

> **Plain-language write-up:** [Prompt-injection detectors, tested on prompts they should let through](https://shirepydev.github.io/posts/pids-bench/)


This repository contains the frozen benchmark, evaluation code, and dataset-construction pipeline behind the paper. Rows whose upstream licenses prevent redistribution require the local rebuild described below.

---

## Overview

Prompt-injection detectors are increasingly deployed as gateway classifiers in front of LLM systems, yet most published evaluations summarize a detector with a single F1 score on a single in-distribution test set. That practice hides the errors that matter once a detector is deployed — in particular, how often it flags *benign* inputs as attacks.

PIDS-Bench evaluates a detector along multiple axes at once: in-distribution accuracy, benign false-positive behaviour on inputs that superficially resemble attacks, detection on obfuscated attacks, and behaviour under domain and structural distribution shift. We evaluate seven detectors spanning classical, fine-tuned, external, and broad-safety designs, with a rule-based heuristic reported separately as a lower-bound reference.

The strongest internal detector, DeBERTa-v3-FT, reaches IID F1 0.9882 and flags 0.3144 ± 0.0525 of the externally-sourced hard-benign prompts (n = 872) at τ = 0.5. Its aggregate hard-benign FPR is 0.3825 ± 0.0474 over the full 1,472-row stress set; this aggregate describes a stress set, not a deployment rate. Hard-negative augmentation nearly eliminates curated over-defense but leaves the externally-sourced FPR largely unchanged; the paper calls this augmentation result **provenance-sensitive over-defense**.

The paper evaluates seven detectors: TF-IDF + LR, DistilBERT, and DeBERTa-v3-FT trained here; the ProtectAI model and DeBERTa-PI; and PromptGuard 2 (86M) and Llama Guard 3-1B. A rule-based detector is a separate lower-bound reference. GPT-4o-mini generated paraphrases during dataset construction and is not an evaluated detector.

---

## What the benchmark measures

At τ = 0.5, DeBERTa-v3-FT has IID F1 0.9882 and flags 0.3144 ± 0.0525 of the externally-sourced hard-benign subset (n = 872). Its aggregate FPR is 0.3825 ± 0.0474 over all 1,472 hard-benign rows. The aggregate describes a constructed stress set, not a deployment false-positive rate.

The benchmark targets three failure modes that single-metric evaluation systematically hides:

| Failure mode | What it captures | Evaluation set |
|---|---|---|
| **Over-defense** | The detector flags benign prompts that pattern-match attacks | `hard_benign_test` — 1,472 benign inputs that mimic injection structure; 664 LMSYS text fields are withheld in the public release |
| **Obfuscation blindness** | Detection degrades when attacks are encoded or surface-perturbed | `obfuscated_attacks` — 405 injections across seven released transform labels |
| **Distribution shift** | Performance changes outside the training domain or surface form | `domain_ood` (specialist domains) and `structural_ood` (structural transforms) |

A fourth axis, `balanced_subtype_test`, is a per-subtype diagnostic disaggregation of the in-distribution split rather than a separate stress condition; it is used to check per-attack-type fairness, not to define a failure mode.

---

## Benchmark Specification — PIDS-Bench v3

The benchmark is frozen under SHA-256 checksums recorded in:

```text
data/pids_bench_v3/FREEZE_MANIFEST.txt
```

(frozen 2026-03-24).

The manifest is the authoritative source of truth for row counts and hashes. Evaluation sets are static and must not be modified.

The public `hard_benign_test.csv` has 1,472 rows, but the `text` field is empty for 664 LMSYS rows. Those prompts require a local rebuild from a licensed LMSYS-Chat-1M copy before evaluating all 1,472 examples.

### Dataset Splits

| Split                   |       Rows | Benign | Injection | Purpose                       |
| ----------------------- | ---------: | -----: | --------: | ----------------------------- |
| `train`                 |     27,093 | 12,846 |    14,247 | Model training                |
| `val`                   |      3,989 |  1,943 |     2,046 | Threshold tuning              |
| `test`                  |      3,918 |  1,848 |     2,070 | In-distribution evaluation    |
| `hard_benign_test`      |     1,472 | 1,472 |         0 | Over-defense stress test; 664 LMSYS text fields withheld |
| `balanced_subtype_test` |      2,297 |  1,200 |     1,097 | Per-subtype diagnostics       |
| `obfuscated_attacks`    |        405 |      0 |       405 | Detection under obfuscation   |
| `domain_ood`            |      2,000 |  1,000 |     1,000 | Domain distribution shift     |
| `structural_ood`        |      1,998 |    999 |       999 | Structural distribution shift |
| **Total**               | **43,172** |        |           |                               |

---

## Dataset construction

The dataset is built by the pipeline in `data_builder/`:

- **Seed collection** draws benign and injection examples from several public sources.
- **Deduplication** applies exact-match normalization followed by TF-IDF character n-gram near-duplicate filtering (cosine threshold 0.95) across the full seed pool **before** the split is applied.
- **Seed-family splitting** assigns every paraphrase and obfuscated descendant of a seed to the same partition before train/validation/test are separated, so structurally similar examples cannot straddle training and evaluation.
- **Paraphrase expansion** generates surface variants for the two scarce attack subtypes (`encoded_attack`, `tool_injection`) using GPT-4o-mini as a paraphrase generator at construction time. This is data-construction tooling only; GPT-4o-mini is not an evaluated detector.
- **Obfuscation augmentation** in the released `obfuscated_attacks.csv` has seven labels: `unicode_confusables` (90), `spacing_jitter` (61), `markdown_split` (58), `homoglyph` (55), `zero_width` (53), `base64_wrap` (50), and `rot13_wrap` (38). All seven labels also occur in `train.csv`, so the released set measures known transforms on unseen text.
- **Hard-negative augmentation** (optional, `data_builder/build_hard_negative.py`) adds curated benign examples that pattern-match injection surface features, cross-deduplicated against all training and evaluation files.

---

## Results

The published headline metrics for the internal models are five-seed means at fixed τ = 0.5; TF-IDF + LR is deterministic and reported without variance. The paper separately reports validation-F1 threshold selections as diagnostics. They vary across seeds: DeBERTa-v3-FT ranges from 0.01 to 0.904, DistilBERT from 0.112 to 0.664, and TF-IDF + LR selects 0.518 in every run. The neural-model threshold instability is not the threshold used for the headline table.

### Internal baselines

Five-seed mean ± standard deviation at fixed threshold τ = 0.5. Hard-benign FPR is the aggregate over the full set (n = 1,472). ↑ higher is better; ↓ lower is better.

| Model | IID F1 ↑ | Real-source F1 ↑ | IID Ben-FPR ↓ | Hard-ben. FPR ↓ | Obf. Recall ↑ | Domain-shift F1 ↑ | Struct.-shift FPR ↓ | ROC-AUC ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR | 0.9656 | 0.9467 | 0.0514 | 0.4090 | 0.9654 | 0.9508 | 0.7918 | 0.9941 |
| DistilBERT | 0.9831 ± 0.0017 | 0.9738 ± 0.0026 | 0.0245 ± 0.0047 | 0.4702 ± 0.0823 | 0.9877 ± 0.0058 | 0.8886 ± 0.0214 | 0.7826 ± 0.0963 | 0.9974 ± 0.0002 |
| DeBERTa-v3-FT | 0.9882 ± 0.0008 | 0.9825 ± 0.0009 | 0.0124 ± 0.0013 | 0.3825 ± 0.0474 | 0.9867 ± 0.0079 | 0.9604 ± 0.0099 | 0.8108 ± 0.0617 | 0.9983 ± 0.0004 |

### Hard-benign over-defense by source origin

False-positive rate on the hard-benign set at τ = 0.5, decomposed by the provenance of the benign input. The effect of hard-negative augmentation on these subsets is shown in Table 10.

| Model | Aggregate ↓ | Externally-sourced (n = 872) ↓ | Curated (n = 600) ↓ |
|---|---:|---:|---:|
| TF-IDF + LR | 0.4090 | 0.4622 | 0.3317 |
| DistilBERT | 0.4702 ± 0.0823 | 0.4606 ± 0.0639 | 0.4843 ± 0.0793 |
| DeBERTa-v3-FT | 0.3825 ± 0.0474 | 0.3144 ± 0.0525 | 0.4813 ± 0.0855 |

> **Note on direction.** At baseline, DeBERTa-v3-FT flags fewer externally-sourced prompts than curated prompts (0.3144 vs. 0.4813). DistilBERT over-blocks both subsets at comparable rates, and TF-IDF over-blocks the externally-sourced subset more than the curated one. Thus baseline false-positive rates differ by model and subset.

### External and broad-safety detectors

Model IDs identified in the paper: ProtectAI (`protectai/deberta-v3-base-prompt-injection-v2`), DeBERTa-PI (`deepset/deberta-v3-base-injection`), PromptGuard 2 (86M) (`meta-llama/Llama-Prompt-Guard-2-86M`), and Llama Guard 3-1B (`meta-llama/Llama-Guard-3-1B`). The table gives the default τ = 0.5 rows for the external prompt-injection detectors and marks validation-swept rows as diagnostic. Broad-safety models are shown at their native operating points.

| Detector and operating point | IID F1 ↑ | IID recall ↑ | Hard-benign FPR ↓ | Obf. recall ↑ | Domain-shift F1 ↑ | Struct.-shift FPR ↓ | ROC-AUC ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ProtectAI (τ = 0.5) | 0.936 | 0.858 | 0.216 | 0.906 | 0.903 | 0.4525 | 0.9741 |
| ProtectAI (τ = 0.002, diagnostic) | 0.928 | 0.913 | 0.322 | 0.965 | 0.9617 | 0.6647 | 0.9741 |
| DeBERTa-PI (τ = 0.5) | 0.804 | 0.944 | 0.804 | 0.970 | 0.7950 | 1.000 | 0.8712 |
| DeBERTa-PI (τ = 0.986, diagnostic) | 0.813 | 0.887 | 0.694 | 0.943 | 0.7973 | 1.000 | 0.8712 |
| PromptGuard 2 (86M; native) | 0.5390 | 0.369 | 0.1515 | 0.325 | 0.1900 | 0.3323 | — |
| Llama Guard 3-1B (native) | 0.5692 | 0.420 | 0.1800 | 0.484 | 0.7923 | 0.2222 | — |

### Rule-based lower-bound reference

The rule-based heuristic is reported separately as a lower-bound reference and is outside the seven-detector comparison. It has F1 0.067, with high precision and low recall.

---

## Key Findings

**In-distribution F1 does not predict benign-side safety.** DeBERTa-v3-FT has IID F1 0.9882 but flags 0.3144 ± 0.0525 of the externally-sourced hard-benign subset at τ = 0.5. Across a full threshold sweep and five seeds, no internal detector reaches an operating point that simultaneously satisfies F1 ≥ 0.95 and externally-sourced hard-benign FPR ≤ 0.10 on this security-adjacent stress distribution.

**Over-defense is not a threshold artifact.** The benign inputs that are over-blocked are scored deep in the injection region, not marginally above the decision boundary, so raising the threshold does not separate them from true positives without destroying recall.

**Provenance-sensitive over-defense under augmentation.** Curated hard-negative augmentation nearly eliminates FPR on the curated subset (DeBERTa-v3-FT 0.4813 → 0.0010; DistilBERT 0.4843 → 0.0123) but barely changes the externally-sourced FPR (DeBERTa-v3-FT 0.3144 → 0.3101; DistilBERT 0.4606 → 0.3899). The augmentation was matched to the curated subset; whether augmentation matched to the externally-sourced distribution would close the gap is untested.

**Obfuscation is largely handled by the learned baselines.** TF-IDF + LR, DistilBERT, and DeBERTa-v3-FT all exceed 96% recall on the obfuscated set, while the rule-based heuristic has low recall. In a check using the repository's TF-IDF pipeline settings and scikit-learn 1.9.1, TF-IDF + LR detects 53 of the 55 released homoglyph attacks at τ = 0.5.

**Distribution shift exposes model-specific weaknesses.** Under domain shift, DeBERTa-v3-FT retains strong F1 (0.9604) while DistilBERT degrades (0.8886), with the code domain specifically reducing DistilBERT recall to 0.435 — a within-class content shift, not generalization to an unseen attack distribution. Under structural shift, all fine-tuned models experience severe benign false-positive pressure, with DeBERTa-v3-FT's structural FPR averaging 0.8108 across transforms and `instruction_prefix_mimic` the worst single transform.

---

## Repository structure

```
data/pids_bench_v3/               Frozen benchmark dataset
  train.csv                       27,093 rows — model training
  val.csv                         3,989 rows — threshold tuning only
  test.csv                        3,918 rows — in-distribution evaluation
  eval_subsets/
    hard_benign_test.csv          1,472 benign rows (664 LMSYS texts withheld)
    balanced_subtype_test.csv     2,297 rows balanced across attack subtypes
    obfuscated_attacks.csv        405 obfuscated injections (seven transform labels)
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

## Setup

**Requirements:** Python 3.10 or later. A CUDA GPU or Apple Silicon (MPS) is recommended for the transformer baselines; the rule-based and TF-IDF baselines run on CPU.

```bash
git clone https://github.com/ShirePyDev/Prompt-Injection-Detection-System.git
cd Prompt-Injection-Detection-System
pip install -r requirements.txt
```

The repository includes the Colab training notebook at `notebooks/deberta_colab.ipynb`.

If you plan to rebuild the OOD sets or run anything that calls a hosted API or a gated HuggingFace model, create a `.env` file in the project root with your own credentials:

```
OPENAI_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

The `.env` file is git-ignored and must never be committed.

---

## Running the baselines

The scripts train learned baselines on `train.csv` and may report validation-tuned thresholds for diagnostic outputs. The published headline metrics use fixed τ = 0.5, not those per-run thresholds.

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

The paper's internal headline numbers are five-seed aggregates. To reproduce the multi-seed runs, use the scripts under `scripts/` (for example, the DeBERTa and DistilBERT multi-seed runners) with seeds {13, 42, 123, 2024, 7777}, then aggregate the results at τ = 0.5.

---

## Evaluation

### Rebuilding restricted LMSYS text

The public `hard_benign_test.csv` has 1,472 rows, of which 664 LMSYS text fields are empty because the LMSYS-Chat-1M license forbids redistribution. Without rebuilding, only 808 rows have non-empty text; evaluating the complete set requires a licensed local copy.

After accepting the LMSYS-Chat-1M license, set `HF_TOKEN` or pass `--hf-token`, then run this from the repository root:

```bash
python data_builder/rebuild_restricted.py --data-root data/pids_bench_v3
```

The rebuild matches the released row fingerprints, which are also listed in `data/pids_bench_v3/restricted_rebuild_manifest.csv`. See `DATA_LICENSES.md`.

### Evaluation commands

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

## Reproducibility

- The fine-tuned results are five-seed means over seeds {13, 42, 123, 2024, 7777}, reported with standard deviations.
- All evaluation sets are frozen under SHA-256 checksums in `data/pids_bench_v3/FREEZE_MANIFEST.txt`; the obfuscation rebuild and its zero-overlap checks are documented there.
- The headline internal metrics use fixed τ = 0.5. Validation-F1 thresholds vary across seeds for the neural models and are reported separately as diagnostics.
- Hyperparameters are specified in each baseline script and the construction config; the pinned software environment is recorded in `requirements.txt`.
- The deduplication, seed-family split, and paraphrase-generation procedures are implemented in `data_builder/utils/` and `data_builder/build_dataset.py`.

A note on scope: the `domain_ood` injection side is drawn from held-out injection data that shares provenance with the training injection class, so its attack-side numbers measure content-level transfer within the injection class rather than generalization to an unseen attack distribution. In a blind two-annotator audit of 200 hard-benign rows, the annotators agreed on 199 rows (Cohen's κ = 0.85).

---

## Citation

If you use PIDS-Bench in your work, please cite:

```bibtex
@article{pids_bench_2026,
  author  = {Shire, Yusuf Khalid and Kim, Sang-Chul},
  title   = {{PIDS-Bench: Evaluating Prompt-Injection Detectors Under
             Over-Defense, Obfuscation, and Distribution Shift}},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {134184--134205},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3728186}
}
```

---

## License

Code is released under the MIT License (see `LICENSE`). Data is for research and non-commercial use and inherits the upstream licenses; see `DATA_LICENSES.md`.
