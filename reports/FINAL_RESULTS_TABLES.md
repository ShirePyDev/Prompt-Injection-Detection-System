# Final Results Tables

These tables and figures are built directly from the saved summaries under `outputs/`. Family-level support tables additionally use local CI intermediates during paper production.

- `outputs/rule_baseline/summary.json`
- `outputs/tfidf_baseline/summary.json`
- `outputs/distilbert_baseline/summary.json`
- `outputs/deberta_v3/summary.json`
- `outputs/external_eval/multiaxis/*/summary.json`
- `reports/FAMILY_CI_OOD_RESULTS.md`
- local family-bootstrap CI intermediates (used to refresh Tables 3 and 4)

## Table 1. Focused detector comparison

This is the clearest main-table candidate for the paper because it keeps attention on the benchmark's central lesson: high IID performance does not guarantee benign-side safety or structural robustness.

| Model | Kind | Test F1 | Test F1 (real) | Test F1 (tmpl) | Hard-benign FPR | Obfuscated Recall | Domain-OOD F1 | Structural-OOD FPR | ROC-AUC (test) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule-based | heuristic floor | 0.0672 | 0.0356 | 0.1153 | 0.2289 | 0.0247 | 0.1326 | 0.3223 | — |
| TF-IDF+LR | classical baseline | 0.9672 | 0.9497 | 0.9956 | 0.3899 | 0.9654 | 0.9508 | 0.7918 | 0.9941 |
| DistilBERT | efficient neural baseline | 0.9820 | 0.9723 | 0.9975 | 0.4484 | 0.9852 | 0.9055 | 0.7317 | 0.9976 |
| DeBERTa-v3 | strong in-repo reference | 0.9889 | 0.9827 | 0.9988 | 0.3492 | 0.9926 | 0.9834 | 0.8248 | 0.9986 |
| ProtectAI small | external PI detector | 0.8842 | — | — | 0.1637 | 0.8444 | 0.8909 | 0.4434 | — |
| DeBERTa-PI | external PI detector | 0.9020 | — | — | 0.2160 | 0.8963 | 0.9103 | 0.4525 | — |
| PromptGuard 2 (86M) | broad malicious-prompt comparator | 0.5390 | — | — | 0.1515 | 0.3235 | 0.1900 | 0.3323 | — |
| GPT-4o-mini (0-shot) | zero-shot LLM baseline | 0.7438 | — | — | 0.1739 | 0.6889 | 0.9419 | 0.3043 | 0.9415 |

## Table 2. Extended multiaxis comparison

This table is useful as the fuller main-results table or as a compact appendix table.

| Model | Kind | Test F1 | Test FPR | Balanced-Subtype F1 | Domain-OOD F1 | Structural-OOD F1 | Structural-OOD FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule-based | heuristic floor | 0.0672 | 0.0000 | 0.0670 | 0.1326 | 0.3384 | 0.3223 |
| TF-IDF+LR | classical baseline | 0.9672 | 0.0471 | 0.9668 | 0.9508 | 0.6891 | 0.7918 |
| DistilBERT | efficient neural baseline | 0.9820 | 0.0249 | 0.9788 | 0.9055 | 0.6121 | 0.7317 |
| DeBERTa-v3 | strong in-repo reference | 0.9889 | 0.0119 | 0.9882 | 0.9834 | 0.7034 | 0.8248 |
| ProtectAI small | external PI detector | 0.8842 | 0.0482 | 0.8612 | 0.8909 | 0.7085 | 0.4434 |
| DeBERTa-PI | external PI detector | 0.9020 | 0.0471 | 0.8861 | 0.9103 | 0.7216 | 0.4525 |
| PromptGuard 2 (86M) | broad malicious-prompt comparator | 0.5390 | 0.0005 | 0.5000 | 0.1900 | 0.6542 | 0.3323 |
| GPT-4o-mini (0-shot) | zero-shot LLM baseline | 0.7438 | 0.0525 | 0.7975 | 0.9419 | 0.8314 | 0.3043 |

## Table 3. Internal statistical support

This table is the clean support table for the benchmark's main claim. It keeps the CI evidence focused on the internal baselines where family-level analysis was completed.

| Model | Test F1 | Test F1 95% CI | Hard-benign FPR | Hard-benign FPR 95% CI | Domain-OOD F1 | Domain-OOD F1 95% CI | Structural-OOD FPR | Structural-OOD FPR 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule-based | 0.0672 | [0.0467, 0.0881] | 0.2289 | [0.2079, 0.2500] | 0.1326 | [0.1047, 0.1592] | 0.3223 | [0.2799, 0.3677] |
| TF-IDF+LR | 0.9672 | [0.9598, 0.9746] | 0.3899 | [0.3662, 0.4151] | 0.9508 | [0.9412, 0.9607] | 0.7918 | [0.7523, 0.8304] |
| DistilBERT | 0.9820 | [0.9762, 0.9872] | 0.4484 | [0.4239, 0.4735] | 0.9055 | [0.8908, 0.9185] | 0.7317 | [0.7065, 0.7558] |
| DeBERTa-v3 | 0.9889 | [0.9837, 0.9934] | 0.3492 | [0.3247, 0.3730] | 0.9834 | [0.9777, 0.9890] | 0.8248 | [0.7889, 0.8544] |

## Table 4. Family-level sensitivity

This table shows that the main deployment-safety result is not explained away by repeated paraphrase rows.

| Model | Test Row F1 | Test Family F1 | Test Delta | Structural Row FPR | Structural Family FPR | Structural Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule-based | 0.0672 | 0.0613 | -0.0059 | 0.3223 | 0.2312 | -0.0911 |
| TF-IDF+LR | 0.9672 | 0.9780 | 0.0108 | 0.7918 | 0.8258 | 0.0340 |
| DistilBERT | 0.9820 | 0.9869 | 0.0050 | 0.7317 | 0.8378 | 0.1061 |
| DeBERTa-v3 | 0.9889 | 0.9897 | 0.0008 | 0.8248 | 0.8498 | 0.0250 |

## Figure outputs

- Main tradeoff figure: `outputs/paper_final_assets/figure_test_f1_vs_hard_benign_fpr.svg`
- Secondary tradeoff figure: `outputs/paper_final_assets/figure_test_f1_vs_structural_ood_fpr.svg`

## Paper-facing lesson

- The benchmark now teaches two distinct detector failures: over-defense on benign stress inputs and under-detection when the model becomes overly conservative.
- `DeBERTa-v3` is still the strongest IID model, but that does not protect it from severe hard-benign and structural benign false positives.
- `PromptGuard 2 (86M)` adds value because it exposes the opposite tradeoff: lower benign-side false positives, but heavy attack under-detection.
- The paper can now show that detector evaluation is not a one-number leaderboard problem; it is a tradeoff and failure-mode characterization problem.
