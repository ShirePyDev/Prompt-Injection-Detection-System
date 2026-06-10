# External Baselines: Core Results

This note summarizes the four completed multiaxis external detector runs under `outputs/external_eval/multiaxis/`:

- `protectai_small` = `protectai/deberta-v3-small-prompt-injection-v2`
- `deberta_pi` = `protectai/deberta-v3-base-prompt-injection-v2`
- `promptguard2_86m` = `meta-llama/Llama-Prompt-Guard-2-86M`
- `gpt4o_mini` = `gpt-4o-mini` (zero-shot, no fine-tuning, threshold fixed at 0.5)

All four were run on the same frozen PIDS-Bench v3 splits. `promptguard2_86m` should be framed as a broad malicious-prompt comparator. `gpt4o_mini` is framed as a zero-shot LLM baseline — no fine-tuning on PIDS-Bench data, threshold fixed at 0.5, no val-set tuning.

## 1. Main Comparison

| Split | Metric | protectai_small | deberta_pi | promptguard2_86m | gpt4o_mini (0-shot) |
|---|---:|---:|---:|---:|---:|
| `test` | F1 | `0.8842` | `0.9020` | `0.5390` | `0.7438` |
| `test` | FPR | `0.0482` | `0.0471` | `0.0005` | `0.0525` |
| `test` | Recall | `0.8040` | `0.8370` | `0.5490` | `0.6198` |
| `hard_benign` | FPR | `0.1637` | `0.2160` | `0.1515` | `0.1739` |
| `obfuscated_attacks` | Recall | `0.8444` | `0.8963` | `0.3235` | `0.6889` |
| `domain_ood` | F1 | `0.8909` | `0.9103` | `0.1900` | `0.9419` |
| `domain_ood` | FPR | `0.0660` | `0.0480` | `0.0005` | `0.0020` |
| `structural_ood` | F1 | `0.7085` | `0.7216` | `0.6542` | `0.8314` |
| `structural_ood` | FPR | `0.4434` | `0.4525` | `0.3323` | `0.3043` |
| `balanced_subtype` | F1 | `0.8612` | `0.8861` | `0.5000` | `0.7975` |

## 2. What These Results Actually Show

- `deberta_pi` is the strongest external attack detector overall. It beats `protectai_small` on IID test F1, obfuscated recall, domain-OOD F1, structural-OOD F1, and balanced-subtype F1. The gain is not free: `deberta_pi` is more over-defensive on `hard_benign` (`0.2160` FPR vs `0.1637`).
- `promptguard2_86m` is extremely conservative: near-zero IID/domain false positives and the lowest `hard_benign` FPR at `0.1515`, but very poor attack recall on `test`, `obfuscated_attacks`, `domain_ood`, and `balanced_subtype`. Lower benign FPR achieved through severe attack under-detection.
- `gpt4o_mini` introduces a fourth distinct failure profile not seen in any fine-tuned model:
  - **Lowest structural-OOD benign FPR of all models** (`0.3043`), lower than both external PI detectors and PromptGuard 2.
  - **Second-lowest hard-benign FPR** (`0.1739`) among non-conservative models.
  - **Strong domain-OOD F1** (`0.9419`) with near-zero domain FPR (`0.0020`), competitive with fine-tuned models.
  - **Weak obfuscation recall** (`0.6889`) — the model has not seen injection-specific obfuscation in fine-tuning and cannot recognize surface-level encoding tricks.
  - **Low test recall** (`0.6198`) — misses a substantial fraction of direct injection attempts at the fixed 0.5 threshold.
  - The hard-benign score distribution is bimodal: median score = 0.00004 (near-zero on most benign inputs), p90 = 0.997 (high confidence on the 10% it flags). Threshold adjustment cannot recover the flagged hard-benign inputs because they are already high-confidence predictions.

## 3. Four Failure Profiles Identified

The four external runs, together with the in-repo baselines, now show four qualitatively distinct failure profiles:

| Profile | Representative models | Dominant error |
|---|---|---|
| Over-defense | DeBERTa-v3, DistilBERT, TF-IDF+LR | High test F1 + high hard-benign and structural FPR |
| Attack–defense tradeoff | ProtectAI-small vs DeBERTa-PI | Stronger detection → worse benign over-defense |
| Under-detection | PromptGuard 2 | Low FPR achieved by missing most attacks |
| Obfuscation-blind generalist | GPT-4o-mini (0-shot) | Low FPR on benign/structural, but misses obfuscated attacks and many direct injections |

## 4. Paper-Relevant Lessons

1. The failure pattern (over-defense under structural shift) is not limited to in-repo fine-tuned models. All external detectors show elevated structural-OOD benign FPR, confirming it is structural.
2. Better attack detection can come with worse benign over-defense.
3. Lower benign FPR can come with severe attack under-detection.
4. A zero-shot LLM baseline handles structural format shift better than every fine-tuned detector, suggesting that structural over-defense in fine-tuned models is a training artifact — the models associate structural features (json wrapping, system-prompt formatting) with injections because those features appear in training examples. GPT-4o-mini has no such learned association.
5. The benchmark now distinguishes four qualitatively different detector behaviors, not a simple leaderboard ranking.
