# Prompt Injection Detector

Research-grade implementation of the **Prompt Injection Detector** paper.  
This repository contains data pipelines, classical and neural baselines, and ensemble techniques for classifying jailbreak / injection attempts that try to override system instructions.

---

## Overview
- **End-to-end workflow** for collecting public prompt-injection corpora, generating synthetic attacks, and exporting stratified train/val/test splits.
- **Baseline models**: heuristics, TF–IDF + Logistic Regression, DistilBERT fine-tuning.
- **Meta-ensemble** that fuses rule, sparse, and transformer signals for robust deployment.
- **Analysis utilities** for comparing models, plotting metrics, and logging reports under `outputs/` and `reports/`.

The code is organized so each stage can be run independently (e.g., only classical baselines for lightweight experiments or full LLM-generated data augmentation for reproducing the paper).

---

## Repository Layout
- `src/data/`: dataset download, cleaning, augmentation, and merging scripts.
- `src/baselines/`: rule-based, TF–IDF, and DistilBERT training code.
- `src/ensemble/`: majority vote + meta-learner ensembling.
- `src/analysis/`: utilities for metric comparison and plotting.
- `run_*.py`: click-to-run wrappers for VS Code / CLI.
- `data/`, `models/`, `outputs/`, `reports/`: artifacts written during experiments.

---

## Setup
1. **Python**: 3.10+ (tested on macOS and Linux). Create a virtual environment if possible.
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Secrets**: copy `.env.example` → `.env` and fill in the values you need. A plain `.env` already works with `python-dotenv`.

| Variable | Purpose |
| --- | --- |
| `HUGGINGFACEHUB_API_TOKEN` | Required for downloading gated models and generating synthetic data with Llama 3 via the HF Inference API. |
| `LLAMA_API_KEY` | Optional alternative provider key if you adapt `synthetic_generate.py`. |

> Never commit `.env`. Git ignores it already.

---

## Data Workflow
The project follows a reusable chain: download → normalize → expand/augment → final splits. Each step can be rerun safely.

| Stage | Command | Output |
| --- | --- | --- |
| 1. Base dataset | `python src/data/download_dataset.py` | Hugging Face dataset saved to `data/raw/prompt_injections/`. |
| 2. Clean + split | `python src/data/prepare_dataset.py` | Canonical `data/processed/{train,val,test}.csv` (662-row seed set). |
| 3. Expand public corpora | `python run_expand_datasets.py` | Aggregated `data/processed_expanded/all_prompts.csv` + summary. |
| 4. Split expanded corpus | `python run_split_expanded.py` | Stratified splits inside `data/processed_expanded/`. |
| 5. Generate synthetic attacks (LLM) | `python src/data/synthetic_generate.py` | Llama 3 variants saved under `data/processed_expanded/synthetic/`. Requires `HUGGINGFACEHUB_API_TOKEN`. |
| 6. Local obfuscation augments | `python src/data/local_augment.py` | Cheap perturbations (`synthetic_local_aug.csv`). |
| 7. Merge everything | `python src/data/merge_all_datasets.py` | Final training corpus `data/final_dataset/{train,val,test}.csv`. |

Tips:
- `src/data/expand_datasets.py` already knows how to ingest `deepset/prompt-injections` and `qualifire/prompt-injections-benchmark`. Add more configs to the `DATASETS` list to scale further.
- `synthetic_generate.py` throttles generation (sleep + 50 variants/seed by default). Tune `N_VARIANTS_PER_PROMPT`, `MAX_SEEDS`, and rate limits based on your quota.
- All scripts write JSON summaries describing class balance, so you can inspect drift after each augmentation.

---

## Training & Evaluation
You can run baselines independently or chain them together for the ensemble. All scripts read from `data/final_dataset/` unless noted otherwise.

### 1. Rule-Based Baseline
Uses regexes + heuristics inspired by the paper.
```bash
python run_rule_baseline.py
```
Reports: `outputs/rule_baseline/{val_report.txt,test_report.txt}`.

### 2. TF–IDF + Logistic Regression
Classical sparse model that runs fast on CPU.
```bash
python run_tfidf_baseline.py              # default hyperparameters
python run_tfidf_baseline.py --grid       # optional GridSearchCV for C/ngrams
```
Artifacts:
- Model weights → `models/tfidf_logreg/tfidf_logreg.joblib`
- Metrics → `outputs/tfidf_baseline/`

### 3. DistilBERT Fine-Tuning
Full transformer baseline with Hugging Face `Trainer`.
```bash
python -m src.baselines.distilbert_baseline
```
Key outputs:
- Trained checkpoint under `models/distilbert/`
- Reports + `summary.json` under `outputs/distilbert_baseline/`
- Supports CPU, Apple MPS, or CUDA (detected automatically).

### 4. Ensemble (Majority + Meta-Learner)
Combines rule, TF–IDF, and DistilBERT predictions.
```bash
python run_ensemble.py --strategy both --threshold 0.5 --rule-confidence 1.0
```
Options:
- `--strategy`: `majority`, `meta`, or `both`.
- `--device`: force inference device for DistilBERT (`cpu`, `mps`, `cuda:0`, ...).
- `--threshold`: probability cut for voting.
Outputs land in `outputs/ensemble/` and a pickled meta model at `models/ensemble/meta_logreg.joblib`.

### 5. Model Comparison + Plots
Creates aggregate metrics, confusion matrices, and F1 plots.
```bash
python run_compare_models.py
```
Results go to `reports/` (JSON + Markdown table) and `outputs/plots/test_f1_comparison.png`.

---

## Reproducing the Paper
1. Run the **data workflow** (Steps 1–7 above) to mirror the labeled corpus used in the Prompt Injection Detector study.
2. Train the **rule**, **TF–IDF**, and **DistilBERT** baselines.
3. Execute the **ensemble** to match the paper’s reported gains.
4. Capture artifacts from `reports/` and `outputs/` for documentation or publication.

All scripts are deterministic given the same seeds (see `--seed` flags in data prep and DistilBERT).

---

## Troubleshooting & Tips
- If Hugging Face dataset downloads fail, ensure `datasets` has access to the hub (check VPN/firewall) and that your token is set.
- Transformer fine-tuning defaults to 2 epochs. Increase via `run_train(num_epochs=4, ...)` inside `src/baselines/distilbert_baseline.py` if you have GPU capacity.
- Use `nspect_results.py` to audit model mistakes (helps discover new heuristics).
- Keep an eye on `data/processed_expanded/split_summary.json` after every augmentation/merge to ensure class balance remains healthy.

---

## Citation
If you use this codebase, cite the **Prompt Injection Detector** paper and this repository. Adapt to your preferred style:

```
@article{promptinjectiondetector,
  title={Prompt Injection Detector},
  year={2024},
  note={Implementation available at https://github.com/... (replace with your repo URL)}
}
```

Happy experimenting! Let us know if you extend the detector or add new data sources.
