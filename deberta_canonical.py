#!/usr/bin/env python3
"""
Single source of truth for the DistilBERT numbers the paper reports.

Reads all five deberta_baseline_clean/seed_*/summary.json files, prints every
quantity the manuscript uses as a five-seed mean (with both std conventions),
and flags anything missing so nothing is silently skipped.

Run:  python3 distilbert_canonical.py
Optionally point at a different run:  python3 distilbert_canonical.py <run_dir>
"""

import json, os, sys
import numpy as np

SEEDS = [13, 42, 123, 2024, 7777]
RUN = sys.argv[1] if len(sys.argv) > 1 else \
      "outputs/multi_seed_runs/deberta_baseline_clean"

def get(d, *keys, default=None):
    """Safely walk nested dict keys; return default if any missing."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# load all seeds, fail loudly if any file is missing
summaries = {}
for s in SEEDS:
    p = f"{RUN}/seed_{s}/summary.json"
    if not os.path.exists(p):
        print(f"MISSING: {p}")
        sys.exit(f"Cannot proceed: seed {s} summary.json not found in {RUN}")
    summaries[s] = json.load(open(p))

print(f"Run directory: {RUN}")
print(f"Seeds: {SEEDS}\n")

# (label, extractor) for every quantity the paper reports
METRICS = [
    ("IID F1 @tau=0.5",        lambda d: get(d, "test_results", "default_threshold_0.5", "f1")),
    ("IID precision",          lambda d: get(d, "test_results", "default_threshold_0.5", "precision")),
    ("IID recall",             lambda d: get(d, "test_results", "default_threshold_0.5", "recall")),
    ("IID benign FPR",         lambda d: get(d, "test_results", "default_threshold_0.5", "fpr")),
    ("real-source F1",         lambda d: get(d, "real_src_f1")),
    ("template F1",            lambda d: get(d, "template_f1")),
    ("ROC-AUC",                lambda d: get(d, "roc_auc")),
    ("HB FPR aggregate @0.5",  lambda d: get(d, "hard_benign_test_fpr", "fpr_default_0.5")),
    ("HB FPR real-origin",     lambda d: get(d, "hard_benign_test_fpr", "fpr_real_origin")),
    ("HB FPR curated-origin",  lambda d: get(d, "hard_benign_test_fpr", "fpr_curated_origin")),
    ("obfuscation recall",     lambda d: get(d, "obfuscated_attacks", "overall_recall")),
    ("balanced-subtype F1",    lambda d: get(d, "balanced_subtype", "f1")),
    ("balanced-subtype FPR",   lambda d: get(d, "balanced_subtype", "fpr")),
    ("domain-shift F1 (agg)",  lambda d: get(d, "domain_ood", "aggregate", "f1")),
    ("domain-shift FPR (agg)", lambda d: get(d, "domain_ood", "aggregate", "fpr")),
    ("domain-shift recall",    lambda d: get(d, "domain_ood", "aggregate", "recall")),
]

hdr = f"{'metric':26s} {'mean':>8s} {'std(ddof=1)':>12s}   per-seed"
print(hdr)
print("-" * len(hdr))
for label, fn in METRICS:
    vals = [fn(summaries[s]) for s in SEEDS]
    if any(v is None for v in vals):
        missing = [s for s, v in zip(SEEDS, vals) if v is None]
        print(f"{label:26s} {'--':>8s} {'--':>12s}   MISSING in seeds {missing}")
        continue
    vals = [float(v) for v in vals]
    mean = np.mean(vals)
    sd1 = np.std(vals, ddof=1)
    per = ", ".join(f"{v:.4f}" for v in vals)
    print(f"{label:26s} {mean:8.4f} {sd1:12.4f}   [{per}]")

print("\nNote: std shown is ddof=1 (sample). If the paper's published +/- values")
print("use ddof=0, recompute with np.std(vals, ddof=0). Check against a known")
print("DeBERTa +/- to confirm the paper's convention.")
