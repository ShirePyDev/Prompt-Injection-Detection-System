#!/usr/bin/env python3
"""Build Table 11 (hard-negative ablation) from 5-seed cluster runs at tau=0.5.
Both baseline and +HardNeg arms: DeBERTa-v3-FT, 5 seeds, fixed threshold 0.5.
Supersedes analyze_hardneg_table11.py (single-seed canonical / tuned threshold).
"""
import json, statistics as st
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

SEEDS = [13, 42, 123, 2024, 7777]
ARMS = {
    "DeBERTa-v3-FT baseline":  "outputs/multi_seed_runs/deberta_baseline_clean",
    "DeBERTa-v3-FT +HardNeg":  "outputs/multi_seed_runs/deberta_hardneg",
}

def iid_at_05(d):
    tp = pd.read_csv(f"{d}/test_predictions.csv")
    y, p = tp["true_label"].values, tp["probability_score"].values
    pred = (p >= 0.5).astype(int)
    return {
        "IID_F1": f1_score(y, pred),
        "IID_precision": precision_score(y, pred),
        "IID_recall": recall_score(y, pred),
        "test_FPR": ((pred==1)&(y==0)).sum()/max((y==0).sum(),1),
        "ROC_AUC": roc_auc_score(y, p),
    }

def metrics_for(seed_dir):
    hb  = json.load(open(f"{seed_dir}/hard_benign_fpr.json"))
    dom = json.load(open(f"{seed_dir}/domain_ood_report.json"))
    stru= json.load(open(f"{seed_dir}/structural_ood_report.json"))
    obf = json.load(open(f"{seed_dir}/obfuscated_attacks_report.json"))
    m = iid_at_05(seed_dir)
    m["hb_FPR_agg"]     = hb["fpr_default_0.5"]
    m["hb_FPR_extsrc"]  = hb["fpr_real_origin"]      # 'externally-sourced' in paper
    m["hb_FPR_curated"] = hb["fpr_curated_origin"]
    # domain/struct F1 — pull aggregate f1 from report (adjust key if needed)
    m["domain_F1"] = dom.get("aggregate",{}).get("f1") or dom.get("f1")
    m["struct_F1"] = stru.get("aggregate",{}).get("f1") or stru.get("f1")
    m["obf_recall"]= obf.get("overall_recall") or obf.get("aggregate",{}).get("recall")
    return m

rows = []
for arm, base in ARMS.items():
    per = {k: [] for k in ["IID_F1","IID_precision","IID_recall","test_FPR","ROC_AUC",
                            "hb_FPR_agg","hb_FPR_extsrc","hb_FPR_curated",
                            "domain_F1","struct_F1","obf_recall"]}
    for s in SEEDS:
        m = metrics_for(f"{base}/seed_{s}")
        for k in per: per[k].append(m[k])
    for k in per:
        vals = [v for v in per[k] if v is not None]
        mean = st.mean(vals); sd = st.stdev(vals) if len(vals)>1 else 0.0
        rows.append({"arm":arm,"metric":k,"n":len(vals),
                     "mean":round(mean,4),"std":round(sd,4),
                     "formatted":f"{mean:.4f} ± {sd:.4f}"})

out = pd.DataFrame(rows)
out.to_csv("outputs/multi_seed_runs/table11_5seed_tau05.csv", index=False)
print(out.to_string(index=False))
print("\nSaved: outputs/multi_seed_runs/table11_5seed_tau05.csv")
