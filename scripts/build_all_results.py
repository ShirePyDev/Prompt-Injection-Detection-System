#!/usr/bin/env python3
"""Single source of truth for all PIDS-Bench result tables.
All hard-benign per-source FPR computed by joining {arm}/seed_*/hard_benign_predictions.csv
to data/pids_bench_v3/eval_subsets/hard_benign_test.csv on text (canonical source_type).
All metrics at tau=0.5, 5-seed mean +/- sample std (n-1).
Supersedes: analyze_hardneg_table11.py, bplus_*.csv per-source values.
"""
import json, statistics as st
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

SEEDS = [13,42,123,2024,7777]
REF = pd.read_csv("data/pids_bench_v3/eval_subsets/hard_benign_test.csv")[["text","source_type"]]

def hb_per_source(seed_dir):
    p = pd.read_csv(f"{seed_dir}/hard_benign_predictions.csv")
    if "source_type" in p.columns: p = p.drop(columns=["source_type"])
    m = p.merge(REF, on="text", how="left")
    assert m["source_type"].notna().all(), f"coverage<1 in {seed_dir}"
    flag = (m["probability_score"].values >= 0.5).astype(int)
    ie = m["source_type"].values=="real"; ic = m["source_type"].values=="curated"
    return flag.mean(), flag[ie].mean(), flag[ic].mean()

def iid(seed_dir):
    tp = pd.read_csv(f"{seed_dir}/test_predictions.csv")
    y,p = tp["true_label"].values, tp["probability_score"].values
    pr = (p>=0.5).astype(int)
    return f1_score(y,pr), precision_score(y,pr), recall_score(y,pr), \
           ((pr==1)&(y==0)).sum()/max((y==0).sum(),1), roc_auc_score(y,p)

def ms(vals):
    return f"{st.mean(vals):.4f} ± {st.stdev(vals):.4f}"

rows=[]
for arm,base in [("Baseline","outputs/multi_seed_runs/deberta_baseline_clean"),
                 ("+HardNeg","outputs/multi_seed_runs/deberta_hardneg")]:
    agg=[];ext=[];cur=[];f1=[];pr=[];rc=[];fp=[];roc=[];dom=[];stru=[];obf=[]
    for s in SEEDS:
        d=f"{base}/seed_{s}"
        a,e,c=hb_per_source(d); agg.append(a);ext.append(e);cur.append(c)
        F,P,R,FP,RC=iid(d); f1.append(F);pr.append(P);rc.append(R);fp.append(FP);roc.append(RC)
        dom.append(json.load(open(f"{d}/domain_ood_report.json"))["aggregate"]["f1"])
        stru.append(json.load(open(f"{d}/structural_ood_report.json"))["aggregate"]["f1"])
        obf.append(json.load(open(f"{d}/obfuscated_attacks_report.json"))["overall_recall"])
    rows.append({"arm":arm,"IID_F1":ms(f1),"IID_prec":ms(pr),"IID_rec":ms(rc),
        "test_FPR":ms(fp),"ROC":ms(roc),"hb_agg":ms(agg),"hb_extsrc":ms(ext),
        "hb_curated":ms(cur),"domain_F1":ms(dom),"struct_F1":ms(stru),"obf_recall":ms(obf)})

df=pd.DataFrame(rows)
df.to_csv("outputs/multi_seed_runs/table11_CANONICAL.csv", index=False)
print(df.to_string(index=False))
print("\nSaved: outputs/multi_seed_runs/table11_CANONICAL.csv")
