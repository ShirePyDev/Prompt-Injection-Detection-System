import json, statistics as st
from pathlib import Path

SEEDS = [13, 42, 123, 2024, 7777]
BASE = Path("outputs/multi_seed_runs/distilbert_cluster")
AUG  = Path("outputs/multi_seed_runs/distilbert_cluster_hardneg")

def ms(v): return (st.mean(v), st.stdev(v) if len(v)>1 else 0.0)
def fmt(v): m,s=ms(v); return f"{m:.4f} ± {s:.4f}"

def collect(root):
    d={k:[] for k in ["iid_f1","real","cur","agg"]}
    for s in SEEDS:
        j=json.load(open(root/f"seed_{s}"/"summary.json"))
        h=j["hard_benign_test_fpr"]
        d["iid_f1"].append(j["test_results"]["default_threshold_0.5"]["f1"])
        d["real"].append(h["fpr_real_origin"])
        d["cur"].append(h["fpr_curated_origin"])
        d["agg"].append(h["fpr_default_0.5"])
    return d

B=collect(BASE); A=collect(AUG)
print("="*60)
print("DistilBERT-FT cluster, 5-seed mean±std @ tau=0.5")
print("="*60)
print(f"IID F1     baseline {fmt(B['iid_f1'])}   aug {fmt(A['iid_f1'])}")
print(f"hb_agg     baseline {fmt(B['agg'])}   aug {fmt(A['agg'])}")
print(f"hb_real    baseline {fmt(B['real'])}   aug {fmt(A['real'])}")
print(f"hb_curated baseline {fmt(B['cur'])}   aug {fmt(A['cur'])}")
print("="*60)
import csv
with open(AUG/"distilbert_cluster_matched.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["metric","baseline_mean","baseline_std","aug_mean","aug_std"])
    for k in ["iid_f1","agg","real","cur"]:
        bm,bs=ms(B[k]); am,as_=ms(A[k])
        w.writerow([k,round(bm,4),round(bs,4),round(am,4),round(as_,4)])
print("saved -> distilbert_cluster_matched.csv")
