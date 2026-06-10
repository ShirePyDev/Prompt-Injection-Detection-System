import json, statistics as st
from pathlib import Path

SEEDS = [13, 42, 123, 2024, 7777]
BASE = Path("outputs/multi_seed_runs/deberta_baseline_clean")
AUG  = Path("outputs/multi_seed_runs/deberta_hardneg")

def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)

def fmt(v):
    m, s = ms(v); return f"{m:.4f} ± {s:.4f}"

# ---- collect baseline (full summary.json) ----
B = {k: [] for k in ["f1_05","fpr_05","f1_t","fpr_t","tau","real_f1","tmpl_f1","roc",
                      "hb_agg_05","hb_agg_t","hb_real_t","hb_cur_t",
                      "obf","dom_f1","str_f1","str_fpr"]}
for s in SEEDS:
    d = json.load(open(BASE / f"seed_{s}" / "summary.json"))
    t05 = d["test_results"]["default_threshold_0.5"]
    tt  = d["test_results"]["val_tuned_threshold"]
    hb  = d["hard_benign_test_fpr"]
    B["f1_05"].append(t05["f1"]);   B["fpr_05"].append(t05["fpr"])
    B["f1_t"].append(tt["f1"]);     B["fpr_t"].append(tt["fpr"])
    B["tau"].append(d["val_tuned_threshold"])
    B["real_f1"].append(d["real_src_f1"]); B["tmpl_f1"].append(d["template_f1"])
    B["roc"].append(d["roc_auc"])
    B["hb_agg_05"].append(hb["fpr_default_0.5"]); B["hb_agg_t"].append(hb["fpr_tuned"])
    B["hb_real_t"].append(hb["fpr_real_origin"]); B["hb_cur_t"].append(hb["fpr_curated_origin"])
    B["obf"].append(d["obfuscated_attacks"]["overall_recall"])
    B["dom_f1"].append(d["domain_ood"]["aggregate"]["f1"])
    B["str_f1"].append(d["structural_ood"]["aggregate"]["f1"])
    B["str_fpr"].append(d["structural_ood"]["aggregate"]["fpr"])

# ---- collect augmented (hard_benign_fpr.json only — that's what's on disk) ----
A = {k: [] for k in ["hb_agg_05","hb_agg_t","hb_real_t","hb_cur_t","tau"]}
for s in SEEDS:
    h = json.load(open(AUG / f"seed_{s}" / "hard_benign_fpr.json"))
    A["hb_agg_05"].append(h["fpr_default_0.5"]); A["hb_agg_t"].append(h["fpr_tuned"])
    A["hb_real_t"].append(h["fpr_real_origin"]); A["hb_cur_t"].append(h["fpr_curated_origin"])
    A["tau"].append(h["tau"])

print("="*70)
print("B+ AGGREGATION — DeBERTa-v3-FT, 5 seeds (cluster, pinned env)")
print("seeds:", SEEDS)
print("="*70)
print("\n--- BASELINE: main-table metrics ---")
print(f"IID F1   (tau=0.5)        {fmt(B['f1_05'])}")
print(f"IID F1   (tuned tau)      {fmt(B['f1_t'])}")
print(f"IID FPR  (tau=0.5)        {fmt(B['fpr_05'])}")
print(f"real-src F1               {fmt(B['real_f1'])}")
print(f"template F1               {fmt(B['tmpl_f1'])}")
print(f"ROC-AUC                   {fmt(B['roc'])}")
print(f"Obf recall                {fmt(B['obf'])}")
print(f"Domain-OOD F1             {fmt(B['dom_f1'])}")
print(f"Structural F1             {fmt(B['str_f1'])}")
print(f"Structural FPR            {fmt(B['str_fpr'])}")
print(f"tuned tau (variance!)     {fmt(B['tau'])}   range [{min(B['tau'])}, {max(B['tau'])}]")

print("\n--- HARD-BENIGN: baseline vs augmented, FIXED tau=0.5 (primary, comparable) ---")
print(f"hb_agg     baseline {fmt(B['hb_agg_05'])}   ->  aug {fmt(A['hb_agg_05'])}")
print("\n--- HARD-BENIGN: baseline vs augmented, each at own tuned tau (secondary) ---")
print(f"hb_agg     baseline {fmt(B['hb_agg_t'])}   ->  aug {fmt(A['hb_agg_t'])}")
print(f"hb_real    baseline {fmt(B['hb_real_t'])}   ->  aug {fmt(A['hb_real_t'])}")
print(f"hb_curated baseline {fmt(B['hb_cur_t'])}   ->  aug {fmt(A['hb_cur_t'])}")
print(f"aug tuned tau range [{min(A['tau'])}, {max(A['tau'])}]")
print("="*70)

# ---- save ----
import csv
def rows(d): return [{"metric": k, "mean": round(ms(v)[0],4), "std": round(ms(v)[1],4)} for k,v in d.items()]
with open(AUG / "bplus_baseline.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["metric","mean","std"]); w.writeheader(); w.writerows(rows(B))
with open(AUG / "bplus_augmented.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["metric","mean","std"]); w.writeheader(); w.writerows(rows(A))
print("saved -> bplus_baseline.csv , bplus_augmented.csv")
