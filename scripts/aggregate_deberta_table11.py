import json, statistics as st
from pathlib import Path

SEEDS = [13, 42, 123, 2024, 7777]
BASE = Path("outputs/multi_seed_runs/deberta_baseline_clean")
AUG  = Path("outputs/multi_seed_runs/deberta_hardneg")

def ms(vals):  # mean, std
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)

base = {"real": [], "curated": [], "agg": [], "iid_f1": []}
aug  = {"real": [], "curated": [], "agg": []}

for s in SEEDS:
    b = json.load(open(BASE / f"seed_{s}" / "summary.json"))
    hb = b["hard_benign_test_fpr"]
    base["real"].append(hb["fpr_real_origin"])
    base["curated"].append(hb["fpr_curated_origin"])
    base["agg"].append(hb["fpr_tuned"])
    base["iid_f1"].append(b["test_results"]["val_tuned_threshold"]["f1"])

    a = json.load(open(AUG / f"seed_{s}" / "hard_benign_fpr.json"))
    aug["real"].append(a["fpr_real_origin"])
    aug["curated"].append(a["fpr_curated_origin"])
    aug["agg"].append(a["fpr_tuned"])

print("=" * 60)
print("DeBERTa-v3-FT — matched 5-seed (each at its own tuned tau)")
print("seeds:", SEEDS)
print("=" * 60)
rows = []
for metric in ["agg", "real", "curated"]:
    bm, bs = ms(base[metric]); am, as_ = ms(aug[metric])
    print(f"hb_FPR_{metric:8s}  baseline {bm:.4f} ± {bs:.4f}   ->   aug {am:.4f} ± {as_:.4f}")
    rows.append({"metric": metric, "baseline_mean": round(bm,4), "baseline_std": round(bs,4),
                 "aug_mean": round(am,4), "aug_std": round(as_,4)})
im, is_ = ms(base["iid_f1"])
print(f"IID_F1 (baseline)   {im:.4f} ± {is_:.4f}")
print("=" * 60)

import csv
with open(AUG / "table11_deberta_matched.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["metric","baseline_mean","baseline_std","aug_mean","aug_std"])
    w.writeheader(); w.writerows(rows)
print(f"saved -> {AUG / 'table11_deberta_matched.csv'}")
