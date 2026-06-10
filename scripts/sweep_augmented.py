"""Recompute threshold sweeps for AUGMENTED models from existing prediction CSVs.
No retraining. Gates each seed's tau=0.5 row against stored fpr_default_0.5.
Reports whether any operating point satisfies F1>=0.95 AND real-origin FPR<=0.10.
"""
import json, csv
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

ARMS = {
    "deberta":   Path("outputs/multi_seed_runs/deberta_hardneg"),
    "distilbert":Path("outputs/multi_seed_runs/distilbert_cluster_hardneg"),
}
SEEDS = [13, 42, 123, 2024, 7777]
TAUS = [round(t,3) for t in np.arange(0.0, 1.0001, 0.001)]

for arm, root in ARMS.items():
    print("="*64); print(f"ARM: {arm}"); print("="*64)
    for s in SEEDS:
        sd = root / f"seed_{s}"
        test = pd.read_csv(sd/"test_predictions.csv")
        hb   = pd.read_csv(sd/"hard_benign_predictions.csv")
        ty, tp = test["true_label"].values, test["probability_score"].values
        hp = hb["probability_score"].values
        real_m = (hb["source_type"]=="real").values
        cur_m  = (hb["source_type"]=="curated").values

        rows=[]
        for t in TAUS:
            f1 = f1_score(ty, (tp>=t).astype(int), zero_division=0)
            r  = (hp[real_m]>=t).mean() if real_m.any() else 0.0
            c  = (hp[cur_m] >=t).mean() if cur_m.any()  else 0.0
            rows.append((t, round(f1,6), round(float(r),6), round(float(c),6)))

        # GATE: tau=0.5 hb aggregate must match stored fpr_default_0.5
        agg05 = (hp>=0.5).mean()
        stored = json.load(open(sd/"hard_benign_fpr.json"))["fpr_default_0.5"]
        gate = abs(agg05 - stored) < 1e-3

        out = sd/"threshold_sweep.csv"
        with open(out,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["tau","test_F1","hb_FPR_real","hb_FPR_curated"]); w.writerows(rows)

        # frontier check on this seed
        viol = [r for r in rows if r[1]>=0.95 and r[2]<=0.10]
        print(f"seed {s}: GATE {'ok' if gate else 'FAIL'} (agg0.5 {agg05:.4f} vs {stored:.4f}) | "
              f"rows F1>=.95 & real<=.10: {len(viol)} -> {out.name}")
print("\nIf all GATE ok and all counts are 0, the frontier persists after augmentation.")
