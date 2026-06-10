"""Regenerate DistilBERT + TF-IDF BASELINE threshold sweeps with real-origin
and curated columns, matching the DeBERTa sweep schema. Gated at tau=0.5.
Writes threshold_sweep_full.csv (does NOT overwrite the 3-column originals).
"""
import json, csv
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

ARMS = {
    "distilbert": Path("outputs/multi_seed_runs/distilbert_cluster"),
    "tfidf":      Path("outputs/multi_seed_runs/tfidf"),
}
SEEDS = [13, 42, 123, 2024, 7777]
TAUS = [round(t, 3) for t in np.arange(0.0, 1.0001, 0.001)]

for arm, root in ARMS.items():
    print("="*60); print(f"ARM: {arm}"); print("="*60)
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
            agg = (hp>=t).mean()
            r = (hp[real_m]>=t).mean() if real_m.any() else 0.0
            c = (hp[cur_m] >=t).mean() if cur_m.any()  else 0.0
            rows.append((t, round(f1,6), round(float(agg),6), round(float(r),6), round(float(c),6)))

        # GATE: recomputed hb_agg@0.5 must match stored fpr_default_0.5
        agg05 = (hp>=0.5).mean()
        stored = json.load(open(sd/"hard_benign_fpr.json"))["fpr_default_0.5"]
        gate = abs(agg05 - stored) < 1e-3

        out = sd/"threshold_sweep_full.csv"
        with open(out,"w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["tau","test_F1","hb_FPR_agg","hb_FPR_real","hb_FPR_curated"])
            w.writerows(rows)

        # min externally-sourced FPR under F1>=0.95
        viol = [r for r in rows if r[1]>=0.95]
        min_real = min((r[3] for r in viol), default=None)
        peak_f1  = max(r[1] for r in rows)
        print(f"seed {s}: GATE {'ok' if gate else 'FAIL'} (agg0.5 {agg05:.4f} vs {stored:.4f}) "
              f"| peakF1={peak_f1:.4f} | min real-FPR @F1>=.95 = "
              f"{'INFEASIBLE (no tau reaches F1>=0.95)' if min_real is None else round(min_real,4)}")
print("\nDone. Full-schema sweeps written as threshold_sweep_full.csv per seed.")
