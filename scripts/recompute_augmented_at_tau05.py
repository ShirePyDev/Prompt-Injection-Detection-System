import json, statistics as st
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

SEEDS = [13, 42, 123, 2024, 7777]
AUG = Path("outputs/multi_seed_runs/deberta_hardneg")
TEST_CSV = Path("data/pids_bench_v3/test.csv")
TAU = 0.5

# canonical source_type map from test.csv (pandas handles quoted/multiline fields)
tdf = pd.read_csv(TEST_CSV)
assert "source_type" in tdf.columns, "test.csv missing source_type"
text2st = dict(zip(tdf["text"], tdf["source_type"]))
print(f"[info] test.csv rows={len(tdf)}  source_type counts={tdf['source_type'].value_counts().to_dict()}")

def metrics_at(df_probs, df_true, tau):
    pred = (df_probs >= tau).astype(int)
    return pred

agg = {k: [] for k in ["iid_f1","iid_fpr","real_f1","tmpl_f1",
                        "hb_real","hb_cur","hb_agg","obf","dom_f1","str_f1","str_fpr"]}

for s in SEEDS:
    sd = AUG / f"seed_{s}"
    # ---- IID test ----
    test = pd.read_csv(sd / "test_predictions.csv")
    p = test["probability_score"].values; y = test["true_label"].values
    pred = (p >= TAU).astype(int)
    iid_f1 = f1_score(y, pred, zero_division=0)
    iid_fpr = ((pred==1)&(y==0)).sum() / max((y==0).sum(),1)
    # real/template via text-join to canonical source_type
    st_col = test["text"].map(text2st)
    join_cov = st_col.notna().mean()
    real_m = (st_col=="real").values; tmpl_m = (st_col=="template").values
    real_f1 = f1_score(y[real_m], pred[real_m], zero_division=0) if real_m.any() else float("nan")
    tmpl_f1 = f1_score(y[tmpl_m], pred[tmpl_m], zero_division=0) if tmpl_m.any() else float("nan")
    # ---- hard-benign (own source_type col) ----
    hb = pd.read_csv(sd / "hard_benign_predictions.csv")
    hp = hb["probability_score"].values
    hbpred = (hp >= TAU).astype(int)
    hb_agg = hbpred.mean()
    hb_real = hbpred[(hb["source_type"]=="real").values].mean()
    hb_cur  = hbpred[(hb["source_type"]=="curated").values].mean()
    # GATE: recomputed hb_agg@0.5 must match stored fpr_default_0.5
    stored = json.load(open(sd / "hard_benign_fpr.json"))["fpr_default_0.5"]
    assert abs(hb_agg - stored) < 1e-3, f"seed {s} GATE FAIL hb_agg {hb_agg:.4f} vs stored {stored:.4f}"
    # ---- obf ----
    obf = pd.read_csv(sd / "obfuscated_attacks_predictions.csv")
    obf_rec = ((obf["probability_score"]>=TAU).astype(int)[(obf["true_label"]==1).values]).mean()
    # ---- domain ----
    dom = pd.read_csv(sd / "domain_ood_predictions.csv")
    dp=(dom["probability_score"]>=TAU).astype(int); dy=dom["true_label"].values
    dom_f1=f1_score(dy,dp,zero_division=0)
    # ---- structural ----
    stc = pd.read_csv(sd / "structural_ood_probability_scores.csv")
    sp=(stc["probability_score"]>=TAU).astype(int); sy=stc["true_label"].values
    str_f1=f1_score(sy,sp,zero_division=0)
    str_fpr=((sp==1)&(sy==0)).sum()/max((sy==0).sum(),1)

    print(f"seed {s}: join_cov={join_cov:.3f} GATE ok (hb_agg {hb_agg:.4f}≈{stored:.4f})")
    for k,v in [("iid_f1",iid_f1),("iid_fpr",iid_fpr),("real_f1",real_f1),("tmpl_f1",tmpl_f1),
                ("hb_real",hb_real),("hb_cur",hb_cur),("hb_agg",hb_agg),("obf",obf_rec),
                ("dom_f1",dom_f1),("str_f1",str_f1),("str_fpr",str_fpr)]:
        agg[k].append(float(v))

print("\n"+"="*60)
print("AUGMENTED DeBERTa-v3-FT @ tau=0.5, 5-seed mean±std")
print("="*60)
for k,v in agg.items():
    m=st.mean(v); sd_=st.stdev(v) if len(v)>1 else 0.0
    print(f"{k:10s} {m:.4f} ± {sd_:.4f}")
