
#!/usr/bin/env python3
"""
Operating-point and base-rate analysis for the internal detectors (Reviewer 3 #3).

R3 asked us to (a) name one primary operating policy and treat the others as
sensitivity analyses, (b) report threshold-free ranking metrics suited to the
low-FPR regime (PR-AUC, partial ROC-AUC), (c) report recall at operationally
meaningful FPR levels, (d) report expected precision under plausible benign-to-
attack base rates, and (e) justify the 0.10 hard-benign FPR target.

PRIMARY OPERATING POLICY
  validation-F1-tuned threshold. This is the only policy that is a genuine
  deployment rule: tau=0.5 is just the point where cross-seed averaging is
  comparable, and the constrained rule (Table 9) is selected on TEST F1 and is
  therefore diagnostic, not deployable. Metrics at the validation-tuned threshold
  are reported as primary; tau=0.5 is the sensitivity comparison.

NO-LEAKAGE DISCIPLINE
  For "recall at fixed FPR", the threshold that achieves a target FPR is chosen on
  the VALIDATION split and recall is then measured on TEST. Choosing the threshold
  on test would be circular. The validation split is the IID distribution and
  contains no hard-benign rows, so a leakage-free "recall at fixed FPR" is only
  honestly definable for IID-benign FPR. For the hard-benign side we therefore
  report the DESCRIPTIVE operating triple (recall, IID-FPR, hard-benign-FPR) at the
  validation-tuned threshold, not a selected-on-hard-benign number we cannot
  compute without leakage.

All inputs verified to reconcile with the paper before reporting.

OUTPUT
  outputs/resub_analysis/operating_point_analysis.csv

USAGE
  python scripts/operating_point_analysis.py
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO = Path("/home/M2026206/khalid/Prompt-Injection-Detector-System")
OUT_DIR = REPO / "outputs" / "resub_analysis"
HB_FILE = REPO / "data" / "pids_bench_v3" / "eval_subsets" / "hard_benign_test.csv"
SEEDS = ["13", "42", "123", "2024", "7777"]

MODELS = {
    "DeBERTa-v3-FT": "outputs/multi_seed_runs/deberta_baseline_clean",
    "DistilBERT":    "outputs/multi_seed_runs/distilbert",
}
IID_FPR_TARGETS = [0.01, 0.02, 0.05]
HB_FPR_TARGETS = [0.05, 0.10, 0.20]          # for the 0.10-target justification
BASE_RATES = [(90, 10), (99, 1), (999, 1)]    # benign:injection in deployment


def threshold_for_fpr(val_scores_benign: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose benign-flag rate on validation is <= target_fpr."""
    # sweep candidate thresholds; pick the lowest tau meeting the FPR cap
    taus = np.linspace(0, 1, 1001)
    for t in taus:
        if (val_scores_benign >= t).mean() <= target_fpr:
            return float(t)
    return 1.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hb = pd.read_csv(HB_FILE)
    hb_key = hb[["text", "source_type"]]
    rows = []

    for model, root in MODELS.items():
        print(f"\n{'='*72}\n{model}\n{'='*72}")
        # load per-seed val, test, hard-benign + the val-tuned threshold
        seed_data = []
        for s in SEEDS:
            sd = REPO / root / f"seed_{s}"
            val = pd.read_csv(sd / "val_predictions.csv")
            test = pd.read_csv(sd / "test_predictions.csv")
            hbp = pd.read_csv(sd / "hard_benign_predictions.csv")
            tau = json.load(open(sd / "summary.json"))["val_tuned_threshold"]
            seed_data.append((val, test, hbp, float(tau)))

        # ---- (1) PRIMARY POLICY: validation-tuned threshold ----
        rec, iidfpr, hbfpr = [], [], []
        for val, test, hbp, tau in seed_data:
            yt = test["true_label"].values
            pt = (test["probability_score"].values >= tau).astype(int)
            rec.append(((yt == 1) & (pt == 1)).sum() / (yt == 1).sum())
            iidfpr.append(((yt == 0) & (pt == 1)).sum() / (yt == 0).sum())
            hbfpr.append((hbp["probability_score"].values >= tau).mean())
        print(f"\n  PRIMARY (validation-tuned threshold), 5-seed mean +/- std:")
        print(f"    recall          {np.mean(rec):.4f} +/- {np.std(rec, ddof=1):.4f}")
        print(f"    IID benign FPR  {np.mean(iidfpr):.4f} +/- {np.std(iidfpr, ddof=1):.4f}")
        print(f"    hard-benign FPR {np.mean(hbfpr):.4f} +/- {np.std(hbfpr, ddof=1):.4f}")
        rows.append({"model": model, "analysis": "primary_val_tuned",
                     "recall": np.mean(rec), "iid_fpr": np.mean(iidfpr),
                     "hb_fpr": np.mean(hbfpr)})

        # ---- (2) threshold-free ranking metrics ----
        prauc, partial_auc = [], []
        for val, test, hbp, tau in seed_data:
            y = test["true_label"].values
            p = test["probability_score"].values
            prauc.append(average_precision_score(y, p))
            # partial ROC-AUC over FPR in [0, 0.1], normalised to [0,1]
            fpr, tpr, _ = roc_curve(y, p)
            mask = fpr <= 0.10
            if mask.sum() >= 2:
                pa = np.trapz(tpr[mask], fpr[mask]) / 0.10
            else:
                pa = np.nan
            partial_auc.append(pa)
        print(f"\n  Threshold-free ranking (test):")
        print(f"    PR-AUC                {np.mean(prauc):.4f} +/- {np.std(prauc, ddof=1):.4f}")
        print(f"    partial ROC-AUC@FPR.10 {np.nanmean(partial_auc):.4f} "
              f"+/- {np.nanstd(partial_auc, ddof=1):.4f}")
        rows.append({"model": model, "analysis": "ranking",
                     "pr_auc": np.mean(prauc), "partial_auc": np.nanmean(partial_auc)})

        # ---- (3) recall at fixed IID-benign FPR (threshold chosen on VAL) ----
        print(f"\n  Recall at fixed IID-benign FPR (threshold selected on validation):")
        for tgt in IID_FPR_TARGETS:
            recs = []
            for val, test, hbp, tau in seed_data:
                vb = val.loc[val["true_label"] == 0, "probability_score"].values
                t = threshold_for_fpr(vb, tgt)
                yt = test["true_label"].values
                pt = (test["probability_score"].values >= t).astype(int)
                recs.append(((yt == 1) & (pt == 1)).sum() / (yt == 1).sum())
            print(f"    FPR<= {tgt:>5.0%} : recall = {np.mean(recs):.4f} "
                  f"+/- {np.std(recs, ddof=1):.4f}")
            rows.append({"model": model, "analysis": f"recall_at_iidfpr_{tgt}",
                         "recall": np.mean(recs)})

        # ---- (4) expected precision under deployment base rates ----
        # precision = (br_inj * recall) / (br_inj * recall + br_ben * iid_fpr)
        R = np.mean(rec); F = np.mean(iidfpr)   # at the primary operating point
        print(f"\n  Expected precision at primary policy under base rates"
              f" (recall={R:.3f}, IID-FPR={F:.3f}):")
        for ben, inj in BASE_RATES:
            pi = inj / (ben + inj)
            prec = (pi * R) / (pi * R + (1 - pi) * F) if (pi * R + (1 - pi) * F) else 0.0
            print(f"    benign:injection = {ben}:{inj:<4}  expected precision = {prec:.4f}")
            rows.append({"model": model, "analysis": f"precision_base_{ben}_{inj}",
                         "precision": prec})

        # ---- (5) 0.10-target justification: is the target reachable at all? ----
        # Descriptive frontier (threshold selected on the hard-benign set itself,
        # so NOT leakage-free -- labelled as such). On a fine grid we find the
        # best test-recall whose hard-benign FPR <= target, but we EXCLUDE the
        # degenerate tau where the detector flags nothing (recall ~ 0), because
        # "0% FPR by flagging nothing" is not an operating point. If the only way
        # to meet a target is that degenerate region, we report the target as
        # effectively unreachable and give the recall at the lowest NON-degenerate
        # feasible threshold.
        print(f"\n  Descriptive frontier (NOT leakage-free; threshold chosen on the")
        print(f"  hard-benign set). Best test-recall with hard-benign FPR<=target,")
        print(f"  excluding the degenerate flag-nothing region (recall<0.50):")
        MIN_USABLE_RECALL = 0.50
        for tgt in HB_FPR_TARGETS:
            best, min_hbfpr_seen = [], []
            for val, test, hbp, tau in seed_data:
                taus = np.linspace(0, 1, 10001)        # fine grid
                yt = test["true_label"].values
                sp = test["probability_score"].values
                hp = hbp["probability_score"].values
                feasible = []
                for t in taus:
                    pt = (sp >= t)
                    r = (pt[yt == 1]).sum() / (yt == 1).sum()
                    if r < MIN_USABLE_RECALL:           # skip degenerate region
                        continue
                    hbf = (hp >= t).mean()
                    if hbf <= tgt:
                        feasible.append(r)
                # also record the minimum hard-benign FPR reachable while keeping
                # usable recall, to characterise reachability
                usable = [( (sp>=t)[yt==1].sum()/(yt==1).sum(), (hp>=t).mean() )
                          for t in taus if (sp>=t)[yt==1].sum()/(yt==1).sum() >= MIN_USABLE_RECALL]
                min_hbfpr_seen.append(min(h for _, h in usable) if usable else float("nan"))
                best.append(max(feasible) if feasible else float("nan"))
            n_reach = sum(1 for b in best if not np.isnan(b))
            if n_reach == 0:
                mn = np.nanmean(min_hbfpr_seen)
                print(f"    HB-FPR<= {tgt:>4.0%} : UNREACHABLE at any usable threshold "
                      f"(lowest hard-benign FPR with recall>={MIN_USABLE_RECALL:.0%} "
                      f"is {mn:.4f}, 5-seed mean)")
                rows.append({"model": model, "analysis": f"maxrecall_at_hbfpr_{tgt}",
                             "recall": float("nan"), "reachable": 0,
                             "min_hbfpr_usable": mn})
            else:
                mr = np.nanmean(best)
                print(f"    HB-FPR<= {tgt:>4.0%} : reachable in {n_reach}/5 seeds, "
                      f"max recall = {mr:.4f}")
                rows.append({"model": model, "analysis": f"maxrecall_at_hbfpr_{tgt}",
                             "recall": mr, "reachable": n_reach})

    pd.DataFrame(rows).to_csv(OUT_DIR / "operating_point_analysis.csv", index=False)
    print(f"\nWrote: {OUT_DIR/'operating_point_analysis.csv'}")


if __name__ == "__main__":
    main()
