# inspect_results.py
"""
Print a nice summary of how each model scored (accuracy, precision, recall, F1)
on val and test splits.
"""

import json
from pathlib import Path

SUMMARY_PATH = Path("reports/model_comparison_summary.json")

def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found at {SUMMARY_PATH}. "
                                "Run run_compare_models.py first.")

    with open(SUMMARY_PATH, "r") as f:
        results = json.load(f)

    for split in ["val", "test"]:
        print(f"\n===== {split.upper()} SPLIT =====")
        print(f"{'Model':20s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")
        print("-" * 52)
        for model_name, metrics in results[split].items():
            acc = metrics["accuracy"]
            p   = metrics["precision"]
            r   = metrics["recall"]
            f1  = metrics["f1"]
            print(f"{model_name:20s}  {acc:6.3f}  {p:6.3f}  {r:6.3f}  {f1:6.3f}")
        print()

if __name__ == "__main__":
    main()
