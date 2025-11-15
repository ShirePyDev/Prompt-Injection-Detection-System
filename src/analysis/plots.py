# src/analysis/plots.py
from pathlib import Path
import matplotlib.pyplot as plt
import json

OUT_DIR = Path("outputs/analysis_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Manually enter your metrics (from TF-IDF, DistilBERT, Ensemble)
metrics = {
    "tfidf": {
        "accuracy": 0.9241,
        "f1": 0.9526
    },
    "distilbert": {
        "accuracy": 0.94,      # replace with your actual number
        "f1": 0.96             # replace with your actual number
    },
    "ensemble": {
        "accuracy": 0.9617,
        "f1": 0.9771
    }
}

def bar_plot(metric_name):
    names = list(metrics.keys())
    values = [metrics[m][metric_name] for m in names]

    plt.figure(figsize=(6, 4))
    plt.bar(names, values)
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} Comparison")

    # 🔍 Zoom the y-axis to highlight small differences
    plt.ylim(0.85, 1.0)

    plt.savefig(OUT_DIR / f"{metric_name}_comparison.png")
    plt.close()


def main():
    bar_plot("accuracy")
    bar_plot("f1")
    print("✅ Plots saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
