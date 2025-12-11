import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ============================================
# 1. DistilBERT confusion matrix (test split)
# ============================================

# Confusion Matrix:
# [[TN, FP],
#  [FN, TP]]
cm = np.array([
    [1376, 34],
    [47,  2368]
])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["benign", "injection"])
disp.plot(colorbar=False)
plt.title("Confusion Matrix (DistilBERT, Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix.pdf")
plt.close()
print("Saved confusion_matrix.pdf")

# ============================================
# 2. Model comparison bar chart (F1 scores)
# ============================================

models = ["Rule-based", "TF-IDF + LR", "DistilBERT", "Ensemble"]
f1_scores = [0.2753, 0.9586, 0.9836, 0.9769]

plt.figure(figsize=(6, 4))
plt.bar(models, f1_scores)
plt.ylabel("F1 Score")
plt.ylim(0.0, 1.0)
plt.title("Model Comparison on Test Set")
plt.tight_layout()
plt.savefig("model_comparison.pdf")
plt.close()
print("Saved model_comparison.pdf")
