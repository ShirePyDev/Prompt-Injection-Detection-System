# run_tfidf_baseline.py
"""
Run the TF-IDF + Logistic Regression baseline from VS Code (no shell).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.tfidf_logreg import run  # noqa: E402

if __name__ == "__main__":
    # Set to True to enable a small GridSearchCV
    run(grid_search=False)
