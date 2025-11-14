# run_compare_models.py
"""
Run final comparison of rule, TF-IDF, DistilBERT, and ensemble.
Generates JSON, markdown table, and F1 bar plot.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.compare_models import run  # noqa: E402

if __name__ == "__main__":
    run()
