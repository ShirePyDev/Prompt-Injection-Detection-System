# run_ensemble.py
"""
Run ensemble fusion (majority vote + meta-learner) from VS Code.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ensemble.ensemble import run  # noqa: E402

if __name__ == "__main__":
    run()
