# run_expand_datasets.py
"""
Download and merge multiple prompt injection datasets into one expanded CSV.
Run this from VS Code (▶).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.expand_datasets import main  # noqa: E402

if __name__ == "__main__":
    main()
