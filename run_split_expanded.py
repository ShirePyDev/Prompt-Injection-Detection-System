# run_split_expanded.py
"""
Create new train/val/test splits from the expanded dataset (all_prompts.csv).
Run this once after expand_datasets.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.split_expanded import main  # noqa: E402  #type: ignore
 
if __name__ == "__main__":
    main()
