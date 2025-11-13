# run_rule_baseline.py (at project root)
"""
Convenience launcher for the rule-based baseline so you can click 'Run' in VS Code.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.rule_based import run  # noqa: E402

if __name__ == "__main__":
    run()
