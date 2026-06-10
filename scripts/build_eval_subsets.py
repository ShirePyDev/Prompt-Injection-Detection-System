#!/usr/bin/env python3
"""
Build evaluation subsets from existing train/val/test.

Use this when you already have train.csv, val.csv, test.csv and want to
(re)build balanced_subtype_test and obfuscated_attacks without running
the full build pipeline.

For full v3 build, use: python -m data_builder.build_dataset --config config_v3.yaml
"""
from pathlib import Path

from data_builder.build_eval_subsets import run

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build eval subsets from existing splits.")
    ap.add_argument("--data_dir", type=Path, default=Path("data/pids_bench_v3"))
    args = ap.parse_args()
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / data_dir
    run(data_dir)
