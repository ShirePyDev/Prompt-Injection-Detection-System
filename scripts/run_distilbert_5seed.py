"""Retrain DistilBERT on the cluster (CUDA), 5 seeds x 2 arms, isolated per-seed dirs.
Uses DistilBERT's OWN config (2 epochs, batch 16, lr 5e-5) — environment-only change.
Does NOT touch existing MPS runs in outputs/multi_seed_runs/distilbert{,_hardneg}/.
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, ".")
from src.baselines.distilbert_baseline import run_train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hard-neg", action="store_true")
    args = ap.parse_args()

    arm = "distilbert_cluster_hardneg" if args.hard_neg else "distilbert_cluster"
    out_dir = Path("outputs/multi_seed_runs") / arm / f"seed_{args.seed}"
    print(f"\n=== DistilBERT {'+HardNeg' if args.hard_neg else 'baseline'} seed={args.seed} -> {out_dir} ===")
    run_train(num_epochs=2, batch_size=16, lr=5e-5, seed=args.seed,
              hard_neg=args.hard_neg, out_dir=out_dir)
    print(f"=== done seed={args.seed} ===")

if __name__ == "__main__":
    main()
