from pathlib import Path
from src.baselines.deberta_v3 import run_train

SEEDS = [13, 42, 123, 2024, 7777]
for seed in SEEDS:
    out = Path(f"outputs/multi_seed_runs/deberta_baseline_clean/seed_{seed}")
    print(f"\n===== BASELINE seed {seed} -> {out} =====")
    run_train(
        num_epochs=3,
        batch_size=8,                      # match augmented device batch
        gradient_accumulation_steps=2,     # match augmented accum (effective 16)
        lr=2e-5,
        seed=seed,
        out_dir=out,
    )
