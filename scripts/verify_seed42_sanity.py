"""
scripts/verify_seed42_sanity.py

Retrains the DeBERTa-v3-FT baseline at seed 42 under the current runtime
environment (intended for the A100 session) and compares against the
CANONICAL paper-model reference values.

SOURCE OF TRUTH
---------------
Reference values are read from two files produced by the original March 30
training run on Colab CUDA:

  outputs/deberta_v3/summary.json
    • val_tuned_threshold : 0.502
    • test F1             : 0.9889  (test_results.val_tuned_threshold.f1)
    • test FPR            : 0.0119  (test_results.val_tuned_threshold.fpr)

  outputs/deberta_v3/hard_benign_fpr.json
    • hb_FPR_agg          : 0.349185  (fpr_tuned, matching paper Table 4 = 0.3492)
    • hb_FPR_real_origin  : 0.288991  (matching paper Table 4 = 0.2890)

These values were confirmed on 2026-05-20 by verifying the canonical
prediction files (hard_benign_predictions.csv + hard_benign_test.csv)
SHA-256 checksums against FREEZE_MANIFEST.txt.

NOTE: the multi-seed all_runs.csv seed_42 row (hb_FPR_agg=0.409,
tau=0.822) is from a SEPARATE later training run and does NOT represent
the canonical paper model. Do not use all_runs.csv as a reference here.

TOLERANCES (user-specified)
---------------------------
  IID_F1       ±0.002
  test_FPR     ±0.002

Saves pip freeze to:
  outputs/multi_seed_runs/deberta_hardneg/pip_freeze_a100.txt

Exit 0 = all passed → proceed with deberta_hardneg runs.
Exit 1 = at least one metric failed → investigate before continuing.

Usage (from repo root, on A100):
    python scripts/verify_seed42_sanity.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

SEED         = 42
CANONICAL_SUMMARY = _ROOT / "outputs" / "deberta_v3" / "summary.json"
CANONICAL_HB_FPR  = _ROOT / "outputs" / "deberta_v3" / "hard_benign_fpr.json"
SANITY_OUT        = _ROOT / "outputs" / "multi_seed_runs" / "sanity_check_seed42"
PIP_FREEZE        = _ROOT / "outputs" / "multi_seed_runs" / "deberta_hardneg" / "pip_freeze_a100.txt"

TOLERANCES: dict[str, float] = {
    "IID_F1":      0.002,
    "hb_FPR_agg":  0.005,
    "hb_FPR_real": 0.005,
    "test_FPR":    0.002,
}



def _load_reference() -> dict[str, float]:
    if not CANONICAL_SUMMARY.exists():
        raise FileNotFoundError(
            f"Canonical summary not found: {CANONICAL_SUMMARY}\n"
            "Ensure outputs/deberta_v3/ is present (the March 30 canonical run)."
        )
    s  = json.load(open(CANONICAL_SUMMARY))
    hb = json.load(open(CANONICAL_HB_FPR)) if CANONICAL_HB_FPR.exists() else {}

    tr = s.get("test_results", {}).get("val_tuned_threshold", {})
    return {
        "IID_F1":      tr.get("f1"),
        "test_FPR":    tr.get("fpr"),
        "hb_FPR_agg":  hb.get("fpr_tuned"),
        "hb_FPR_real": hb.get("fpr_real_origin"),
    }


def _save_pip_freeze() -> None:
    PIP_FREEZE.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=True,
    )
    PIP_FREEZE.write_text(result.stdout)
    print(f"  pip freeze saved → {PIP_FREEZE}")


def main() -> None:
    print("=" * 70)
    print(f"DeBERTa-v3-FT  Seed-42 Sanity Check  (repo: {_ROOT})")
    print(f"Reference: {CANONICAL_SUMMARY}")
    print("=" * 70)

    # Save pip freeze before training so it's available even if training fails
    print("\n[Step 1] Saving pip freeze ...")
    _save_pip_freeze()

    # Load canonical reference values
    print(f"\n[Step 2] Loading canonical reference values ...")
    recorded = _load_reference()
    for m, v in recorded.items():
        tol = TOLERANCES.get(m, "?")
        print(f"  {m:<20}  {v:.6f}  (tol=±{tol})")

    if torch.cuda.is_available():
        batch, accum = 8, 2
    elif torch.backends.mps.is_available():
        batch, accum = 2, 8
    else:
        batch, accum = 4, 4

    print(f"\n[Step 3] Retraining DeBERTa-v3-FT  seed={SEED}"
          f"  batch={batch}  accum={accum}  →  {SANITY_OUT}")
    SANITY_OUT.mkdir(parents=True, exist_ok=True)

    from src.baselines.deberta_v3 import run_train
    metrics = run_train(
        num_epochs=3,
        batch_size=batch,
        lr=2e-5,
        seed=SEED,
        gradient_accumulation_steps=accum,
        out_dir=SANITY_OUT,
    )

    print("\n" + "=" * 70)
    print("COMPARISON — canonical reference vs reproduced")
    print("=" * 70)
    print(f"{'Metric':<20}  {'Reference':>10}  {'Reproduced':>10}  {'|Diff|':>8}  {'Tol':>7}  Result")
    print("-" * 70)

    passed = True
    for m, tol in TOLERANCES.items():
        ref = recorded.get(m)
        rep = metrics.get(m)
        if ref is None:
            print(f"{m:<20}  {'N/A ref':>10}  {'':>10}  {'':>8}  {tol:>7}  SKIP")
            continue
        if rep is None:
            print(f"{m:<20}  {ref:>10.6f}  {'MISSING':>10}  {'':>8}  {tol:>7}  FAIL")
            passed = False
            continue
        diff = abs(float(rep) - ref)
        ok   = diff <= tol
        if not ok and m in ("IID_F1", "test_FPR"):
            passed = False
        print(f"{m:<20}  {ref:>10.6f}  {float(rep):>10.6f}  {diff:>8.6f}  ±{tol:.3f}  {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 70)
    if passed:
        print("RESULT: ALL PASSED")
        print("  Environment reproduces canonical values. Proceed with augmented runs:")
        print("    python scripts/train_multi_seed.py --models deberta_hardneg")
        sys.exit(0)
    else:
        print("RESULT: FAILED — environment divergence detected.")
        print("  Do NOT proceed with hardneg runs until resolved.")
        print(f"  Check pip freeze at {PIP_FREEZE}")
        print("  against the versions used for the canonical March 30 Colab CUDA run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
