# PIDS-Bench v3 — Benchmark report

**Authoritative row counts and checksums:** `data/pids_bench_v3/FREEZE_MANIFEST.txt` (frozen **2026-03-24**; obfuscated subset rebuilt **2026-03-20** — see manifest NOTE).

---

## 1. Overview

| Split | Rows |
|-------|------|
| Train | 27,093 |
| Val   | 3,989 |
| Test  | 3,918 |
| **Main corpus total** | **35,000** |
| Eval: hard_benign_test | 1,472 |
| Eval: balanced_subtype_test | 2,297 |
| Eval: obfuscated_attacks | **405** |
| OOD: domain_ood | 2,000 |
| OOD: structural_ood | 1,998 |
| **Grand total (all frozen sets in manifest)** | **43,172** |

---

## 2. Class balance

| Split | Benign (0) | Injection (1) | Total |
|-------|-----------|--------------|-------|
| Train | 12,846 (47.4%) | 14,247 (52.6%) | 27,093 |
| Val   | 1,943 (48.7%)  | 2,046 (51.3%)  | 3,989 |
| Test  | 1,848 (47.2%)  | 2,070 (52.8%)  | 3,918 |

---

## 3. Evaluation subsets

| Set | Rows | Benign | Injection | Purpose |
|-----|------|--------|-----------|---------|
| hard_benign_test | 1,472 | 1,472 | 0 | Over-defense (FPR) |
| balanced_subtype_test | 2,297 | 1,200 | 1,097 | Subtype fairness |
| obfuscated_attacks | **405** | 0 | **405** | Obfuscation robustness (cleaned; no train leakage — see FREEZE_MANIFEST NOTE) |
| domain_ood | 2,000 | 1,000 | 1,000 | Domain distribution shift |
| structural_ood | 1,998 | 999 | 999 | Structural distribution shift |

---

## 4. Freeze checksums

SHA-256 values below match **`data/pids_bench_v3/FREEZE_MANIFEST.txt`**.

| Set | SHA-256 |
|-----|---------|
| train | `859124bfffe20483b337956efcb11c6bdaf6191835150f51a5e6da272ee5a2eb` |
| val | `1445c0b33c220078d967fa34bb4da5e2981842ce288925b8f2542a4f9352cc64` |
| test | `149092dc6a7a83d3da73e8c7b413ea5e215e3f96c5565767b6db56b3b7835d96` |
| hard_benign_test | `bf90090315a285ec93ea75439dd0654d4bd945efaa00731b68bcd42cb5c493a1` |
| balanced_subtype_test | `210e339329081b97ce95d2dfe4f4581e19bc83dd16a23b86cd36bd15b9ff2023` |
| obfuscated_attacks | `b151322f05ef83922c734c589804fc7eab53331b61b5ad153c72aaf683f7c691` |
| domain_ood | `d14acb99a69ac10559fde9ee22d0f20aa7a5816c4f48f279838d12ed0a6944f0` |
| structural_ood | `63d56a8ebbd0639404158b51b35863fc87b02e9869ee2526f5f90eaa48108dad` |

---

## 5. Note on obfuscated_attacks

An earlier **4,376-row** file was retired: **66%** exact duplicates from `train.csv` and **81%** rows sharing `parent_seed_ids` with train (see manifest). The current **405-row** file is extracted from **test** obfuscated injections with **zero** train overlap — suitable for generalization claims **for this construction**.
