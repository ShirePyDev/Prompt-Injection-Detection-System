import pandas as pd

# ============================================================
# Paths
# ============================================================

train_path = "data/pids_bench_v3/train.csv"
test_path = "data/pids_bench_v3/test.csv"


# ============================================================
# Load datasets
# ============================================================

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_texts = set(train_df["text"].astype(str))
test_texts = set(test_df["text"].astype(str))


# ============================================================
# Check overlap
# ============================================================

overlap = train_texts.intersection(test_texts)

print("Train samples:", len(train_texts))
print("Test samples:", len(test_texts))
print("Overlapping prompts:", len(overlap))


# ============================================================
# Show examples if leakage exists
# ============================================================

if overlap:
    print("\nExample overlapping prompts:\n")

    for i, prompt in enumerate(list(overlap)[:10]):
        print(f"{i+1}. {prompt}\n")

else:
    print("\n✅ No exact prompt overlap detected.")