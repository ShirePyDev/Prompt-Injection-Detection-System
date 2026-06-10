from __future__ import annotations
from collections import Counter, defaultdict

def count_by(rows, key):
    return Counter([r.get(key, "NA") for r in rows])

def obfuscation_table(rows):
    # overall + by label + by attack_type
    overall = count_by(rows, "obfuscation")
    by_label = defaultdict(Counter)
    by_type = defaultdict(Counter)
    for r in rows:
        by_label[r["label"]][r.get("obfuscation","none")] += 1
        by_type[r.get("attack_type","none")][r.get("obfuscation","none")] += 1
    return overall, by_label, by_type