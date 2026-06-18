#\using_libraries\intent_training\compute_normalization.py
from using_libraries.intent_training.scientific_intent_dataset import ScientificIntentDataset
from using_libraries.intent_training.slot_schema import REGRESSION_SLOT_ORDER

import json
import torch
from collections import defaultdict
from pathlib import Path

from using_libraries.intent_training.slot_schema import (
    CONTINUOUS_SLOTS_V2,
    CONT_SLOT_INDEX,
    SLOT_MASKS_V2,
    SCHEMA_VERSION
)


def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


from collections import defaultdict
import numpy as np
import json

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def compute_statistics(dataset_path):

    # Accumulators per continuous slot
    values = defaultdict(list)

    for sample in load_jsonl(dataset_path):

        domain = sample["domain"]
        action = sample["action"]
        slots = sample.get("slots", {})

        mask_info = SLOT_MASKS_V2.get((domain, action), None)
        if mask_info is None:
            continue

        continuous_slots = mask_info["continuous"]

        for slot_name in continuous_slots:
            if slot_name in slots and slots[slot_name] is not None:
                values[slot_name].append(slots[slot_name])

    # Compute mean/std
    stats = {}

    for slot_name, arr in values.items():
        arr = np.array(arr, dtype=float)

        stats[slot_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr) + 1e-8)  # numerical safety
        }

    return stats


def save_stats(stats, output_path):
    with open(output_path, "w") as f:
        json.dump({
            "schema_version": SCHEMA_VERSION,
            "continuous_stats": stats
        }, f, indent=4)


if __name__ == "__main__":

    dataset_path = "synthetic_dataset_hyd_qubits.jsonl"
    output_path = "normalization_stats_v1.json"

    stats = compute_statistics(dataset_path)
    print(stats)
    save_stats(stats, output_path)

    print("Normalization statistics saved.")

    dataset = ScientificIntentDataset("synthetic_dataset_hyd_qubits.jsonl", stats)
    sample = dataset[0]
    sample_raw = dataset.samples[0]
    print(sample_raw["domain"])
    print(sample_raw["slots"])
    for slot in REGRESSION_SLOT_ORDER:
        print(slot, slot in sample_raw["slots"])

    print(sample["reg_targets"].shape)   # must be (13,)
    print(sample["reg_mask"])

    for i in range(5):
        sample = dataset[i]
        print(sample["domain_label"], sample["action_label"])
        print(sample["reg_mask"])
        print("---")

import numpy as np
from collections import defaultdict, Counter

dataset = ScientificIntentDataset("synthetic_dataset_hyd_qubits.jsonl", stats)

print("\n==============================")
print("DATASET SIZE")
print("==============================")
print("Total samples:", len(dataset))


# ==========================================================
# 1. DOMAIN / ACTION BALANCE
# ==========================================================

domain_counter = Counter()
action_counter = Counter()

for sample in dataset.samples:
    domain_counter[sample["domain"]] += 1
    action_counter[sample["action"]] += 1

print("\n==============================")
print("DOMAIN DISTRIBUTION")
print("==============================")
for k, v in domain_counter.items():
    print(f"{k:15s} : {v}")

print("\n==============================")
print("ACTION DISTRIBUTION")
print("==============================")
for k, v in action_counter.items():
    print(f"{k:15s} : {v}")


# ==========================================================
# 2. SLOT OCCURRENCE FREQUENCY
# ==========================================================

slot_counter = Counter()

for sample in dataset.samples:
    for slot in sample["slots"]:
        slot_counter[slot] += 1

print("\n==============================")
print("SLOT FREQUENCY")
print("==============================")
for slot in REGRESSION_SLOT_ORDER:
    print(f"{slot:20s} : {slot_counter.get(slot, 0)}")


# ==========================================================
# 3. REGRESSION STATISTICS (CRITICAL)
# ==========================================================

slot_values = defaultdict(list)

for sample in dataset.samples:
    for slot, value in sample["slots"].items():
        if slot in REGRESSION_SLOT_ORDER:
            slot_values[slot].append(value)

print("\n==============================")
print("REGRESSION SLOT STATS")
print("==============================")

for slot in REGRESSION_SLOT_ORDER:
    values = np.array(slot_values.get(slot, []))
    
    if len(values) == 0:
        print(f"{slot:20s} : NO VALUES")
        continue
    
    mean = values.mean()
    std = values.std()
    min_v = values.min()
    max_v = values.max()
    
    print(f"{slot:20s} | "
          f"count={len(values):4d} | "
          f"mean={mean:8.4f} | "
          f"std={std:8.4f} | "
          f"min={min_v:8.4f} | "
          f"max={max_v:8.4f}")


# ==========================================================
# 4. MASK ACTIVATION RATE
# ==========================================================

mask_sum = np.zeros(len(REGRESSION_SLOT_ORDER))

for i in range(len(dataset)):
    sample = dataset[i]
    mask_sum += sample["reg_mask"].numpy()

print("\n==============================")
print("MASK ACTIVATION RATE")
print("==============================")

for i, slot in enumerate(REGRESSION_SLOT_ORDER):
    rate = mask_sum[i] / len(dataset)
    print(f"{slot:20s} : {rate:.3f}")


# ==========================================================
# 5. QUICK OUTLIER CHECK (3-sigma rule)
# ==========================================================

print("\n==============================")
print("OUTLIER CHECK (|x - mean| > 3σ)")
print("==============================")

for slot in REGRESSION_SLOT_ORDER:
    values = np.array(slot_values.get(slot, []))
    if len(values) < 10:
        continue
    
    mean = values.mean()
    std = values.std()
    
    if std == 0:
        print(f"{slot:20s} : ZERO VARIANCE ⚠️")
        continue
    
    outliers = np.sum(np.abs(values - mean) > 3 * std)
    print(f"{slot:20s} : {outliers} outliers")