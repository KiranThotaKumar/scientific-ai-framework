#using_libraries\intent_training\lib_interim_test_three.py

from using_libraries.intent_training.slot_schema import NUM_DOMAINS, NUM_ACTIONS, NUM_SPECTRUM_MODES, NUM_EVOLUTION_MODES 
from using_libraries.intent_training.slot_schema import NUM_CONTINUOUS, MODEL_CONFIG, DOMAIN_ACTION_MODES_CONFIG, REGRESSION_SLOT_ORDER, NUM_SERIES_LABELS
from using_libraries.intent_training.slot_schema import NUM_OPEN_SYSTEM_MODES, NUM_COUPLING_TOPOLOGIES, NUM_INITIAL_STATE_FAMILIES, NUM_OBSERVABLE_TYPES
from using_libraries.intent_training.scientific_intent_model import ScientificIntentModel
from using_libraries.intent_training.scientific_intent_dataset import ScientificIntentDataset
import json
import torch
from torch.utils.data import DataLoader
from using_libraries.intent_training.lib_losses import compute_loss
from using_libraries.intent_training.slot_schema import SLOT_MASKS_V2, DOMAIN_ACTION_MODES_CONFIG
from using_libraries.intent_training.lib_train_serialization import save_checkpoint, load_model
from using_libraries.intent_training.predict_inference import parse_query
from using_libraries.intent_training.predict_inference import denormalize_regression
from nlp.detectors.neural_intent_adapter import NeuralIntentAdapter
from execution.nlp_intent.intent_engine import IntentEngine
from core.router.scientific_router import ScientificRouter
from transformers import AutoTokenizer
from core.router.scientific_router import ScientificRouter
from execution.intent_execution_bridge import IntentExecutionBridge
from registrys.executor_registry import ExecutorRegistry
from execution.executors.hydrogen_executor import HydrogenDomainExecutor
from using_libraries.scientific_ai_engine import ScientificAIEngine
from collections import Counter


def _load_jsonl(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

#############  #############  Integration Testing - All Domains  #############
# from collections import Counter

# # --------------------------------------------------
# # 1. Load normalization stats
# # --------------------------------------------------
# with open("normalization_stats_v1.json", "r") as f:
#     stats = json.load(f)

# # --------------------------------------------------
# # 2. Dataset + Dataloader
# # --------------------------------------------------

# from collections import Counter
# import json

# counter = Counter()

# with open("synthetic_dataset_hyd_qubits.jsonl", "r") as f:

#     for line in f:

#         sample = json.loads(line)

#         slots = sample.get("slots", {})

#         if "evolution_mode" in slots:

#             counter[slots["evolution_mode"]] += 1

# print(counter)

# engine = ScientificAIEngine(
#     "scientific_intent_v1.pt"
#     )

# query = \
#     "Please give single qubit evolution parameters from file synthetic_single_qubit.npz"
#     #"Simulate single qubit evolution"
#     #"Please show me two qubit evolution dynamics"
    
    
#     #"Infer parameters from two qubit file synthetic_multi_qubit_data.npz"    
    
#     #"Generate Balmer spectrum"    
            
#     #"Infer parameters from Hydrogen spectrum file synthetic_hydrogen.npz"    
    
    
# intent = engine.detect(query)

# print(intent)

# result = engine.run(query)

# print(result)

# exit()
##############################################################################  

# --------------------------------------------------
# 1. Load normalization stats
# --------------------------------------------------
with open("normalization_stats_v1.json", "r") as f:
    stats = json.load(f)

# --------------------------------------------------
# 2. Dataset + Dataloader
# --------------------------------------------------
dataset = ScientificIntentDataset(
    jsonl_path="synthetic_dataset_hyd_qubits.jsonl",
    stats=stats,
    max_length=128
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# 3. Distribution Checks
# --------------------------------------------------
action_counter = Counter(sample["action"] for sample in dataset.samples)
domain_counter = Counter(sample["domain"] for sample in dataset.samples)

print("Action distribution:", action_counter)
print("Domain distribution:", domain_counter)

from collections import Counter

joint_counter = Counter(
    (sample["domain"], sample["action"])
    for sample in dataset.samples
)

print("Joint distribution (domain, action):")
for key, value in joint_counter.items():
    print(f"{key}: {value}")

label_counter = Counter()
for i in range(len(dataset)):
    item = dataset[i]
    label_counter[item["action_label"].item()] += 1

print("Encoded Label Distribution:", label_counter)

# --------------------------------------------------
# 4. Mask Integrity Check (Full Dataset)
# --------------------------------------------------
spec_total = 0
evo_total = 0
overlap_total = 0
topology_mask_total  = 0
observable_mask_total  = 0
initial_state_mask_total  = 0
open_system_mask_total  = 0


for batch in dataloader:
    spec_total += batch["spectrum_mask"].sum().item()
    evo_total += batch["evolution_mask"].sum().item()
    
    open_system_mask_total += batch["open_system_mask"].sum().item()
    topology_mask_total += batch["topology_mask"].sum().item()
    initial_state_mask_total += batch["initial_state_mask"].sum().item()
    observable_mask_total += batch["observable_mask"].sum().item()

    overlap_total += (
        batch["spectrum_mask"] * batch["evolution_mask"]
    ).sum().item()


print("Total spectrum mask:", spec_total)
print("Total evolution mask:", evo_total)
print("Total overlap:", overlap_total)

print("Total Open System mask:", open_system_mask_total)
print("Total Topology mask:", topology_mask_total)
print("Total Initial State mask:", initial_state_mask_total)
print("Total Observable mask:", observable_mask_total)

assert overlap_total == 0, "ERROR: Mask overlap detected!"

# --------------------------------------------------
# 5. Single Batch Shape Check
# --------------------------------------------------
batch = next(iter(dataloader))

print("\nBatch Tensor Shapes:")
for k, v in batch.items():
    print(k, v.shape)

# --------------------------------------------------
# 6. Model Forward + Backward Sanity
# --------------------------------------------------
model = ScientificIntentModel(
    num_domains=NUM_DOMAINS,
    num_actions=NUM_ACTIONS,
    num_spectrum_modes=NUM_SPECTRUM_MODES,
    num_series_labels=NUM_SERIES_LABELS,
    num_evolution_modes=NUM_EVOLUTION_MODES,
    num_open_system_modes=NUM_OPEN_SYSTEM_MODES,
    num_coupling_topologies=NUM_COUPLING_TOPOLOGIES,
    num_initial_state_families=NUM_INITIAL_STATE_FAMILIES,
    num_observable_types=NUM_OBSERVABLE_TYPES,
    num_continuous=NUM_CONTINUOUS,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

epoch_mae = 0.0
epoch_samples = 0

slot_mae_accumulator = {s: 0.0 for s in REGRESSION_SLOT_ORDER}
slot_counts = {s: 0.0 for s in REGRESSION_SLOT_ORDER}

for epoch in range(2):

    total_loss = 0.0
    
    individual_losses = {
        "domain": 0.0,
        "action": 0.0,
        "spectrum": 0.0,
        "series":0.0,
        "evolution": 0.0,
        "open_system": 0.0,
        "topology": 0.0,
        "initial_state": 0.0,
        "observable": 0.0,
        "regression": 0.0
    }

    for batch in dataloader:

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        #print("topology_logits shape: ",outputs["topology_logits"].shape)
        #print("initial_state shape: ", outputs["initial_state_logits"].shape)
        #print("observable_logits shape: ", outputs["observable_logits"].shape)

        model.train()
        loss, individual = compute_loss(outputs, batch)           
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        # accumulate individual losses
        for key in individual_losses:
            individual_losses[key] += individual[key]

    
        mae_total, mae_per_slot = denormalize_regression(
        outputs["regression_output"].detach(),
        batch["reg_targets"],
        batch["reg_mask"],
        stats,
        REGRESSION_SLOT_ORDER
        )

        epoch_mae += mae_total * batch["reg_mask"].sum().item()
        epoch_samples += batch["reg_mask"].sum().item()

        for slot, value in mae_per_slot.items():
            slot_mae_accumulator[slot] += value
            slot_counts[slot] += 1

    avg_loss = total_loss / len(dataloader)

    # normalize individual losses
    for key in individual_losses:
        individual_losses[key] /= len(dataloader)

    print(f"\nEpoch {epoch}: {avg_loss:.4f}")

    for key, value in individual_losses.items():
        print(f"{key}: {value:.4f}")

    print("Denormalized MAE (global):", epoch_mae / epoch_samples)

    print("Per-slot MAE:")
    for slot in REGRESSION_SLOT_ORDER:
        if slot_counts[slot] > 0:
            print(f"{slot:20s}: {slot_mae_accumulator[slot]/slot_counts[slot]:.4f}")

#save_checkpoint(model, stats, MODEL_CONFIG, DOMAIN_ACTION_MODES_CONFIG)
save_checkpoint(model, stats, MODEL_CONFIG, DOMAIN_ACTION_MODES_CONFIG)

