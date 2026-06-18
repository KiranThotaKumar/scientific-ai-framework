#using_libraries.intent_training.lib_train_serialization.py

import os
import json
import torch
from pathlib import Path
from using_libraries.intent_training.scientific_intent_model import ScientificIntentModel

def save_checkpoint(model, stats, model_confg, domain_action_modes_config, path="models/checkpoints/scientific_intent_v1.pt"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "normalization_stats": stats,
        "model_config": model_confg,
        "domain_action_modes_config": domain_action_modes_config
    }
 
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)

    model_config = checkpoint["model_config"]
    stats = checkpoint["normalization_stats"]
    domain_action_modes_config = checkpoint["domain_action_modes_config"]

    model = ScientificIntentModel(
    num_domains=model_config["num_domains"],
    num_actions=model_config["num_actions"],
    num_spectrum_modes=model_config["num_spectrum_modes"],
    num_series_labels=model_config["num_series_labels"],
    num_evolution_modes=model_config["num_evolution_modes"],

    num_open_system_modes=model_config["num_open_system_modes"],
    num_coupling_topologies=model_config["num_coupling_topologies"],
    num_initial_state_families=model_config["num_initial_state_families"],
    num_observable_types=model_config["num_observable_types"],

    num_continuous=model_config["num_continuous"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, stats, model_config, domain_action_modes_config 