
#using_libraries\intent_training\scientific_intent_dataset.py
from using_libraries.intent_training.slot_schema import ACTION2ID, DOMAIN2ID, REGRESSION_SLOT_ORDER, SLOT_MASKS_V2, SERIES2ID
from using_libraries.intent_training.slot_schema import SPECTRUM_MODE2ID, EVOLUTION_MODE2ID, QUBIT_DOMAIN_IDS, HYDROGEN_DOMAIN_ID
from using_libraries.intent_training.slot_schema import OBSERVABLE_TYPE_LABELS, INITIAL_STATE_FAMILY_LABELS, COUPLING_TOPOLOGY_LABELS, OPEN_SYSTEM_MODE_LABELS     
    
            
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ScientificIntentDataset(Dataset):

    def __init__(self, jsonl_path, stats, max_length=128, model_name="distilbert-base-uncased"):
        self.samples = self._load_jsonl(jsonl_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self.stats = stats
        #self.stats = stats["continuous_stats"]

        if "continuous_stats" in stats:
            self.stats = stats["continuous_stats"]
        else:
            self.stats = stats

        self.max_length = max_length

    def _load_jsonl(self, path):
        data = []
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        text = sample["input_text"]
        domain = sample["domain"]
        action = sample["action"]
        slots = sample["slots"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # -------------------------------------------------
        # Domain & Action Labels
        # -------------------------------------------------

        domain_label = torch.tensor(DOMAIN2ID[domain])
        action_label = torch.tensor(ACTION2ID[action])

        # -------------------------------------------------
        # Slot Mask Configuration
        # -------------------------------------------------

        mask_info = SLOT_MASKS_V2[(domain, action)]

        # -------------------------------------------------
        # Regression Targets
        # -------------------------------------------------

        reg_targets = torch.zeros(len(REGRESSION_SLOT_ORDER))
        reg_mask = torch.zeros(len(REGRESSION_SLOT_ORDER))

        for slot_name in mask_info["continuous"]:

            if slot_name not in REGRESSION_SLOT_ORDER:
                continue

            i = REGRESSION_SLOT_ORDER.index(slot_name)

            value = slots.get(slot_name, 0.0)

            mean = self.stats[slot_name]["mean"]
            std = self.stats[slot_name]["std"]

            norm_value = (value - mean) / std

            reg_targets[i] = norm_value
            reg_mask[i] = 1.0

        # -------------------------------------------------
        # Series Classification
        # -------------------------------------------------

        series_target = torch.tensor(-1)
        series_mask = torch.tensor(0)

        if "series" in mask_info["categorical"]:

            series_value = slots.get("series", "none")

            series_target = torch.tensor(
                SERIES2ID[series_value]
            )

            series_mask = torch.tensor(1)

        # -------------------------------------------------
        # Spectrum Mode Classification
        # -------------------------------------------------

        spectrum_target = torch.tensor(-1)
        spectrum_mask = torch.tensor(0)

        if "spectrum_mode" in mask_info["categorical"]:

            spectrum_value = slots.get(
                "spectrum_mode",
                "none"
            )

            spectrum_target = torch.tensor(
                SPECTRUM_MODE2ID[spectrum_value]
            )

            spectrum_mask = torch.tensor(1)

        # -------------------------------------------------
        # Evolution Mode Classification
        # -------------------------------------------------

        evolution_target = torch.tensor(-1)
        evolution_mask = torch.tensor(0)

        if "evolution_mode" in mask_info["categorical"]:

            evolution_value = slots.get(
                "evolution_mode",
                "none"
            )

            evolution_target = torch.tensor(
                EVOLUTION_MODE2ID[evolution_value]
            )

            evolution_mask = torch.tensor(1)

        # -------------------------------------------------
        # Open System Mode Classification
        # -------------------------------------------------
        open_system_target = torch.tensor(-1)
        open_system_mask = torch.tensor(0.0)

        if "open_system_mode" in mask_info["categorical"]:

            open_system_value = slots.get(
                "open_system_mode",
                "none"
            )

            open_system_target = torch.tensor(
                OPEN_SYSTEM_MODE_LABELS[open_system_value]
            )

            open_system_mask = torch.tensor(
                float(
                    "open_system_mode"
                    in mask_info["categorical"]
                )
            )


        # -------------------------------------------------
        # Coupling Topology Classification
        # -------------------------------------------------
        topology_target = torch.tensor(-1)
        topology_mask = torch.tensor(0.0)

        if "coupling_topology" in mask_info["categorical"]:

            topology_value = slots.get(
                "coupling_topology",
                "none"
            )

            topology_target = torch.tensor(
                COUPLING_TOPOLOGY_LABELS[topology_value]
            )

            topology_mask = torch.tensor(
                float(
                    "coupling_topology"
                    in mask_info["categorical"]
                )
            )

        # -------------------------------------------------
        # Initial State Classification
        # -------------------------------------------------
        initial_state_target = torch.tensor(-1)
        initial_state_mask = torch.tensor(0.0)

        if "initial_state_family" in mask_info["categorical"]:

            initial_state_value = slots.get(
                "initial_state_family",
                "none"
            )

            initial_state_target = torch.tensor(
                INITIAL_STATE_FAMILY_LABELS[
                    initial_state_value
                ]
            )

            initial_state_mask = torch.tensor(
                float(
                    "initial_state_family"
                    in mask_info["categorical"]
                )
            )
        # -------------------------------------------------
        # Observable Type Classification
        # -------------------------------------------------
        observable_target = torch.tensor(-1)
        observable_mask = torch.tensor(0.0)

        if "observable_type" in mask_info["categorical"]:

            observable_value = slots.get(
                "observable_type",
                "none"
            )

            observable_target = torch.tensor(
                OBSERVABLE_TYPE_LABELS[
                    observable_value
                ]
            )

            observable_mask = torch.tensor(
                float(
                    "observable_type"
                    in mask_info["categorical"]
                )
            )


        # -------------------------------------------------
        # Return Batch Item
        # -------------------------------------------------

        return {

            "input_ids": input_ids,
            "attention_mask": attention_mask,

            "domain_label": domain_label,
            "action_label": action_label,

            "reg_targets": reg_targets,
            "reg_mask": reg_mask,

            "series_target": series_target,
            "series_mask": series_mask,

            "spectrum_target": spectrum_target,
            "spectrum_mask": spectrum_mask,

            "evolution_target": evolution_target,
            "evolution_mask": evolution_mask,

            "open_system_target": open_system_target,
            "open_system_mask": open_system_mask,

            "topology_target": topology_target,
            "topology_mask": topology_mask,

            "initial_state_target": initial_state_target,
            "initial_state_mask": initial_state_mask,

            "observable_target": observable_target,
            "observable_mask": observable_mask,

        }