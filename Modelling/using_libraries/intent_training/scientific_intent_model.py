#using_libraries\intent_training\scientific_intent_model.py


import torch
import torch.nn as nn
from transformers import DistilBertModel


class ScientificIntentModel(nn.Module):

    def __init__(
        self,
        num_domains,
        num_actions,
        num_spectrum_modes,
        num_series_labels,
        num_evolution_modes,
        num_open_system_modes,
        num_coupling_topologies,
        num_initial_state_families,
        num_observable_types,
        num_continuous,
        hidden_dim=256,
    ):
        super().__init__()

        # ---------------- Encoder ----------------
        self.encoder = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        encoder_dim = self.encoder.config.hidden_size  # 768

        # ---------------- Shared projection ----------------
        self.shared = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # ---------------- Classification Heads ----------------
        self.domain_head = nn.Linear(hidden_dim, num_domains)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.spectrum_head = nn.Linear(hidden_dim, num_spectrum_modes)
        self.series_head = nn.Linear(hidden_dim, num_series_labels)
        self.evolution_head = nn.Linear(hidden_dim, num_evolution_modes)
        self.open_system_head = nn.Linear(hidden_dim, num_open_system_modes)
        self.topology_head = nn.Linear(hidden_dim, num_coupling_topologies)
        self.initial_state_head = nn.Linear(hidden_dim, num_initial_state_families)
        self.observable_head = nn.Linear(hidden_dim, num_observable_types)
        
        # ---------------- Regression Head ----------------

        self.regression_head = nn.Linear(hidden_dim, num_continuous)

    def forward(self, input_ids, attention_mask):

        # ---- Encode ----
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # DistilBERT does not have pooler_output
        # Use CLS token (position 0)
        cls = outputs.last_hidden_state[:, 0]

        shared = self.shared(cls)

        # ---- Heads ----
        domain_logits = self.domain_head(shared)
        action_logits = self.action_head(shared)
        spectrum_logits = self.spectrum_head(shared)
        series_logits = self.series_head(shared)
        evolution_logits = self.evolution_head(shared)

        open_system_logits = self.open_system_head(shared)
        topology_logits = self.topology_head(shared)
        initial_state_logits =  self.initial_state_head(shared)
        observable_logits = self.observable_head(shared)
        
        regression_output = self.regression_head(shared)

        return {
            "domain_logits": domain_logits,
            "action_logits": action_logits,
            "spectrum_logits": spectrum_logits,
            "evolution_logits": evolution_logits,
            "series_logits": series_logits,            
            "open_system_logits":open_system_logits,
            "topology_logits":topology_logits,
            "initial_state_logits":initial_state_logits,
            "observable_logits":observable_logits,
            "regression_output": regression_output
        }
