#using_libraries.intent_training.predict_inference.py
from using_libraries.intent_training.slot_schema import DOMAIN_LABELS, CONTINUOUS_SLOTS_V2, DOMAIN_LABELS,ACTION_LABELS, SPECTRUM_MODE_LABELS, EVOLUTION_MODE_LABELS, SERIES_LABELS, ID2EVOLUTION_MODE

import torch

def parse_query(query, model, tokenizer, stats, domain_action_modes_config, device):

    inputs = tokenizer(
        query,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    #print(outputs.keys())
    result = postprocess_outputs(
        outputs,
        stats,
        domain_action_modes_config
    )

    return result


import torch
import torch.nn.functional as F


def postprocess_outputs(outputs, stats, domain_action_modes_config):

    result = {}

    # --------------------------------------------------
    # 1. DOMAIN
    # --------------------------------------------------
    domain_logits = outputs["domain_logits"]
    domain_probs = F.softmax(domain_logits, dim=-1)
    domain_idx = torch.argmax(domain_probs, dim=-1).item()

    domain_label = DOMAIN_LABELS[domain_idx]

    result["domain"] = domain_label
    result["domain_confidence"] = domain_probs[0, domain_idx].item()

    # --------------------------------------------------
    # 2. ACTION
    # --------------------------------------------------
    action_logits = outputs["action_logits"]
    action_probs = F.softmax(action_logits, dim=-1)
    action_idx = torch.argmax(action_probs, dim=-1).item()

    action_label = ACTION_LABELS[action_idx]

    result["action"] = action_label
    result["action_confidence"] = action_probs[0, action_idx].item()

    # --------------------------------------------------
    # 3. Allowed slots
    # --------------------------------------------------
    allowed = domain_action_modes_config \
        .get(domain_label, {}) \
        .get(action_label, {})

    # --------------------------------------------------
    # 4. SERIES (if present and allowed)
    # --------------------------------------------------
    if allowed.get("series", False) and "series_logits" in outputs:

        series_logits = outputs["series_logits"]
        series_probs = F.softmax(series_logits, dim=-1)
        series_idx = torch.argmax(series_probs, dim=-1).item()

        result["series"] = SERIES_LABELS[series_idx]
        result["series_confidence"] = series_probs[0, series_idx].item()

    # --------------------------------------------------
    # 5. SPECTRUM MODE
    # --------------------------------------------------
    if allowed.get("spectrum_mode", False):

        spectrum_logits = outputs["spectrum_logits"]
        spectrum_probs = F.softmax(spectrum_logits, dim=-1)
        spectrum_idx = torch.argmax(spectrum_probs, dim=-1).item()

        result["spectrum_mode"] = SPECTRUM_MODE_LABELS[spectrum_idx]
        result["spectrum_mode_confidence"] = spectrum_probs[0, spectrum_idx].item()

    # --------------------------------------------------
    # 6. EVOLUTION MODE
    # --------------------------------------------------
    if allowed.get("evolution_mode", False):

        evolution_logits = outputs["evolution_logits"]
        evolution_probs = F.softmax(evolution_logits, dim=-1)
        evolution_idx = torch.argmax(evolution_probs, dim=-1).item()
                
        result["evolution_mode"] = ID2EVOLUTION_MODE[evolution_idx]
        result["evolution_mode_confidence"] = evolution_probs[0, evolution_idx].item()

    # --------------------------------------------------
    # 7. CONTINUOUS SLOTS
    # --------------------------------------------------
    reg_values = outputs["regression_output"]
    slot_names = CONTINUOUS_SLOTS_V2  # same list used in dataset

    if reg_values.dim() == 2:
        reg_values = reg_values.squeeze(0)

    assert reg_values.dim() == 1
    assert len(slot_names) == reg_values.shape[0]

    #print("reg_values.shape =", reg_values.shape)
    #print("len(slot_names) =", len(slot_names))    
    
    means = []
    stds = []

    for slot in slot_names:
        slot_stats = stats["continuous_stats"][slot]
        means.append(slot_stats["mean"])
        stds.append(slot_stats["std"])

    mean = torch.tensor(means, device=reg_values.device)
    std  = torch.tensor(stds,  device=reg_values.device)

    real_values = reg_values * std + mean

    continuous_dict = {}

    for i, slot_name in enumerate(CONTINUOUS_SLOTS_V2):

        value = real_values[i].item()

        if abs(value) < 1e-6:
            continue

        continuous_dict[slot_name] = value

    result["continuous_slots"] = continuous_dict

    return result

def denormalize_regression(pred, target, mask, stats, slot_order):
    """
    pred:      (B, S)
    target:    (B, S)
    mask:      (B, S)
    stats:     normalization statistics
    slot_order: ordered list of slot names

    Returns:
        mae_total (scalar)
        mae_per_slot (dict)
    """

    pred_denorm = pred.clone()
    target_denorm = target.clone()
    
    if "continuous_stats" in stats:
        stats = stats["continuous_stats"]


    for i, slot in enumerate(slot_order):
        mean = stats[slot]["mean"]
        std = stats[slot]["std"]

        pred_denorm[:, i] = pred[:, i] * std + mean
        target_denorm[:, i] = target[:, i] * std + mean

    abs_error = torch.abs(pred_denorm - target_denorm)
    abs_error = abs_error * mask

    total_mae = abs_error.sum() / (mask.sum() + 1e-8)

    # Per-slot MAE
    mae_per_slot = {}
    for i, slot in enumerate(slot_order):
        slot_mask = mask[:, i]
        if slot_mask.sum() > 0:
            slot_mae = abs_error[:, i].sum() / slot_mask.sum()
            mae_per_slot[slot] = slot_mae.item()

    return total_mae.item(), mae_per_slot
