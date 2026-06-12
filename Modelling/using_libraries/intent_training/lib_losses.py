#using_libraries\intent_training\lib_losses.py


import torch.nn.functional as F
import torch.nn as nn

def masked_mse_loss(pred, target, mask):

    diff = (pred - target) ** 2
    masked = diff * mask

    loss = masked.sum() / (mask.sum() + 1e-8)
    return loss



def compute_loss(outputs, batch, lambda_reg=0.3):

    # ----- Domain -----
    loss_domain = F.cross_entropy(
        outputs["domain_logits"],
        batch["domain_label"]
    )

    # ----- Action -----
    loss_action = F.cross_entropy(
        outputs["action_logits"],
        batch["action_label"]
    )

    # ----- Regression (masked MSE) -----
    mse = (outputs["regression_output"] - batch["reg_targets"]) ** 2
    mse = mse * batch["reg_mask"]
    loss_reg = mse.sum() / (batch["reg_mask"].sum() + 1e-8)

    ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)

    # ----- Spectrum -----
    spectrum_loss_raw = ce(
        outputs["spectrum_logits"],
        batch["spectrum_target"]
    )

    spectrum_loss = (
        spectrum_loss_raw * batch["spectrum_mask"]
    ).sum() / (batch["spectrum_mask"].sum() + 1e-8)

        # ----- Series -----
    series_loss_raw = ce(
        outputs["series_logits"],
        batch["series_target"]
    )

    series_loss = (
        series_loss_raw * batch["series_mask"]
    ).sum() / (batch["series_mask"].sum() + 1e-8)

    # ----- Evolution -----
    evolution_loss_raw = ce(
        outputs["evolution_logits"],
        batch["evolution_target"]
    )

    evolution_loss = (
        evolution_loss_raw * batch["evolution_mask"]
    ).sum() / (batch["evolution_mask"].sum() + 1e-8)

    # ----- Open System -----
    open_system_loss_raw = ce(
        outputs["open_system_logits"],
        batch["open_system_target"]
    )

    open_system_loss = (
        open_system_loss_raw * batch["open_system_mask"]
    ).sum() / (batch["open_system_mask"].sum() + 1e-8)

    # ----- Topology -----
    topology_loss_raw = ce(
        outputs["topology_logits"],
        batch["topology_target"]
    )

    topology_loss = (
        topology_loss_raw * batch["topology_mask"]
    ).sum() / (batch["topology_mask"].sum() + 1e-8)

    # ----- Initial State -----
    initial_state_loss_raw = ce(
        outputs["initial_state_logits"],
        batch["initial_state_target"]
    )

    initial_state_loss = (
        initial_state_loss_raw * batch["initial_state_mask"]
    ).sum() / (batch["initial_state_mask"].sum() + 1e-8)

    # ----- Observables -----
    observable_loss_raw = ce(
        outputs["observable_logits"],
        batch["observable_target"]
    )

    observable_loss = (
        observable_loss_raw * batch["observable_mask"]
    ).sum() / (batch["observable_mask"].sum() + 1e-8)

    # ----- Total -----
    total_loss = (
        loss_domain
        + loss_action
        + spectrum_loss
        + series_loss
        + evolution_loss
        + initial_state_loss
        + observable_loss
        + topology_loss
        + open_system_loss
        + lambda_reg * loss_reg
    )

    return total_loss, {
    "domain": loss_domain.item(),
    "action": loss_action.item(),
    "spectrum": spectrum_loss.item(),
    "series":series_loss.item(),
    "evolution": evolution_loss.item(),
    "initial_state":initial_state_loss.item(),
    "observable":observable_loss.item(),
    "topology":topology_loss.item(),
    "open_system":open_system_loss.item(),

    "regression": loss_reg.item()
}