"""Training losses for CARP heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .modeling_cpu_lite import CPULiteForCausalLM


@dataclass
class CARPLossOutput:
    loss: torch.Tensor
    lm_loss: torch.Tensor
    router_loss: torch.Tensor
    router_accuracy: float


def carp_sft_loss(
    model: CPULiteForCausalLM,
    batch: dict[str, torch.Tensor],
    router_loss_weight: float = 0.2,
) -> CARPLossOutput:
    lm_out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch["labels"],
    )
    if lm_out.loss is None:
        raise RuntimeError("LM loss is required for CARP SFT")
    router_loss = torch.zeros((), dtype=lm_out.loss.dtype, device=lm_out.loss.device)
    router_accuracy = 0.0
    if model.router_head is not None and "router_difficulty" in batch:
        heads = model.carp_heads(batch["input_ids"], attention_mask=batch.get("attention_mask"))
        if heads.router_logits is not None:
            router_loss = F.cross_entropy(heads.router_logits, batch["router_difficulty"])
            pred = torch.argmax(heads.router_logits, dim=-1)
            router_accuracy = float((pred == batch["router_difficulty"]).float().mean().detach().item())
    loss = lm_out.loss + router_loss_weight * router_loss
    return CARPLossOutput(
        loss=loss,
        lm_loss=lm_out.loss.detach(),
        router_loss=router_loss.detach(),
        router_accuracy=router_accuracy,
    )
