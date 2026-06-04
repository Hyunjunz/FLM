"Training losses for CARP heads."""

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
    ranking_loss: torch.Tensor
    router_accuracy: float


def carp_sft_loss(
    model: CPULiteForCausalLM,
    batch: dict[str, torch.Tensor],
    router_loss_weight: float = 0.2,
    ranking_loss_weight: float = 0.5,
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

    ranking_loss = torch.zeros((), dtype=lm_out.loss.dtype, device=lm_out.loss.device)
    if ranking_loss_weight > 0 and "candidate_ids" in batch:
        ranking_loss = choice_ranking_loss(model, batch)

    loss = lm_out.loss + router_loss_weight * router_loss + ranking_loss_weight * ranking_loss
    return CARPLossOutput(
        loss=loss,
        lm_loss=lm_out.loss.detach(),
        router_loss=router_loss.detach(),
        ranking_loss=ranking_loss.detach(),
        router_accuracy=router_accuracy,
    )


def choice_ranking_loss(model: CPULiteForCausalLM, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Cross-entropy over candidate mean log probabilities.

    Supports mixed LM + choice batches through rank_active. Non-choice rows
    are ignored instead of disabling the entire rank loss.
    """
    prompt_ids = batch["prompt_ids"]
    prompt_mask = batch["prompt_attention_mask"].bool()
    candidate_ids = batch["candidate_ids"]
    candidate_mask_tokens = batch["candidate_attention_mask"].bool()
    candidate_mask = batch["candidate_mask"].bool()
    rank_active = batch.get("rank_active")

    if rank_active is None:
        rank_active = candidate_mask.any(dim=1)
    else:
        rank_active = rank_active.bool()

    active_rows = torch.nonzero(rank_active, as_tuple=False).flatten()
    if active_rows.numel() == 0:
        return torch.zeros((), device=prompt_ids.device, dtype=torch.float)

    scores = []
    gold = []
    for batch_idx_tensor in active_rows:
        batch_idx = int(batch_idx_tensor.item())
        prompt = prompt_ids[batch_idx, prompt_mask[batch_idx]]
        row_scores = []
        for choice_idx in range(candidate_ids.size(1)):
            if not bool(candidate_mask[batch_idx, choice_idx]):
                row_scores.append(torch.full((), -1e4, device=prompt_ids.device, dtype=torch.float))
                continue
            candidate = candidate_ids[batch_idx, choice_idx, candidate_mask_tokens[batch_idx, choice_idx]]
            row_scores.append(_mean_logprob(model, prompt, candidate))
        scores.append(torch.stack(row_scores))
        gold.append(batch["gold_choice"][batch_idx])

    logits = torch.stack(scores, dim=0)
    gold_choice = torch.stack(gold).to(device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, gold_choice)


def _mean_logprob(model: CPULiteForCausalLM, prompt: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    ids = torch.cat([prompt, candidate]).unsqueeze(0)
    out = model(ids, use_cache=False)
    prompt_len = prompt.numel()
    logits = out.logits[:, max(0, prompt_len - 1) : -1, :]
    labels = ids[:, prompt_len:]

    if logits.numel() == 0 or labels.numel() == 0:
        return torch.full((), -100.0, device=ids.device, dtype=torch.float)

    logprobs = F.log_softmax(logits, dim=-1)
    chosen = torch.gather(logprobs, -1, labels.unsqueeze(-1)).squeeze(-1)
    return chosen.mean()
