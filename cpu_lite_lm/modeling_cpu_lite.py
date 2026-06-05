"""PyTorch implementation of CPULiteLM."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_cpu_lite import CPULiteConfig

try:
    from transformers import PreTrainedModel
except Exception:  # pragma: no cover
    class PreTrainedModel(nn.Module):
        config_class = None

        def __init__(self, config: Any) -> None:
            super().__init__()
            self.config = config

        def save_pretrained(self, save_directory: str | Path) -> None:
            save_path = Path(save_directory)
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), save_path / "pytorch_model.bin")
            self.config.save_pretrained(save_path)

        @classmethod
        def from_pretrained(cls, path: str | Path, *args: Any, **kwargs: Any):
            config = cls.config_class.from_pretrained(path)
            model = cls(config)
            state = torch.load(Path(path) / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state)
            return model


PastKeyValue = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class CPULiteCausalLMOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    past_key_values: Optional[List[PastKeyValue]]
    moe_loss: Optional[torch.Tensor] = None


@dataclass
class CPULiteCARPHeadOutput:
    router_logits: Optional[torch.Tensor]
    verifier_logits: Optional[torch.Tensor]


class CPULiteRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        # Fast path for CPU: avoid unnecessary copies if possible
        var = x_float.pow(2).mean(dim=-1, keepdim=True)
        return (x_float * torch.rsqrt(var + self.eps)).to(dtype) * self.weight


class CPULiteRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings
        self._set_cache(max_position_embeddings)

    def _set_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_pos = int(position_ids.max().item()) + 1
        if max_pos > self.cos_cached.size(2):
            self._set_cache(max_pos)
        
        # Avoid repeated to() calls if possible
        cos = self.cos_cached.narrow(2, 0, max_pos)[:, :, position_ids, :]
        sin = self.sin_cached.narrow(2, 0, max_pos)[:, :, position_ids, :]
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # q/k layout: [B, H, T, D]
    bsz, _, seq_len, head_dim = q.shape

    def fix_freq(x: torch.Tensor) -> torch.Tensor:
        # Common incoming shapes:
        # [T, D]
        # [B, T, D]
        # [B, T, 1, D]
        # [B, 1, T, D]
        # [1, B, T, D]
        x = x.to(device=q.device, dtype=q.dtype)

        if x.dim() == 2:
            # [T, D] -> [1, 1, T, D]
            x = x.unsqueeze(0).unsqueeze(0)

        elif x.dim() == 3:
            # [B, T, D] -> [B, 1, T, D]
            if x.shape[0] == bsz and x.shape[1] == seq_len:
                x = x.unsqueeze(1)
            # [T, B, D] -> [B, 1, T, D]
            elif x.shape[0] == seq_len and x.shape[1] == bsz:
                x = x.transpose(0, 1).unsqueeze(1)
            # [1, T, D] -> [1, 1, T, D]
            elif x.shape[0] == 1 and x.shape[1] == seq_len:
                x = x.unsqueeze(1)
            else:
                raise RuntimeError(f"Unsupported RoPE freq 3D shape {tuple(x.shape)} for q {tuple(q.shape)}")

        elif x.dim() == 4:
            # [B, T, 1, D] -> [B, 1, T, D]
            if x.shape[0] == bsz and x.shape[1] == seq_len and x.shape[2] == 1:
                x = x.transpose(1, 2)
            # [B, 1, T, D] already OK
            elif x.shape[0] == bsz and x.shape[1] == 1 and x.shape[2] == seq_len:
                pass
            # [1, B, T, D] -> [B, 1, T, D]
            elif x.shape[0] == 1 and x.shape[1] == bsz and x.shape[2] == seq_len:
                x = x.squeeze(0).unsqueeze(1)
            # [1, 1, T, D] already OK
            elif x.shape[0] == 1 and x.shape[1] == 1 and x.shape[2] == seq_len:
                pass
            else:
                raise RuntimeError(f"Unsupported RoPE freq 4D shape {tuple(x.shape)} for q {tuple(q.shape)}")

        elif x.dim() == 5:
            # Observed broken shape: [1, 1, B, T, D]
            # Convert to [B, 1, T, D].
            if x.shape[0] == 1 and x.shape[1] == 1 and x.shape[2] == bsz and x.shape[3] == seq_len:
                x = x.squeeze(0).squeeze(0).unsqueeze(1)
            # Or [1, 1, 1, T, D] -> [1, 1, T, D]
            elif x.shape[0] == 1 and x.shape[1] == 1 and x.shape[2] == 1 and x.shape[3] == seq_len:
                x = x.squeeze(0).squeeze(0)
            else:
                raise RuntimeError(f"Unsupported RoPE freq 5D shape {tuple(x.shape)} for q {tuple(q.shape)}")

        else:
            raise RuntimeError(f"Unsupported RoPE freq shape {tuple(x.shape)} for q {tuple(q.shape)}")

        return x

    cos = fix_freq(cos)
    sin = fix_freq(sin)

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class CPULiteAttention(nn.Module):
    def __init__(self, config: CPULiteConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.use_sdpa = getattr(config, "use_sdpa", True)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = CPULiteRotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def _shape(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[PastKeyValue]]:
        bsz, q_len, _ = hidden_states.shape
        query = self._shape(self.q_proj(hidden_states), self.num_heads)
        key = self._shape(self.k_proj(hidden_states), self.num_kv_heads)
        value = self._shape(self.v_proj(hidden_states), self.num_kv_heads)
        cos, sin = self.rotary_emb(value, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_value is not None:
            if cache_position is not None:
                past_key_value[0][:, :, cache_position] = key.to(past_key_value[0].dtype)
                past_key_value[1][:, :, cache_position] = value.to(past_key_value[1].dtype)
                key = past_key_value[0][:, :, :cache_position[-1] + 1]
                value = past_key_value[1][:, :, :cache_position[-1] + 1]
            else:
                key = torch.cat([past_key_value[0], key], dim=2)
                value = torch.cat([past_key_value[1], value], dim=2)
        
        present = (key, value) if use_cache else None

        if self.num_kv_groups > 1:
            key_for_attn = key.repeat_interleave(self.num_kv_groups, dim=1)
            value_for_attn = value.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            key_for_attn = key
            value_for_attn = value

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                query,
                key_for_attn,
                value_for_attn,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=attention_mask is None and q_len > 1,
            )
        else:
            kv_len = key_for_attn.size(-2)
            attn_weights = torch.matmul(query, key_for_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            elif q_len > 1:
                past_len = kv_len - q_len
                q_pos = torch.arange(past_len, past_len + q_len, device=hidden_states.device)[:, None]
                k_pos = torch.arange(kv_len, device=hidden_states.device)[None, :]
                causal = k_pos <= q_pos
                attn_weights = attn_weights.masked_fill(
                    ~causal[None, None, :, :], torch.finfo(attn_weights.dtype).min
                )
            
            attn_probs = F.softmax(attn_weights.float(), dim=-1).to(query.dtype)
            attn_output = torch.matmul(attn_probs, value_for_attn)
        
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), present


class CPULiteMLP(nn.Module):
    def __init__(self, config: CPULiteConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CPULiteMoE(nn.Module):
    """Mixture of Experts for CPULiteLM."""
    def __init__(self, config: CPULiteConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([CPULiteMLP(config) for _ in range(self.num_experts)])
        self.moe_loss_weight = config.moe_loss_weight

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden_dim = x.shape
        flat_x = x.view(-1, hidden_dim)
        
        router_logits = self.router(flat_x)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Load balancing loss (auxiliary loss)
        # Based on Switch Transformer / GShard
        probs = routing_weights.mean(dim=0)
        # Fraction of tokens assigned to each expert
        _, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        expert_mask = F.one_hot(selected_experts, self.num_experts).float()
        density = expert_mask.mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(probs * density)
        
        # Top-k routing
        weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        results = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            # Mask for tokens that chose this expert
            mask = (selected_experts == i).any(dim=-1)
            if not mask.any():
                continue
            
            expert_input = flat_x[mask]
            expert_output = expert(expert_input)
            
            # Find the weight assigned to this expert for each token
            # Note: a token might have selected this expert in any of the top-k slots
            expert_weights = (selected_experts[mask] == i).float() * weights[mask]
            # Sum weights if multiple slots selected same expert (unlikely with top-k unique but safe)
            combined_weight = expert_weights.sum(dim=-1, keepdim=True)
            
            results[mask] += combined_weight * expert_output
            
        return results.view(bsz, seq_len, hidden_dim), aux_loss


class CPULiteDecoderLayer(nn.Module):
    def __init__(self, config: CPULiteConfig) -> None:
        super().__init__()
        self.input_layernorm = CPULiteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CPULiteAttention(config)
        self.post_attention_layernorm = CPULiteRMSNorm(config.hidden_size, config.rms_norm_eps)
        
        if config.num_experts > 0:
            self.mlp = CPULiteMoE(config)
        else:
            self.mlp = CPULiteMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[PastKeyValue],
        use_cache: bool,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[PastKeyValue], torch.Tensor]:
        residual = hidden_states
        attn_out, present = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
            cache_position,
        )
        hidden_states = residual + attn_out
        
        residual = hidden_states
        mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
        
        moe_loss = torch.zeros((), device=hidden_states.device)
        if isinstance(mlp_out, tuple):
            mlp_hidden, moe_loss = mlp_out
            hidden_states = residual + mlp_hidden
        else:
            hidden_states = residual + mlp_out
            
        return hidden_states, present, moe_loss


class CPULitePreTrainedModel(PreTrainedModel):
    config_class = CPULiteConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class CPULiteModel(CPULitePreTrainedModel):
    def __init__(self, config: CPULiteConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([CPULiteDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = CPULiteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.apply(self._init_weights)

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        bsz: int,
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if q_len <= 1 and attention_mask is None:
            return None
        past_len = kv_len - q_len
        q_pos = torch.arange(past_len, past_len + q_len, device=device)[:, None]
        k_pos = torch.arange(kv_len, device=device)[None, :]
        causal = k_pos <= q_pos
        mask = torch.zeros((q_len, kv_len), device=device, dtype=dtype)
        mask = mask.masked_fill(~causal, torch.finfo(dtype).min)
        mask = mask[None, None, :, :].expand(bsz, 1, q_len, kv_len)
        if attention_mask is not None:
            if attention_mask.size(-1) != kv_len:
                pad = kv_len - attention_mask.size(-1)
                attention_mask = F.pad(attention_mask, (pad, 0), value=1)
            padding = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
            mask = mask + padding
        return mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[PastKeyValue]] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        output_layer: Optional[int] = None,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[PastKeyValue]], Optional[int], Optional[List[torch.Tensor]], torch.Tensor]:
        bsz, seq_len = input_ids.shape
        if cache_position is not None:
            past_len = cache_position[0].item()
        else:
            past_len = 0 if past_key_values is None else past_key_values[0][0].size(2)

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0)
            position_ids = position_ids.expand(bsz, -1)

        hidden_states = self.embed_tokens(input_ids)
        kv_len = past_len + seq_len
        attn_mask = self._prepare_attention_mask(
            attention_mask, bsz, seq_len, kv_len, hidden_states.device, hidden_states.dtype
        )

        next_cache: List[PastKeyValue] = []
        all_hidden_states = [] if return_hidden_states else None
        actual_exit_layer = len(self.layers)
        total_moe_loss = torch.zeros((), device=hidden_states.device)

        for idx, layer in enumerate(self.layers):
            if return_hidden_states:
                all_hidden_states.append(hidden_states)

            past = None if past_key_values is None else past_key_values[idx]
            hidden_states, present, moe_loss = layer(
                hidden_states, attn_mask, position_ids, past, use_cache, cache_position
            )
            total_moe_loss = total_moe_loss + moe_loss
            
            if use_cache and present is not None:
                next_cache.append(present)

            if output_layer is not None and idx == output_layer:
                actual_exit_layer = idx + 1
                break

        final_hidden = self.norm(hidden_states)
        if return_hidden_states:
            all_hidden_states.append(final_hidden)

        return final_hidden, next_cache if use_cache else None, actual_exit_layer, all_hidden_states, total_moe_loss


class CPULiteForCausalLM(CPULitePreTrainedModel):
    def __init__(self, config: CPULiteConfig) -> None:
        super().__init__(config)
        self.model = CPULiteModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_head = (
            nn.Linear(config.hidden_size, config.carp_router_labels, bias=True)
            if getattr(config, "carp_router_labels", 0) > 0
            else None
        )
        self.verifier_head = (
            nn.Linear(config.hidden_size, config.carp_verifier_labels, bias=True)
            if getattr(config, "carp_verifier_labels", 0) > 0
            else None
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def allocate_kv_cache(
        self,
        batch_size: int,
        max_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[List[torch.Tensor]]:
        cache_device = device if device is not None else next(self.parameters()).device
        cache_dtype = dtype if dtype is not None else self.dtype
        return [
            [
                torch.zeros(
                    (
                        batch_size,
                        self.config.num_key_value_heads,
                        max_length,
                        self.config.head_dim,
                    ),
                    device=cache_device,
                    dtype=cache_dtype,
                ),
                torch.zeros(
                    (
                        batch_size,
                        self.config.num_key_value_heads,
                        max_length,
                        self.config.head_dim,
                    ),
                    device=cache_device,
                    dtype=cache_dtype,
                ),
            ]
            for _ in range(self.config.num_hidden_layers)
        ]

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[PastKeyValue]] = None,
        use_cache: bool = False,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        output_layer: Optional[int] = None,
        multi_exit_loss: bool = False,
        logits_to_keep: int = 0,
    ) -> CPULiteCausalLMOutput:
        hidden, cache, _, all_hidden, moe_loss = self.model(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            cache_position,
            output_layer=output_layer,
            return_hidden_states=multi_exit_loss,
        )
        logits_hidden = hidden[:, -logits_to_keep:, :] if logits_to_keep > 0 else hidden
        logits = self.lm_head(logits_hidden)
        loss = None
        if labels is not None:
            if logits_to_keep > 0:
                raise ValueError("logits_to_keep cannot be used when labels are provided.")
            # Standard Loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            # Add MoE auxiliary loss
            if moe_loss > 0:
                loss = loss + self.config.moe_loss_weight * moe_loss

            # Multi-Exit Loss: Encourage intermediate layers to be useful
            if multi_exit_loss and all_hidden is not None:
                num_inter_layers = len(all_hidden) - 2
                if num_inter_layers > 0:
                    inter_loss_weight = 0.3 / num_inter_layers
                    for i in range(1, len(all_hidden) - 1):
                        inter_hidden = self.model.norm(all_hidden[i])
                        inter_logits = self.lm_head(inter_hidden)
                        shift_inter_logits = inter_logits[:, :-1, :].contiguous()
                        inter_loss = F.cross_entropy(
                            shift_inter_logits.view(-1, self.config.vocab_size),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        )
                        loss = loss + inter_loss_weight * inter_loss
                        del inter_hidden, inter_logits, shift_inter_logits

        return CPULiteCausalLMOutput(loss=loss, logits=logits, past_key_values=cache, moe_loss=moe_loss)

    def carp_heads(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> CPULiteCARPHeadOutput:
        hidden, _, _, _, _ = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        pooled = hidden[:, -1, :]
        router_logits = self.router_head(pooled) if self.router_head is not None else None
        verifier_logits = self.verifier_head(pooled) if self.verifier_head is not None else None
        return CPULiteCARPHeadOutput(router_logits=router_logits, verifier_logits=verifier_logits)

    def resize_token_embeddings(self, new_size: int) -> nn.Embedding:
        if new_size <= 0:
            raise ValueError("new_size must be positive")
        old_embed = self.model.embed_tokens
        old_size, hidden_size = old_embed.weight.shape
        if new_size == old_size:
            return old_embed
        new_embed = nn.Embedding(new_size, hidden_size, self.config.pad_token_id)
        self._init_weights(new_embed)
        copy_size = min(old_size, new_size)
        with torch.no_grad():
            new_embed.weight[:copy_size] = old_embed.weight[:copy_size]
        self.model.embed_tokens = new_embed.to(device=old_embed.weight.device, dtype=old_embed.weight.dtype)
        self.config.vocab_size = new_size
        if self.config.tie_word_embeddings:
            self.lm_head = nn.Linear(hidden_size, new_size, bias=False).to(
                device=old_embed.weight.device,
                dtype=old_embed.weight.dtype,
            )
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            old_head = self.lm_head
            new_head = nn.Linear(hidden_size, new_size, bias=False)
            self._init_weights(new_head)
            with torch.no_grad():
                new_head.weight[:copy_size] = old_head.weight[:copy_size]
            self.lm_head = new_head.to(device=old_head.weight.device, dtype=old_head.weight.dtype)
        return self.model.embed_tokens

    @torch.no_grad()
    def generate_simple(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        use_cache: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        self.eval()
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        device = input_ids.device
        bsz = input_ids.size(0)
        prompt_len = input_ids.size(1)
        max_len = prompt_len + max_new_tokens
        generated = torch.empty((bsz, max_len), dtype=input_ids.dtype, device=device)
        generated[:, :prompt_len] = input_ids
        
        past = None
        if use_cache:
            past = self.allocate_kv_cache(bsz, max_len, device=device)
        
        cur_pos = 0
        out_len = prompt_len
        next_input = input_ids
        
        for _ in range(max_new_tokens):
            model_input = next_input if use_cache else generated[:, :out_len]
            cache_pos = torch.arange(cur_pos, cur_pos + model_input.size(1), device=device)
            out = self(
                model_input,
                past_key_values=past,
                use_cache=use_cache,
                cache_position=cache_pos if use_cache else None,
                logits_to_keep=1,
            )
            cur_pos += model_input.size(1) if use_cache else 0
            logits = out.logits[:, -1, :]
            
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            generated[:, out_len : out_len + 1] = next_token
            out_len += 1
            next_input = next_token
            
            if eos is not None and bool((next_token == eos).all()):
                break
                
        return generated[:, :out_len]

    @torch.no_grad()
    def generate_streaming(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        use_cache: bool = True,
        eos_token_id: Optional[int] = None,
    ):
        self.eval()
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        device = input_ids.device
        bsz = input_ids.size(0)
        
        past = None
        if use_cache:
            max_len = input_ids.size(1) + max_new_tokens
            past = self.allocate_kv_cache(bsz, max_len, device=device)
        
        cur_pos = 0
        next_input = input_ids
        emitted: List[torch.Tensor] = []
        
        for i in range(max_new_tokens):
            model_input = next_input if use_cache else input_ids if i == 0 else torch.cat([input_ids, *emitted], dim=1)
            cache_pos = torch.arange(cur_pos, cur_pos + model_input.size(1), device=device)
            out = self(
                model_input,
                past_key_values=past,
                use_cache=use_cache,
                cache_position=cache_pos if use_cache else None,
                logits_to_keep=1,
            )
            
            cur_pos += model_input.size(1) if use_cache else 0
            logits = out.logits[:, -1, :]
            
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            yield next_token
            
            next_input = next_token
            if not use_cache:
                emitted.append(next_token)
            if eos is not None and bool((next_token == eos).all()):
                break

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        self.config.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, path: str | Path, *args: Any, **kwargs: Any) -> "CPULiteForCausalLM":
        config = CPULiteConfig.from_pretrained(path)
        model = cls(config)
        state_path = Path(path) / "pytorch_model.bin"
        if not state_path.exists():
            state_path = Path(path) / "model.pt"
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
        return model
