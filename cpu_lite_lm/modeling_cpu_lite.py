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


class CPULiteRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
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
        cos = self.cos_cached.to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached.to(device=x.device, dtype=x.dtype)
        cos = cos[0, 0, position_ids, :].unsqueeze(1)
        sin = sin[0, 0, position_ids, :].unsqueeze(1)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class CPULiteAttention(nn.Module):
    def __init__(self, config: CPULiteConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = CPULiteRotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def _shape(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PastKeyValue]]:
        bsz, q_len, _ = hidden_states.shape
        query = self._shape(self.q_proj(hidden_states), self.num_heads)
        key = self._shape(self.k_proj(hidden_states), self.num_kv_heads)
        value = self._shape(self.v_proj(hidden_states), self.num_kv_heads)
        cos, sin = self.rotary_emb(value, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        present = (key, value) if use_cache else None

        if self.num_kv_groups > 1:
            key_for_attn = key.repeat_interleave(self.num_kv_groups, dim=1)
            value_for_attn = value.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            key_for_attn = key
            value_for_attn = value

        attn_weights = torch.matmul(query, key_for_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)
        kv_len = key_for_attn.size(-2)
        if attention_mask is None:
            past_len = kv_len - q_len
            q_pos = torch.arange(past_len, past_len + q_len, device=hidden_states.device)[:, None]
            k_pos = torch.arange(kv_len, device=hidden_states.device)[None, :]
            causal = k_pos <= q_pos
            attn_weights = attn_weights.masked_fill(~causal[None, None, :, :], torch.finfo(attn_weights.dtype).min)
        else:
            attn_weights = attn_weights + attention_mask
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


class CPULiteDecoderLayer(nn.Module):
    def __init__(self, config: CPULiteConfig) -> None:
        super().__init__()
        self.input_layernorm = CPULiteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CPULiteAttention(config)
        self.post_attention_layernorm = CPULiteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = CPULiteMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[PastKeyValue],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[PastKeyValue]]:
        residual = hidden_states
        attn_out, present = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class CPULitePreTrainedModel(PreTrainedModel):
    config_class = CPULiteConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
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
    ) -> Tuple[torch.Tensor, Optional[List[PastKeyValue]]]:
        bsz, seq_len = input_ids.shape
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
        for idx, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[idx]
            hidden_states, present = layer(hidden_states, attn_mask, position_ids, past, use_cache)
            if use_cache and present is not None:
                next_cache.append(present)
        return self.norm(hidden_states), next_cache if use_cache else None


class CPULiteForCausalLM(CPULitePreTrainedModel):
    def __init__(self, config: CPULiteConfig) -> None:
        super().__init__(config)
        self.model = CPULiteModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[PastKeyValue]] = None,
        use_cache: bool = False,
        labels: Optional[torch.LongTensor] = None,
    ) -> CPULiteCausalLMOutput:
        hidden, cache = self.model(input_ids, attention_mask, position_ids, past_key_values, use_cache)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CPULiteCausalLMOutput(loss=loss, logits=logits, past_key_values=cache)

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
        generated = input_ids.clone()
        past = None
        next_input = generated
        for _ in range(max_new_tokens):
            out = self(next_input, past_key_values=past, use_cache=use_cache)
            past = out.past_key_values if use_cache else None
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
            generated = torch.cat([generated, next_token], dim=1)
            next_input = next_token if use_cache else generated
            if eos is not None and bool((next_token == eos).all()):
                break
        return generated

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
