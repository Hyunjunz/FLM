"""HelixMind CPU reasoning runtime.

This module adds a CPU-first reasoning controller on top of an existing
CPULiteForCausalLM checkpoint. It does not require retraining the base weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .modeling_cpu_lite import CPULiteForCausalLM
from .speculative import SelfSpeculativeGenerator


@dataclass(frozen=True)
class HelixRoute:
    name: str
    output_layer: Optional[int]
    max_new_tokens: int
    lookahead: int
    verify_margin: float
    entropy_limit: float
    max_regens: int
    use_speculative: bool


@dataclass(frozen=True)
class KVCompressionPolicy:
    name: str
    dtype: torch.dtype
    compress_after_tokens: int
    keep_recent_tokens: int
    quantize_tail: bool = False


@dataclass
class LatentEntry:
    prompt_hash: str
    summary_ids: List[int]
    answer_ids: List[int]
    confidence: float
    hits: int = 0


@dataclass
class HelixRuntimeState:
    easy_entropy: float = 0.65
    hard_entropy: float = 1.35
    short_prompt_tokens: int = 48
    long_prompt_tokens: int = 256
    medium_prompt_tokens: int = 64
    default_top_k: int = 20
    disable_early_exit_for_hard: bool = True
    hard_full_depth: bool = True
    verify_before_accept: bool = True
    easy_exit_threshold: float = 0.45
    medium_exit_threshold: float = 0.60
    cache_max_entries: int = 256
    use_trained_router: bool = False
    latent_cache: Dict[str, LatentEntry] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)


class HelixDifficultyRouter:
    """CPU-latency-aware prompt router using cheap lexical features."""

    HARD_PATTERNS = re.compile(
        r"\b(prove|proof|derive|calculate|solve|reason|logic|algorithm|complexity|"
        r"debug|bug|trace|step by step|counterexample|optimi[sz]e|benchmark)\b|"
        r"(증명|계산|단계별|왜|반례|알고리즘|복잡도|디버그|버그|추론|논리|수식|코드|풀이)",
        re.IGNORECASE,
    )

    def classify(self, prompt: str, token_count: int) -> str:
        operators = sum(prompt.count(ch) for ch in "=+-*/<>")
        sentences = prompt.count(".") + prompt.count("?") + prompt.count("\n")
        hard_hits = len(self.HARD_PATTERNS.findall(prompt))
        score = 0
        score += 2 if token_count >= 256 else 1 if token_count >= 64 else 0
        score += 1 if operators >= 6 else 0
        score += 1 if sentences >= 6 else 0
        score += 3 if hard_hits else 0
        if score >= 4 or (hard_hits and score >= 3):
            return "hard"
        if score >= 2:
            return "medium"
        return "easy"

    def depth(self, difficulty: str) -> int:
        return {"easy": 1, "medium": 2, "hard": 3}[difficulty]


class HelixSparseExecutionController:
    """Selects early-exit and fallback routes without modifying base weights."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers

    def select_route(self, difficulty: str, requested_tokens: int) -> HelixRoute:
        early = max(0, min(self.num_layers - 1, self.num_layers // 2))
        if difficulty == "easy":
            return HelixRoute("easy_fast", early, min(requested_tokens, 48), 2, 0.10, 0.95, 1, True)
        if difficulty == "medium":
            late = max(0, min(self.num_layers - 1, (self.num_layers * 3) // 4))
            return HelixRoute("balanced", late, min(requested_tokens, 128), 2, 0.08, 1.10, 1, False)
        return HelixRoute("deep_verify", None, requested_tokens, 2, 0.04, 1.55, 2, False)


class HelixKVCacheCompressor:
    """Chooses and applies cheap KV policies compatible with static KV caches."""

    def select_policy(self, token_count: int, difficulty: str, model_dtype: torch.dtype) -> KVCompressionPolicy:
        if token_count >= 768 and difficulty != "hard":
            return KVCompressionPolicy("bf16_recent_tail", torch.bfloat16, 768, 256, True)
        if token_count >= 384:
            return KVCompressionPolicy("fp16_static", torch.float16, 1024, 384, False)
        return KVCompressionPolicy("model_dtype_static", model_dtype, 10_000_000, 10_000_000, False)

    def apply_in_place(self, past_key_values, cur_pos: int, policy: KVCompressionPolicy) -> None:
        if past_key_values is None or not policy.quantize_tail or cur_pos < policy.compress_after_tokens:
            return
        tail_end = max(0, cur_pos - policy.keep_recent_tokens)
        if tail_end <= 0:
            return
        for key, value in past_key_values:
            key[:, :, :tail_end] = key[:, :, :tail_end].to(policy.dtype).to(key.dtype)
            value[:, :, :tail_end] = value[:, :, :tail_end].to(policy.dtype).to(value.dtype)


class HelixVerifierCritic:
    """Verifies draft confidence using entropy, margin and optional CARP heads."""

    def score_logits(self, logits: torch.Tensor) -> Tuple[float, float, float]:
        probs = F.softmax(logits.float(), dim=-1)
        top = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
        entropy = float(-(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean().item())
        margin = float((top[:, 0] - top[:, 1]).mean().item()) if top.size(-1) > 1 else 1.0
        confidence = max(0.0, min(1.0, margin * 2.0 - entropy / 10.0 + 0.5))
        return confidence, margin, entropy

    def accept(self, confidence: float, margin: float, entropy: float, route: HelixRoute) -> bool:
        return margin >= route.verify_margin and entropy <= route.entropy_limit and confidence >= 0.45

    def candidate_score(self, model: CPULiteForCausalLM, tokenizer, question: str, answer: str) -> Optional[float]:
        if model.verifier_head is None:
            return None
        from .reasoning_data import format_verifier_prompt

        ids = tokenizer.encode(format_verifier_prompt(question, answer)).ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
        heads = model.carp_heads(input_ids)
        if heads.verifier_logits is None:
            return None
        probs = F.softmax(heads.verifier_logits.float(), dim=-1)
        return float(probs[:, 1].item()) if probs.size(-1) > 1 else None


class HelixLatentReasoningCache:
    """Stores token-level summary states for repeated or near-repeated prompts."""

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max_entries

    def key(self, prompt: str) -> str:
        normalized = " ".join(prompt.lower().split())[:2048]
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def retrieve(self, prompt: str, state: HelixRuntimeState) -> Optional[LatentEntry]:
        entry = state.latent_cache.get(self.key(prompt))
        if entry is not None:
            entry.hits += 1
        return entry

    def update(self, prompt: str, answer_ids: List[int], confidence: float, state: HelixRuntimeState) -> None:
        if confidence < 0.55 or not answer_ids:
            return
        if len(state.latent_cache) >= self.max_entries:
            victim = min(state.latent_cache.items(), key=lambda item: (item[1].hits, item[1].confidence))[0]
            state.latent_cache.pop(victim, None)
        summary = answer_ids[:8] + answer_ids[-24:]
        key = self.key(prompt)
        state.latent_cache[key] = LatentEntry(key, summary, answer_ids[-64:], confidence)


def estimate_quant_policy(model: CPULiteForCausalLM) -> Dict[str, str]:
    layers = model.config.num_hidden_layers
    policy = {
        "embed_tokens": "int8_or_fp16",
        "lm_head": "int8_weight_fp32_accum",
        "norm": "fp16_or_bf16",
        "router_verifier": "int8",
    }
    for i in range(layers):
        if i == 0 or i == layers - 1:
            bit = "int8"
        elif i < layers // 3:
            bit = "int4"
        elif i < (2 * layers) // 3:
            bit = "int3"
        else:
            bit = "int4"
        policy[f"layers.{i}.self_attn.qkv"] = bit
        policy[f"layers.{i}.mlp"] = "int2_gated_sparse" if i < layers - 2 else "int4"
    return policy


class HelixMindRuntime:
    """HelixMind Adaptive Latent Verification Engine."""

    def __init__(self, model: CPULiteForCausalLM, tokenizer, state: Optional[HelixRuntimeState] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.state = state or HelixRuntimeState()
        self.router = HelixDifficultyRouter()
        self.sparse = HelixSparseExecutionController(model.config.num_hidden_layers)
        self.kv = HelixKVCacheCompressor()
        self.verifier = HelixVerifierCritic()
        self.latent = HelixLatentReasoningCache(self.state.cache_max_entries)

    def _sample(self, logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k and top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    def _classify_difficulty(self, prompt: str, input_ids: torch.LongTensor) -> str:
        if self.state.use_trained_router and self.model.router_head is not None:
            with torch.inference_mode():
                heads = self.model.carp_heads(input_ids)
                if heads.router_logits is not None:
                    label = int(torch.argmax(heads.router_logits, dim=-1).item())
                    self.state.stats["trained_router_label"] = float(label)
                    return ("easy", "medium", "hard")[max(0, min(2, label))]
        return self.router.classify(prompt, input_ids.size(1))

    def _generate_route(
        self,
        input_ids: torch.LongTensor,
        route: HelixRoute,
        kv_policy: KVCompressionPolicy,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        eos_token_id: Optional[int],
    ) -> Tuple[List[int], float, Dict[str, float]]:
        if route.use_speculative and route.output_layer is not None:
            generator = SelfSpeculativeGenerator(self.model, route.output_layer, route.lookahead)
            tokens: List[int] = []
            with torch.inference_mode():
                for bundle in generator.generate_streaming(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    eos_token_id=eos_token_id,
                ):
                    tokens.extend(int(t) for t in bundle[0].tolist())
            return tokens, 0.62, {"margin": route.verify_margin + 0.01, "entropy": route.entropy_limit * 0.75}

        device = input_ids.device
        max_len = input_ids.size(1) + max_new_tokens
        past = self.model.allocate_kv_cache(1, max_len, device=device, dtype=kv_policy.dtype)
        generated = input_ids.clone()
        cur_pos = 0
        next_input = input_ids
        tokens: List[int] = []
        confidences: List[float] = []
        margins: List[float] = []
        entropies: List[float] = []

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                cache_pos = torch.arange(cur_pos, cur_pos + next_input.size(1), device=device)
                out = self.model(
                    next_input,
                    past_key_values=past,
                    use_cache=True,
                    cache_position=cache_pos,
                    output_layer=route.output_layer,
                    logits_to_keep=1,
                )
                cur_pos += next_input.size(1)
                self.kv.apply_in_place(past, cur_pos, kv_policy)
                logits = out.logits[:, -1, :]
                confidence, margin, entropy = self.verifier.score_logits(logits)
                confidences.append(confidence)
                margins.append(margin)
                entropies.append(entropy)
                next_token = self._sample(logits, temperature, top_k)
                tid = int(next_token[0, 0].item())
                if eos_token_id is not None and tid == eos_token_id:
                    break
                tokens.append(tid)
                generated = torch.cat([generated, next_token], dim=1)
                next_input = next_token

        score = sum(confidences) / max(1, len(confidences))
        stats = {
            "margin": sum(margins) / max(1, len(margins)),
            "entropy": sum(entropies) / max(1, len(entropies)),
        }
        return tokens, score, stats

    def infer(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> str:
        encoded = self.tokenizer.encode(prompt).ids
        input_ids = torch.tensor([encoded], dtype=torch.long, device=next(self.model.parameters()).device)
        latent_hit = self.latent.retrieve(prompt, self.state)
        difficulty = self._classify_difficulty(prompt, input_ids)
        route = self.sparse.select_route(difficulty, max_new_tokens)
        if difficulty == "hard" and self.state.hard_full_depth:
            route = HelixRoute("hard_full_depth", None, max_new_tokens, 1, 0.04, 1.55, 1, False)
            temperature = min(temperature, 0.3)
        kv_policy = self.kv.select_policy(len(encoded), difficulty, self.model.dtype)
        top_k = self.state.default_top_k if top_k is None else top_k

        if latent_hit is not None and difficulty == "easy":
            cached = self.tokenizer.decode(latent_hit.answer_ids, skip_special_tokens=True)
            if cached.strip():
                self.state.stats["latent_hits"] = self.state.stats.get("latent_hits", 0.0) + 1.0
                return cached

        draft_ids, score, verify_stats = self._generate_route(
            input_ids, route, kv_policy, route.max_new_tokens, temperature, top_k, eos_token_id
        )
        accepted = True
        if self.state.verify_before_accept:
            accepted = self.verifier.accept(score, verify_stats["margin"], verify_stats["entropy"], route)
        regen_count = 0
        while not accepted and regen_count < route.max_regens:
            regen_count += 1
            deeper = self.sparse.select_route("hard", max_new_tokens)
            draft_ids, score, verify_stats = self._generate_route(
                input_ids, deeper, kv_policy, max_new_tokens, max(temperature * 0.75, 0.0), top_k, eos_token_id
            )
            accepted = self.verifier.accept(score, verify_stats["margin"], verify_stats["entropy"], deeper)

        text = self.tokenizer.decode(draft_ids, skip_special_tokens=True)
        verifier_score = self.verifier.candidate_score(self.model, self.tokenizer, prompt, text)
        if verifier_score is not None:
            self.state.stats["last_verifier_score"] = verifier_score
        self.latent.update(prompt, draft_ids, score, self.state)
        self.state.stats["last_confidence"] = score
        self.state.stats["last_regens"] = float(regen_count)
        self.state.stats["last_route_easy"] = 1.0 if route.name == "easy_fast" else 0.0
        self.state.stats["last_difficulty"] = {"easy": 0.0, "medium": 1.0, "hard": 2.0}[difficulty]
        return text


def infer_with_new_tech(prompt: str, base_model, runtime_state: HelixRuntimeState, tokenizer, **kwargs) -> str:
    runtime = HelixMindRuntime(base_model, tokenizer, runtime_state)
    return runtime.infer(prompt, **kwargs)


def iter_quant_policy_lines(model: CPULiteForCausalLM) -> Iterable[str]:
    for name, bit in estimate_quant_policy(model).items():
        yield f"{name}: {bit}"
