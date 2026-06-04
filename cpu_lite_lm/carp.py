"""CARP inference utilities for CPULiteLM.

This module implements the runtime side of Compressed Adaptive Reasoning Path:
heuristic difficulty routing, optional internal reasoning tokens, candidate
generation, lightweight verification, and self-speculative decoding hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .modeling_cpu_lite import CPULiteForCausalLM
from .speculative import SelfSpeculativeGenerator


class DifficultyLevel(IntEnum):
    EASY = 0
    MEDIUM = 1
    HARD = 2
    CRITICAL = 3


@dataclass(frozen=True)
class RouteDecision:
    difficulty_level: DifficultyLevel
    reasoning_budget: int
    verifier_required: bool
    candidate_count: int
    max_output_tokens: int
    risk_level: str
    confidence: float
    reason: str

    @property
    def difficulty_name(self) -> str:
        return self.difficulty_level.name.lower()


@dataclass(frozen=True)
class CandidateScore:
    text: str
    token_ids: List[int]
    score: float
    mean_logprob: float
    length_penalty: float


@dataclass(frozen=True)
class CARPGenerationResult:
    answer: str
    route: RouteDecision
    reasoning_tokens: List[str]
    candidates: List[CandidateScore]


REASONING_TOKEN_PREFIX = "<R"


def reasoning_token_strings(count: int) -> List[str]:
    if count < 0:
        raise ValueError("count must be non-negative")
    return [f"{REASONING_TOKEN_PREFIX}{idx}>" for idx in range(count)]


def add_reasoning_tokens(tokenizer, count: int) -> int:
    """Add <R0>..<R{n-1}> as special tokens and return how many were added."""
    tokens = reasoning_token_strings(count)
    if not tokens:
        return 0
    return int(tokenizer.add_special_tokens(tokens))


def save_tokenizer_with_reasoning_tokens(tokenizer, output_dir: str | Path, count: int) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    add_reasoning_tokens(tokenizer, count)
    path = output / "tokenizer.json"
    tokenizer.save(str(path))
    return path


class HeuristicDifficultyRouter:
    """A deterministic router that is useful before training a router head."""

    critical_keywords = {
        "의료",
        "법률",
        "금융",
        "투자",
        "보안",
        "해킹",
        "취약점",
        "malware",
        "exploit",
        "password",
        "diagnosis",
        "legal",
    }
    hard_keywords = {
        "deadlock",
        "race condition",
        "디버깅",
        "증명",
        "분석",
        "설계",
        "복잡",
        "수학",
        "알고리즘",
        "동시성",
        "최적화",
        "왜",
    }
    medium_keywords = {
        "비교",
        "설명",
        "요약",
        "고쳐",
        "작성",
        "코드",
        "계산",
    }

    def route(self, prompt: str, max_output_tokens: int = 128) -> RouteDecision:
        text = prompt.strip()
        lowered = text.lower()
        tokenish_len = max(1, len(text.split()))
        line_count = max(1, text.count("\n") + 1)

        if any(keyword in lowered for keyword in self.critical_keywords):
            return RouteDecision(
                DifficultyLevel.CRITICAL,
                reasoning_budget=12,
                verifier_required=True,
                candidate_count=4,
                max_output_tokens=max(max_output_tokens, 192),
                risk_level="critical",
                confidence=0.78,
                reason="critical keyword",
            )
        if (
            any(keyword in lowered for keyword in self.hard_keywords)
            or tokenish_len >= 80
            or line_count >= 8
            or lowered.count("?") >= 2
        ):
            return RouteDecision(
                DifficultyLevel.HARD,
                reasoning_budget=8,
                verifier_required=True,
                candidate_count=3,
                max_output_tokens=max(max_output_tokens, 160),
                risk_level="normal",
                confidence=0.72,
                reason="hard keyword or long prompt",
            )
        if any(keyword in lowered for keyword in self.medium_keywords) or tokenish_len >= 18:
            return RouteDecision(
                DifficultyLevel.MEDIUM,
                reasoning_budget=4,
                verifier_required=False,
                candidate_count=1,
                max_output_tokens=max_output_tokens,
                risk_level="normal",
                confidence=0.68,
                reason="medium keyword or moderate prompt",
            )
        return RouteDecision(
            DifficultyLevel.EASY,
            reasoning_budget=0,
            verifier_required=False,
            candidate_count=1,
            max_output_tokens=min(max_output_tokens, 96),
            risk_level="low",
            confidence=0.75,
            reason="short prompt",
        )


class ReasoningCompressor:
    """Selects compact internal reasoning tokens using route and prompt cues."""

    category_slots = {
        "math": 0,
        "code": 32,
        "logic": 64,
        "retrieval": 96,
        "writing": 128,
        "safety": 160,
        "uncertainty": 192,
        "korean": 224,
    }

    def __init__(self, tokenizer, num_reasoning_tokens: int = 256) -> None:
        self.tokenizer = tokenizer
        self.num_reasoning_tokens = max(0, num_reasoning_tokens)

    def select(self, prompt: str, route: RouteDecision) -> List[str]:
        budget = min(route.reasoning_budget, self.num_reasoning_tokens)
        if budget <= 0:
            return []
        lowered = prompt.lower()
        categories = self._categories(lowered, route)
        tokens: List[str] = []
        for idx in range(budget):
            category = categories[idx % len(categories)]
            base = self.category_slots.get(category, 64)  # Default to logic
            token_id = (base + idx) % self.num_reasoning_tokens
            token = f"{REASONING_TOKEN_PREFIX}{token_id}>"
            if self.tokenizer.token_to_id(token) is not None:
                tokens.append(token)
        return tokens

    def encode(self, tokens: Sequence[str]) -> List[int]:
        ids: List[int] = []
        for token in tokens:
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                ids.append(int(token_id))
        return ids

    def _categories(self, lowered: str, route: RouteDecision) -> List[str]:
        categories: List[str] = []
        # Check for Korean first to include korean slot
        if any("\uac00" <= char <= "\ud7a3" for char in lowered):
            categories.append("korean")
        
        if any(word in lowered for word in ("수학", "계산", "방정식", "math")):
            categories.append("math")
        if any(word in lowered for word in ("코드", "버그", "deadlock", "함수", "class", "python")):
            categories.append("code")
        if any(word in lowered for word in ("왜", "논리", "조건", "증명", "모순", "commonsense", "choices")):
            categories.append("logic")
        if len(lowered) > 500 or "\n" in lowered:
            categories.append("retrieval")
        if any(word in lowered for word in ("요약", "문체", "작성", "바꿔")):
            categories.append("writing")
        if route.difficulty_level == DifficultyLevel.CRITICAL:
            categories.extend(["safety", "uncertainty"])
        
        if not categories:
            categories.append("logic" if route.difficulty_level >= DifficultyLevel.HARD else "writing")
        return categories


class TinyVerifier:
    """Scores candidates with mean log probability and light instruction heuristics."""

    def __init__(self, model: CPULiteForCausalLM, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def score(
        self,
        prompt_text: str,
        candidate_text: str,
        device: Optional[torch.device] = None,
    ) -> CandidateScore:
        device = device or next(self.model.parameters()).device
        prompt_ids = self.tokenizer.encode(prompt_text).ids
        answer_ids = self.tokenizer.encode(candidate_text).ids
        if not answer_ids:
            answer_ids = [self.model.config.eos_token_id or 0]
        ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
        out = self.model(ids, use_cache=False)
        logits = out.logits[:, max(0, len(prompt_ids) - 1) : -1, :]
        labels = ids[:, len(prompt_ids) :]
        if logits.numel() == 0 or labels.numel() == 0:
            mean_logprob = -100.0
        else:
            logprobs = F.log_softmax(logits, dim=-1)
            chosen = torch.gather(logprobs, -1, labels.unsqueeze(-1)).squeeze(-1)
            mean_logprob = float(chosen.mean().item())
        length_penalty = self._length_penalty(candidate_text)
        score = mean_logprob + length_penalty + self._format_bonus(candidate_text)
        return CandidateScore(
            text=candidate_text,
            token_ids=answer_ids,
            score=score,
            mean_logprob=mean_logprob,
            length_penalty=length_penalty,
        )

    def select_best(self, prompt_text: str, candidates: Iterable[str]) -> CandidateScore:
        scored = [self.score(prompt_text, candidate) for candidate in candidates]
        if not scored:
            raise ValueError("at least one candidate is required")
        return max(scored, key=lambda item: item.score)

    def _length_penalty(self, text: str) -> float:
        length = len(text.strip())
        if length == 0:
            return -5.0
        if length < 8:
            return -0.7
        if length > 2500:
            return -1.0
        return 0.0

    def _format_bonus(self, text: str) -> float:
        stripped = text.strip()
        if not stripped:
            return -2.0
        if "모르" in stripped or "확실" in stripped:
            return 0.1
        return 0.0


class CARPGenerator:
    """End-to-end adaptive CARP generation pipeline."""

    def __init__(
        self,
        model: CPULiteForCausalLM,
        tokenizer,
        router: Optional[HeuristicDifficultyRouter] = None,
        num_reasoning_tokens: Optional[int] = None,
        draft_layer: int = 1,
        lookahead: int = 3,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.router = router or HeuristicDifficultyRouter()
        configured_tokens = getattr(model.config, "carp_num_reasoning_tokens", 0)
        self.compressor = ReasoningCompressor(
            tokenizer,
            configured_tokens if num_reasoning_tokens is None else num_reasoning_tokens,
        )
        self.verifier = TinyVerifier(model, tokenizer)
        self.draft_layer = draft_layer
        self.lookahead = lookahead

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 20,
        use_speculative: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> CARPGenerationResult:
        route = self.router.route(prompt, max_output_tokens=max_new_tokens)
        reasoning_tokens = self.compressor.select(prompt, route)
        prompt_text = self._build_prompt(prompt, reasoning_tokens)
        candidate_count = max(1, route.candidate_count)
        if route.difficulty_level <= DifficultyLevel.MEDIUM:
            candidate_count = 1

        candidates: List[str] = []
        for idx in range(candidate_count):
            candidate_temperature = temperature if idx == 0 else max(0.2, temperature + 0.15 * idx)
            candidates.append(
                self._generate_one(
                    prompt_text,
                    max_new_tokens=route.max_output_tokens,
                    temperature=candidate_temperature,
                    top_k=top_k,
                    use_speculative=use_speculative and route.difficulty_level >= DifficultyLevel.MEDIUM,
                    eos_token_id=eos_token_id,
                )
            )
        scored = [self.verifier.score(prompt_text, candidate) for candidate in candidates]
        best = max(scored, key=lambda item: item.score)
        return CARPGenerationResult(best.text, route, reasoning_tokens, scored)

    def _build_prompt(self, prompt: str, reasoning_tokens: Sequence[str]) -> str:
        if reasoning_tokens:
            internal = " ".join(reasoning_tokens)
            return f"### Question:\n{prompt.strip()}\n\n### Reasoning Tokens:\n{internal}\n\n### Answer:\n"
        return f"### Question:\n{prompt.strip()}\n\n### Answer:\n"

    @torch.no_grad()
    def _generate_one(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        use_speculative: bool,
        eos_token_id: Optional[int],
    ) -> str:
        device = next(self.model.parameters()).device
        ids = torch.tensor([self.tokenizer.encode(prompt_text).ids], dtype=torch.long, device=device)
        chunks: List[int] = []
        if use_speculative:
            generator = SelfSpeculativeGenerator(
                self.model,
                draft_layer=min(self.draft_layer, max(0, self.model.config.num_hidden_layers - 1)),
                lookahead=self.lookahead,
            )
            for token_bundle in generator.generate_streaming(
                ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_token_id,
            ):
                chunks.extend(token_bundle[0].tolist())
        else:
            for token in self.model.generate_streaming(
                ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
                eos_token_id=eos_token_id,
            ):
                chunks.extend(token[0].tolist())
        return self.tokenizer.decode(chunks, skip_special_tokens=True).strip()
