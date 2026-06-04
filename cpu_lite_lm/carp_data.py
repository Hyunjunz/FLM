"""Dataset formatting helpers for CARP training data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from .carp import DifficultyLevel, reasoning_token_strings


DIFFICULTY_TO_ID = {
    "easy": int(DifficultyLevel.EASY),
    "medium": int(DifficultyLevel.MEDIUM),
    "hard": int(DifficultyLevel.HARD),
    "critical": int(DifficultyLevel.CRITICAL),
}


@dataclass(frozen=True)
class CARPTraceExample:
    question: str
    answer: str
    reasoning_tokens: List[str]
    difficulty: str = "medium"
    verifier_required: bool = False
    candidate_count: int = 1


def normalize_reasoning_tokens(tokens: Sequence[str] | None, max_tokens: int) -> List[str]:
    allowed = set(reasoning_token_strings(max_tokens))
    normalized: List[str] = []
    for token in tokens or []:
        text = str(token).strip()
        if text in allowed:
            normalized.append(text)
    return normalized


def parse_carp_trace(example: Dict[str, Any], max_reasoning_tokens: int = 128) -> CARPTraceExample:
    question = _first_present(example, ["question", "prompt", "instruction", "query", "input"])
    answer = _first_present(example, ["answer", "final_answer", "output", "response", "completion"])
    raw_tokens = example.get("reasoning_tokens") or example.get("r_tokens") or []
    if isinstance(raw_tokens, str):
        raw_tokens = raw_tokens.split()
    difficulty = str(example.get("difficulty", example.get("difficulty_level", "medium"))).lower()
    if difficulty not in DIFFICULTY_TO_ID:
        difficulty = "medium"
    return CARPTraceExample(
        question=question.strip(),
        answer=answer.strip(),
        reasoning_tokens=normalize_reasoning_tokens(raw_tokens, max_reasoning_tokens),
        difficulty=difficulty,
        verifier_required=bool(example.get("verifier_required", difficulty in {"hard", "critical"})),
        candidate_count=int(example.get("candidate_count", 1 if difficulty in {"easy", "medium"} else 3)),
    )


def build_carp_instruction_text(trace: CARPTraceExample) -> tuple[str, str]:
    prompt = f"### Question:\n{trace.question.strip()}\n\n"
    if trace.reasoning_tokens:
        prompt += "### Reasoning Tokens:\n" + " ".join(trace.reasoning_tokens) + "\n\n"
    prompt += "### Answer:\n"
    return prompt, trace.answer.strip()


def build_router_label(trace: CARPTraceExample) -> Dict[str, int | bool]:
    return {
        "difficulty": DIFFICULTY_TO_ID[trace.difficulty],
        "verifier_required": trace.verifier_required,
        "candidate_count": trace.candidate_count,
        "reasoning_budget": len(trace.reasoning_tokens),
    }


def build_verifier_rows(question: str, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        text = str(candidate.get("answer", candidate.get("text", ""))).strip()
        if not text:
            continue
        label = candidate.get("label", candidate.get("score", 0))
        if isinstance(label, str):
            label_id = 1 if label.lower() in {"correct", "best", "chosen", "safe"} else 0
        else:
            label_id = int(label)
        rows.append({"question": question.strip(), "answer": text, "label": label_id})
    return rows


class CARPJsonlSFTDataset(Dataset):
    """JSONL dataset for CARP SFT traces.

    Rows can be either raw traces with question/answer/reasoning_tokens or
    preformatted rows with a text field and router_label.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        block_size: int = 256,
        max_reasoning_tokens: int = 128,
        max_examples: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.rows: List[Dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if "text" in raw:
                    text = str(raw["text"])
                    router = raw.get("router_label", {})
                else:
                    trace = parse_carp_trace(raw, max_reasoning_tokens=max_reasoning_tokens)
                    prompt, answer = build_carp_instruction_text(trace)
                    text = prompt + answer
                    router = build_router_label(trace)
                if text.strip():
                    self.rows.append({"text": text, "router_label": router})
                if max_examples is not None and len(self.rows) >= max_examples:
                    break
        if not self.rows:
            raise ValueError(f"No CARP SFT rows loaded from {path}")
        self.eos_id = tokenizer.token_to_id("<eos>")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        ids = self.tokenizer.encode(row["text"]).ids[: self.block_size]
        if self.eos_id is not None and len(ids) < self.block_size:
            ids.append(self.eos_id)
        if len(ids) < 2:
            ids = ids + [self.eos_id or 2]
        router = row.get("router_label") or {}
        difficulty = int(router.get("difficulty", DIFFICULTY_TO_ID["medium"]))
        verifier_required = int(bool(router.get("verifier_required", False)))
        candidate_count = int(router.get("candidate_count", 1))
        reasoning_budget = int(router.get("reasoning_budget", 0))
        x = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids": x,
            "labels": x.clone(),
            "router_difficulty": torch.tensor(difficulty, dtype=torch.long),
            "router_verifier_required": torch.tensor(verifier_required, dtype=torch.float),
            "router_candidate_count": torch.tensor(candidate_count, dtype=torch.float),
            "router_reasoning_budget": torch.tensor(reasoning_budget, dtype=torch.float),
        }


def collate_carp_sft(batch: Sequence[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for idx, item in enumerate(batch):
        n = item["input_ids"].numel()
        input_ids[idx, :n] = item["input_ids"]
        labels[idx, :n] = item["labels"]
        attention_mask[idx, :n] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "router_difficulty": torch.stack([item["router_difficulty"] for item in batch]),
        "router_verifier_required": torch.stack([item["router_verifier_required"] for item in batch]),
        "router_candidate_count": torch.stack([item["router_candidate_count"] for item in batch]),
        "router_reasoning_budget": torch.stack([item["router_reasoning_budget"] for item in batch]),
    }


def _first_present(example: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = example.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""
