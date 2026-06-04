
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
    candidates: List[str] | None = None
    gold_label: str = ""


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
        candidates=[str(item) for item in example.get("candidates", [])] or None,
        gold_label=str(example.get("gold_label", "")).strip(),
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

    Rows can be:
    - text-only LM rows: {"text": "..."}
    - prompt/answer SFT rows: {"prompt": "...", "answer": "...", "text": "..."}
    - choice/CARP rows with candidates/gold_label for ranking.
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
                    text = str(raw.get("text", ""))
                    prompt_text = str(raw.get("prompt", ""))
                    answer_text = str(raw.get("answer", ""))
                    router = raw.get("router_label", {})
                    candidates = [str(item) for item in raw.get("candidates", [])]
                    gold_label = str(raw.get("gold_label", "")).strip()

                    # If prompt+answer exist, reconstruct text so the supervised
                    # answer span is not accidentally empty or mismatched.
                    if prompt_text and answer_text:
                        text = prompt_text + answer_text
                else:
                    trace = parse_carp_trace(raw, max_reasoning_tokens=max_reasoning_tokens)
                    prompt_text, answer_text = build_carp_instruction_text(trace)
                    text = prompt_text + answer_text
                    router = build_router_label(trace)
                    candidates = trace.candidates or []
                    gold_label = trace.gold_label

                if text.strip():
                    self.rows.append(
                        {
                            "text": text,
                            "prompt": prompt_text,
                            "answer": answer_text,
                            "router_label": router,
                            "candidates": candidates,
                            "gold_label": gold_label,
                        }
                    )
                if max_examples is not None and len(self.rows) >= max_examples:
                    break

        if not self.rows:
            raise ValueError(f"No CARP SFT rows loaded from {path}")

        self.eos_id = tokenizer.token_to_id("<eos>")
        if self.eos_id is None:
            self.eos_id = 2

    def __len__(self) -> int:
        return len(self.rows)

    def _encode_with_eos(self, text: str, max_len: int | None = None) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if max_len is not None:
            ids = ids[:max_len]
        if self.eos_id is not None and (not ids or ids[-1] != self.eos_id):
            if max_len is None or len(ids) < max_len:
                ids.append(self.eos_id)
        return ids

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        text = str(row["text"])
        prompt_text = str(row.get("prompt") or "")
        answer_text = str(row.get("answer") or "")

        # Prompt/answer rows: mask prompt labels, supervise answer labels.
        # Text-only rows: ordinary LM over the whole text.
        if prompt_text and answer_text:
            full_text = prompt_text + answer_text
            full_ids = self._encode_with_eos(full_text, self.block_size)
            prompt_ids_raw = self.tokenizer.encode(prompt_text).ids

            labels = list(full_ids)
            mask_n = min(len(prompt_ids_raw), len(labels))
            for pos in range(mask_n):
                labels[pos] = -100

            # If truncation removed the full answer supervision, keep at least
            # the final token supervised if possible rather than training "empty answer".
            if all(label == -100 for label in labels) and labels:
                labels[-1] = full_ids[-1]
        else:
            full_ids = self._encode_with_eos(text, self.block_size)
            labels = list(full_ids)

        if len(full_ids) < 2:
            full_ids = full_ids + [self.eos_id or 2]
            labels = labels + [self.eos_id or 2]

        router = row.get("router_label") or {}
        difficulty = int(router.get("difficulty", DIFFICULTY_TO_ID["medium"]))
        verifier_required = int(bool(router.get("verifier_required", False)))
        candidate_count = int(router.get("candidate_count", 1))
        reasoning_budget = int(router.get("reasoning_budget", 0))

        x = torch.tensor(full_ids, dtype=torch.long)
        item: Dict[str, Any] = {
            "input_ids": x,
            "labels": torch.tensor(labels, dtype=torch.long),
            "router_difficulty": torch.tensor(difficulty, dtype=torch.long),
            "router_verifier_required": torch.tensor(verifier_required, dtype=torch.float),
            "router_candidate_count": torch.tensor(candidate_count, dtype=torch.float),
            "router_reasoning_budget": torch.tensor(reasoning_budget, dtype=torch.float),
        }

        candidates = row.get("candidates") or []
        if candidates:
            # Use the same prompt used in LM supervision.
            prompt_for_rank = prompt_text
            if not prompt_for_rank:
                # Fallback: use text before the answer marker if possible.
                marker = "### Answer:\n"
                if marker in text:
                    prompt_for_rank = text.split(marker, 1)[0] + marker
            prompt_ids = self.tokenizer.encode(prompt_for_rank).ids[: self.block_size]
            candidate_ids = [self.tokenizer.encode(str(candidate)).ids[:64] for candidate in candidates]

            gold_label = str(row.get("gold_label") or "").strip()
            gold_idx = 0
            for cand_idx, candidate in enumerate(candidates):
                cand_text = str(candidate).strip()
                if gold_label and cand_text.startswith(f"{gold_label}."):
                    gold_idx = cand_idx
                    break
                if gold_label and cand_text.lower() == gold_label.lower():
                    gold_idx = cand_idx
                    break

            item["prompt_ids"] = torch.tensor(prompt_ids or [self.eos_id or 2], dtype=torch.long)
            item["candidate_ids"] = [
                torch.tensor(ids or [self.eos_id or 2], dtype=torch.long) for ids in candidate_ids
            ]
            item["gold_choice"] = torch.tensor(gold_idx, dtype=torch.long)
        return item


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

    out = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "router_difficulty": torch.stack([item["router_difficulty"] for item in batch]),
        "router_verifier_required": torch.stack([item["router_verifier_required"] for item in batch]),
        "router_candidate_count": torch.stack([item["router_candidate_count"] for item in batch]),
        "router_reasoning_budget": torch.stack([item["router_reasoning_budget"] for item in batch]),
    }
    out.update(_collate_choice_fields(batch, pad_token_id))
    return out


def _collate_choice_fields(batch: Sequence[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collate ranking fields for mixed LM + choice batches.

    Old behavior returned {} unless every sample had candidates, causing rank=0
    for mixed batches. This version keeps rank fields and marks non-choice rows
    inactive.
    """
    active_indices = [
        idx for idx, item in enumerate(batch) if "candidate_ids" in item and "prompt_ids" in item
    ]
    if not active_indices:
        return {}

    batch_size = len(batch)
    max_prompt = max(batch[idx]["prompt_ids"].numel() for idx in active_indices)
    max_choices = max(len(batch[idx]["candidate_ids"]) for idx in active_indices)
    max_candidate = max(
        candidate.numel()
        for idx in active_indices
        for candidate in batch[idx]["candidate_ids"]
    )

    prompt_ids = torch.full((batch_size, max_prompt), pad_token_id, dtype=torch.long)
    prompt_attention_mask = torch.zeros((batch_size, max_prompt), dtype=torch.long)
    candidate_ids = torch.full((batch_size, max_choices, max_candidate), pad_token_id, dtype=torch.long)
    candidate_attention_mask = torch.zeros((batch_size, max_choices, max_candidate), dtype=torch.long)
    candidate_mask = torch.zeros((batch_size, max_choices), dtype=torch.bool)
    gold_choice = torch.zeros((batch_size,), dtype=torch.long)
    rank_active = torch.zeros((batch_size,), dtype=torch.bool)

    for batch_idx in active_indices:
        item = batch[batch_idx]
        n = item["prompt_ids"].numel()
        prompt_ids[batch_idx, :n] = item["prompt_ids"]
        prompt_attention_mask[batch_idx, :n] = 1
        gold_choice[batch_idx] = item["gold_choice"]
        rank_active[batch_idx] = True

        for choice_idx, candidate in enumerate(item["candidate_ids"]):
            m = candidate.numel()
            candidate_ids[batch_idx, choice_idx, :m] = candidate
            candidate_attention_mask[batch_idx, choice_idx, :m] = 1
            candidate_mask[batch_idx, choice_idx] = True

    return {
        "prompt_ids": prompt_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "candidate_ids": candidate_ids,
        "candidate_attention_mask": candidate_attention_mask,
        "candidate_mask": candidate_mask,
        "gold_choice": gold_choice,
        "rank_active": rank_active,
    }


def _first_present(example: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = example.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""
