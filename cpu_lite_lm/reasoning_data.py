"""Reasoning SFT and verifier JSONL data helpers."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


REASONING_TOKENS = [f"<R{i}>" for i in range(256)]
VERIFIER_IGNORE_INDEX = -100


def format_reasoning_example(example: Dict[str, Any], latent_tokens: bool = False) -> tuple[str, str]:
    question = _first_present(example, ["question", "prompt", "instruction", "query", "input"])
    plan = str(example.get("plan", "") or "").strip()
    solution = str(example.get("solution", example.get("rationale", "")) or "").strip()
    answer = _first_present(example, ["answer", "final_answer", "output", "response", "completion"])

    if not solution and "text" in example:
        text = str(example["text"])
        if "### Answer:" in text:
            return text.split("### Answer:", 1)[0] + "### Answer:\n", text.split("### Answer:", 1)[1].strip()
        return "", text.strip()

    prompt = f"### Question:\n{question.strip()}\n\n"
    if latent_tokens:
        tokens = example.get("reasoning_tokens") or example.get("r_tokens") or []
        if isinstance(tokens, str):
            tokens = tokens.split()
        token_text = " ".join(str(tok) for tok in tokens if str(tok).startswith("<R"))
        if token_text:
            prompt += f"### Reasoning Tokens:\n{token_text}\n\n"
    if plan:
        prompt += f"### Plan:\n{plan}\n\n"
    if solution:
        prompt += f"### Solution:\n{solution}\n\n"
    prompt += "### Answer:\n"
    return prompt, answer.strip()


def format_verifier_prompt(question: str, candidate_answer: str, include_label: bool = False, label: int | None = None) -> str:
    text = (
        "### Question:\n"
        f"{question.strip()}\n\n"
        "### Candidate Answer:\n"
        f"{candidate_answer.strip()}\n\n"
        "### Is the candidate answer correct?\n"
    )
    if include_label and label is not None:
        text += str(int(label))
    return text


def parse_verifier_row(example: Dict[str, Any]) -> Dict[str, Any] | None:
    has_verifier_shape = any(
        key in example for key in ("candidate_answer", "draft_answer", "candidate", "verifier_label", "accepted", "correct", "is_correct")
    )
    if not has_verifier_shape:
        return None
    question = _first_present(example, ["question", "prompt", "instruction", "query", "input"])
    candidate = _first_present(example, ["candidate_answer", "draft_answer", "candidate", "answer", "response"])
    if not question or not candidate:
        return None
    label = example.get("verifier_label")
    if label is None:
        label = example.get("accepted", example.get("correct", example.get("is_correct")))
    if label is None:
        label_id = VERIFIER_IGNORE_INDEX
    elif isinstance(label, str):
        label_id = 1 if label.lower() in {"1", "true", "yes", "correct", "accepted"} else 0
    else:
        label_id = 1 if bool(label) else 0
    return {"question": question.strip(), "candidate_answer": candidate.strip(), "verifier_label": label_id}


def make_hard_negative(question: str, answer: str, kind: str = "auto") -> str:
    if kind in {"auto", "math"}:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", answer)
        if numbers:
            target = numbers[-1]
            if "." in target:
                replacement = str(float(target) + 1.0)
            else:
                replacement = str(int(target) + 1)
            return answer[: answer.rfind(target)] + replacement + answer[answer.rfind(target) + len(target) :]
    if kind in {"auto", "code"}:
        return "The code is correct; the failure is caused by the runtime environment, so no source change is needed."
    return "The candidate skips a required condition and gives an unsupported conclusion."


def build_synthetic_reasoning_samples(count: int = 64, seed: int = 1234) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    templates = [
        lambda a, b: {
            "question": f"What is {a} * {b}?",
            "plan": "Multiply directly and keep the final number separate.",
            "solution": f"{a} * {b} = {a * b}.",
            "answer": str(a * b),
            "category": "math",
        },
        lambda a, b: {
            "question": f"x + {a} = {b}. Solve for x.",
            "plan": "Subtract the constant from both sides.",
            "solution": f"x = {b} - {a} = {b - a}.",
            "answer": str(b - a),
            "category": "math",
        },
        lambda a, b: {
            "question": "다음 코드의 버그를 찾아라: for i in range(len(xs)+1): print(xs[i])",
            "plan": "반복 범위와 인덱스 접근을 확인한다.",
            "solution": "range(len(xs)+1)는 마지막에 xs[len(xs)]를 접근해 범위를 벗어난다.\n수정 코드: for i in range(len(xs)): print(xs[i])",
            "answer": "인덱스 범위 오류",
            "category": "code_debug_ko",
        },
        lambda a, b: {
            "question": f"철수는 사과 {a}개를 사고 {b}개를 더 샀다. 몇 개인가?",
            "plan": "처음 개수와 추가 개수를 더한다.",
            "solution": f"{a} + {b} = {a + b}.",
            "answer": str(a + b),
            "category": "math_ko",
        },
    ]
    for idx in range(count):
        a = rng.randint(2, 19)
        b = rng.randint(5, 31)
        rows.append(templates[idx % len(templates)](a, b))
    return rows


class ReasoningSFTJsonlDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        block_size: int = 1024,
        max_examples: Optional[int] = None,
        train_on_prompt: bool = False,
        latent_tokens: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.train_on_prompt = train_on_prompt
        self.latent_tokens = latent_tokens
        self.eos_id = tokenizer.token_to_id("<eos>") or tokenizer.token_to_id("</s>") or 2
        self.rows = list(_iter_jsonl(path))
        if max_examples is not None:
            self.rows = self.rows[:max_examples]
        if not self.rows:
            raise ValueError(f"No reasoning rows found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, answer = format_reasoning_example(self.rows[idx], latent_tokens=self.latent_tokens)
        prompt_ids = self.tokenizer.encode(prompt).ids
        answer_ids = self.tokenizer.encode(answer).ids
        eos_extra = 1
        if len(prompt_ids) + len(answer_ids) + eos_extra > self.block_size:
            max_answer = max(1, min(len(answer_ids), self.block_size // 2))
            answer_ids = answer_ids[:max_answer]
            prompt_ids = prompt_ids[-max(1, self.block_size - max_answer - eos_extra) :]
        ids = (prompt_ids + answer_ids + [self.eos_id])[: self.block_size]
        if len(ids) < 2:
            ids.append(self.eos_id)
        labels = ids.copy()
        if not self.train_on_prompt:
            labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
            if all(label == -100 for label in labels):
                labels[-1] = ids[-1]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}


def ensure_reasoning_tokens(tokenizer, mode: str, auto_add: bool = False, count: int = 128) -> int:
    if mode != "latent":
        return 0
    missing = [tok for tok in REASONING_TOKENS[:count] if tokenizer.token_to_id(tok) is None]
    if not missing:
        return 0
    if not auto_add:
        raise ValueError(
            f"Tokenizer is missing reasoning tokens such as {missing[0]}. "
            "Use plain mode or pass --auto-add-reasoning-tokens."
        )
    return tokenizer.add_special_tokens(missing)


def _iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _first_present(example: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = example.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""
