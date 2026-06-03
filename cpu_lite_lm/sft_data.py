"""SFT dataset helpers for instruction/chat fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


def load_hf_dataset(path: str | Path, split: str = "train"):
    try:
        from datasets import load_from_disk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SFT loading requires datasets. Install with `pip install datasets`.") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SFT dataset path not found: {path}")
    ds = load_from_disk(str(path))
    if hasattr(ds, "keys"):
        if split not in ds:
            raise KeyError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")
        return ds[split]
    return ds


def format_sft_example(example: Dict[str, Any]) -> tuple[str, str]:
    """Return prompt and answer strings from common SFT schemas."""
    if "messages" in example and example["messages"]:
        messages = example["messages"]
        prompt_parts: List[str] = []
        answer = ""
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role in {"assistant", "gpt"}:
                answer = content
            else:
                prompt_parts.append(f"{role}: {content}")
        prompt = "\n".join(prompt_parts).strip()
        return prompt, answer

    instruction = _first_present(example, ["instruction", "prompt", "question", "query", "input"])
    extra_input = ""
    if "instruction" in example:
        extra_input = str(example.get("input", "") or "").strip()
    answer = _first_present(example, ["output", "response", "answer", "completion", "chosen"])
    if extra_input and extra_input != instruction:
        prompt = f"{instruction}\n\n{extra_input}".strip()
    else:
        prompt = instruction.strip()
    return prompt, answer.strip()


def _first_present(example: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = example.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def build_instruction_text(prompt: str, answer: str) -> tuple[str, str]:
    prompt_text = (
        "### Question:\n"
        f"{prompt.strip()}\n\n"
        "### Answer:\n"
    )
    return prompt_text, answer.strip()


def _is_preformatted_text_example(example: Dict[str, Any]) -> bool:
    if "text" not in example or not str(example.get("text") or "").strip():
        return False
    structured_keys = {
        "messages",
        "instruction",
        "prompt",
        "question",
        "query",
        "output",
        "response",
        "answer",
        "completion",
        "chosen",
    }
    return not any(key in example and str(example.get(key) or "").strip() for key in structured_keys)


class SFTDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        tokenizer,
        block_size: int = 768,
        split: str = "train",
        max_examples: Optional[int] = None,
        train_on_prompt: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.train_on_prompt = train_on_prompt
        raw = load_hf_dataset(dataset_path, split=split)
        if max_examples is not None:
            raw = raw.select(range(min(max_examples, len(raw))))
        self.raw = raw
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = dict(self.raw[idx])
        if _is_preformatted_text_example(example):
            ids = self.tokenizer.encode(str(example["text"])).ids
            if self.eos_id is not None:
                ids.append(self.eos_id)
            ids = ids[: self.block_size]
            if len(ids) < 2:
                ids = ids + [self.eos_id or 2]
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(ids, dtype=torch.long),
            }

        prompt, answer = format_sft_example(example)
        if not prompt or not answer:
            prompt = prompt or "Please answer the following."
            answer = answer or ""
        prompt_text, answer_text = build_instruction_text(prompt, answer)
        prompt_ids = self.tokenizer.encode(prompt_text).ids
        answer_ids = self.tokenizer.encode(answer_text).ids
        eos_extra = 1 if self.eos_id is not None else 0
        if len(prompt_ids) + len(answer_ids) + eos_extra > self.block_size:
            max_answer = max(1, min(len(answer_ids), self.block_size // 2))
            answer_ids = answer_ids[:max_answer]
            max_prompt = max(1, self.block_size - len(answer_ids) - eos_extra)
            prompt_ids = prompt_ids[-max_prompt:]
        ids = prompt_ids + answer_ids
        if self.eos_id is not None:
            ids.append(self.eos_id)
        ids = ids[: self.block_size]
        if len(ids) < 2:
            ids = ids + [self.eos_id or 2]

        labels = ids.copy()
        if not self.train_on_prompt:
            mask_len = min(len(prompt_ids), len(labels))
            labels[:mask_len] = [-100] * mask_len
            if all(label == -100 for label in labels):
                labels[-1] = ids[-1]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

