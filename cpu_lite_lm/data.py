"""Small causal language modeling dataset and local HF cache readers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
from typing import Dict, Iterator, Optional

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


def iter_texts_from_path(
    data_path: str | Path,
    text_column: str = "text",
    max_docs: Optional[int] = None,
    min_chars: int = 0,
    skip_docs: int = 0,
    quality_filter: bool = False,
) -> Iterator[str]:
    """Yield text records from a plain text file or HF datasets Arrow cache directory."""
    path = Path(data_path)
    yielded = 0
    skipped = 0

    def allow_more() -> bool:
        return max_docs is None or yielded < max_docs

    def should_yield(text: str) -> bool:
        nonlocal skipped
        if len(text) < min_chars:
            return False
        if quality_filter and not is_reasonable_webtext(text):
            return False
        if skipped < skip_docs:
            skipped += 1
            return False
        return True

    if path.is_file():
        if path.suffix.lower() == ".arrow":
            for text in _iter_arrow_texts(path, text_column, max_docs):
                if should_yield(text):
                    yield text
        else:
            text = path.read_text(encoding="utf-8")
            if text.strip() and should_yield(text):
                yielded += 1
                yield text
        return

    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {path}")

    arrow_files = sorted(path.rglob("*.arrow"))
    if arrow_files:
        for arrow_path in arrow_files:
            for text in _iter_arrow_texts(arrow_path, text_column, None):
                if not allow_more():
                    return
                if should_yield(text):
                    yielded += 1
                    yield text
        return

    text_files = sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".text"}
    )
    if not text_files:
        raise FileNotFoundError(f"No .arrow or .txt files found under: {path}")
    for text_path in text_files:
        if not allow_more():
            return
        yielded += 1
        text = text_path.read_text(encoding="utf-8")
        if text.strip() and should_yield(text):
            yield text


def is_reasonable_webtext(text: str) -> bool:
    """Cheap filters for noisy Korean webtext.

    These filters intentionally avoid model-specific assumptions. They remove
    very symbol-heavy, control-character-heavy, or extremely repetitive records.
    """
    stripped = text.strip()
    if len(stripped) < 20:
        return False
    sample = stripped[:4000]
    control = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\n\t\r")
    if control / max(len(sample), 1) > 0.01:
        return False
    alnum_or_ko = sum(1 for ch in sample if ch.isalnum() or ("가" <= ch <= "힣"))
    if alnum_or_ko / max(len(sample), 1) < 0.35:
        return False
    top_char_count = max(Counter(sample).values()) if sample else 0
    if top_char_count / max(len(sample), 1) > 0.30:
        return False
    return True


def _iter_arrow_texts(
    arrow_path: Path,
    text_column: str = "text",
    max_docs: Optional[int] = None,
) -> Iterator[str]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Reading HF Arrow cache requires pyarrow. Install it with `pip install pyarrow`."
        ) from exc

    count = 0
    with pa.memory_map(str(arrow_path), "r") as source:
        reader = ipc.open_stream(source)
        for batch in reader:
            if text_column not in batch.schema.names:
                raise KeyError(
                    f"Column '{text_column}' not found in {arrow_path}. "
                    f"Available columns: {batch.schema.names}"
                )
            values = batch.column(text_column).to_pylist()
            for value in values:
                if max_docs is not None and count >= max_docs:
                    return
                count += 1
                if isinstance(value, str) and value.strip():
                    yield value


class TextCausalLMDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        block_size: int = 128,
        text_column: str = "text",
        max_docs: Optional[int] = None,
        min_chars: int = 0,
        skip_docs: int = 0,
        quality_filter: bool = False,
        max_chars: int = 0,
        stride: Optional[int] = None,
    ) -> None:
        ids = []
        used_chars = 0
        eos_id = tokenizer.token_to_id("<eos>")
        for text in iter_texts_from_path(
            data_path,
            text_column=text_column,
            max_docs=max_docs,
            min_chars=min_chars,
            skip_docs=skip_docs,
            quality_filter=quality_filter,
        ):
            if max_chars > 0:
                remaining = max_chars - used_chars
                if remaining <= 0:
                    break
                text = text[:remaining]
            used_chars += len(text)
            encoded = tokenizer.encode(text).ids
            ids.extend(encoded)
            if eos_id is not None and (not ids or ids[-1] != eos_id):
                ids.append(eos_id)
            if max_chars > 0 and used_chars >= max_chars:
                break
        if len(ids) < 2:
            raise ValueError(f"Not enough tokens in {data_path} to build a causal LM dataset")
        self.block_size = block_size
        self.ids = ids
        self.examples = []
        step = max(1, stride if stride is not None else block_size)
        for start in range(0, max(1, len(ids) - 1), step):
            chunk = ids[start : start + block_size]
            if len(chunk) >= 2:
                self.examples.append(chunk)
        if not self.examples:
            self.examples.append(ids[: min(len(ids), block_size)])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.examples[idx]
        x = torch.tensor(chunk, dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}


def collate_causal_lm(batch, pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].numel()
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class StreamingTextCausalLMDataset(IterableDataset):
    """Streaming dataset for large local HF Arrow caches.

    It keeps only a token buffer in memory and yields fixed-size causal LM chunks.
    This is slower than pre-tokenizing but is safe for full-dataset CPU smoke runs.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        block_size: int = 128,
        text_column: str = "text",
        max_docs: Optional[int] = None,
        min_chars: int = 0,
        skip_docs: int = 0,
        quality_filter: bool = False,
        max_chars: int = 0,
        shuffle_buffer: int = 0,
        seed: int = 1234,
        stride: Optional[int] = None,
    ) -> None:
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_column = text_column
        self.max_docs = max_docs
        self.min_chars = min_chars
        self.skip_docs = skip_docs
        self.quality_filter = quality_filter
        self.max_chars = max_chars
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.eos_id = tokenizer.token_to_id("<eos>")
        self.stride = stride if stride is not None else block_size
        if self.stride != block_size:
            raise ValueError("StreamingTextCausalLMDataset currently requires stride == block_size")

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        buffer = []
        chunk_buffer = []
        rng = random.Random(self.seed + worker_id)

        def emit_chunk(chunk):
            x = torch.tensor(chunk, dtype=torch.long)
            return {"input_ids": x, "labels": x.clone()}

        used_chars = 0
        for doc_idx, text in enumerate(iter_texts_from_path(
            self.data_path,
            text_column=self.text_column,
            max_docs=self.max_docs,
            min_chars=self.min_chars,
            skip_docs=self.skip_docs,
            quality_filter=self.quality_filter,
        )):
            if num_workers > 1 and doc_idx % num_workers != worker_id:
                continue
            if self.max_chars > 0:
                remaining = self.max_chars - used_chars
                if remaining <= 0:
                    break
                text = text[:remaining]
            used_chars += len(text)
            buffer.extend(self.tokenizer.encode(text).ids)
            if self.eos_id is not None and (not buffer or buffer[-1] != self.eos_id):
                buffer.append(self.eos_id)
            while len(buffer) >= self.block_size:
                chunk = buffer[: self.block_size]
                del buffer[: self.block_size]
                if self.shuffle_buffer > 1:
                    chunk_buffer.append(chunk)
                    if len(chunk_buffer) >= self.shuffle_buffer:
                        idx = rng.randrange(len(chunk_buffer))
                        yield emit_chunk(chunk_buffer.pop(idx))
                else:
                    yield emit_chunk(chunk)
            if self.max_chars > 0 and used_chars >= self.max_chars:
                break
        if len(buffer) >= 2:
            if self.shuffle_buffer > 1:
                chunk_buffer.append(buffer)
            else:
                yield emit_chunk(buffer)
        while chunk_buffer:
            idx = rng.randrange(len(chunk_buffer))
            yield emit_chunk(chunk_buffer.pop(idx))
