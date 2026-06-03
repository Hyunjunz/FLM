"""Tokenizer training and loading helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from .data import iter_texts_from_path


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
DEFAULT_HF_CACHE = Path("data/hf_cache/HAERAE-HUB___korean-webtext")
DEFAULT_SAMPLE = Path("data/sample_corpus.txt")


def default_data_path() -> str:
    return str(DEFAULT_HF_CACHE if DEFAULT_HF_CACHE.exists() else DEFAULT_SAMPLE)


def train_tokenizer(
    data_path: str | Path,
    output_dir: str | Path,
    vocab_size: int = 1024,
    text_column: str = "text",
    max_docs: int | None = 2000,
    log_every: int = 1000,
) -> Path:
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    print(
        f"Training tokenizer from {data_path} "
        f"(vocab_size={vocab_size}, max_docs={max_docs})",
        flush=True,
    )
    if data_path.is_file() and data_path.suffix.lower() not in {".arrow"}:
        tokenizer.train([str(data_path)], trainer)
    else:
        if (data_path / "dataset_info.json").exists() or (data_path / "state.json").exists():
            iterator = iter_texts_from_saved_dataset(data_path, text_column, max_docs, log_every)
            tokenizer.train_from_iterator(iterator, trainer=trainer, length=max_docs)
            path = output_dir / "tokenizer.json"
            tokenizer.save(str(path))
            print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}", flush=True)
            return path

        def logging_iterator():
            for idx, text in enumerate(
                iter_texts_from_path(data_path, text_column=text_column, max_docs=max_docs),
                start=1,
            ):
                if log_every > 0 and idx % log_every == 0:
                    print(f"tokenizer docs read: {idx}", flush=True)
                yield text

        iterator = logging_iterator()
        tokenizer.train_from_iterator(iterator, trainer=trainer, length=max_docs)
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<bos>", 1), ("<eos>", 2)],
    )
    path = output_dir / "tokenizer.json"
    tokenizer.save(str(path))
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}", flush=True)
    return path


def iter_texts_from_saved_dataset(
    data_path: Path,
    text_column: str = "text",
    max_docs: int | None = None,
    log_every: int = 1000,
):
    try:
        from datasets import load_from_disk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Reading a saved HF dataset requires datasets.") from exc

    ds = load_from_disk(str(data_path))
    if hasattr(ds, "keys"):
        ds = ds["train"]
    limit = len(ds) if max_docs is None else min(max_docs, len(ds))
    formatter = None
    if text_column not in ds.column_names:
        from .sft_data import build_instruction_text, format_sft_example

        formatter = (format_sft_example, build_instruction_text)
    for idx in range(limit):
        if log_every > 0 and (idx + 1) % log_every == 0:
            print(f"tokenizer docs read: {idx + 1}", flush=True)
        row = dict(ds[idx])
        if formatter is None:
            value = row[text_column]
        else:
            format_sft_example, build_instruction_text = formatter
            prompt, answer = format_sft_example(row)
            prompt_text, answer_text = build_instruction_text(prompt, answer)
            value = prompt_text + answer_text
        if isinstance(value, str) and value.strip():
            yield value


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    return Tokenizer.from_file(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=default_data_path())
    parser.add_argument("--output-dir", default="artifacts/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()
    path = train_tokenizer(
        args.data,
        args.output_dir,
        args.vocab_size,
        args.text_column,
        args.max_docs,
        args.log_every,
    )
    print(f"Saved tokenizer to {path}")


if __name__ == "__main__":
    main()
