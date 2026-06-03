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
    if data_path.is_file() and data_path.suffix.lower() not in {".arrow"}:
        tokenizer.train([str(data_path)], trainer)
    else:
        iterator = iter_texts_from_path(data_path, text_column=text_column, max_docs=max_docs)
        tokenizer.train_from_iterator(iterator, trainer=trainer, length=max_docs)
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<bos>", 1), ("<eos>", 2)],
    )
    path = output_dir / "tokenizer.json"
    tokenizer.save(str(path))
    return path


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
    args = parser.parse_args()
    path = train_tokenizer(args.data, args.output_dir, args.vocab_size, args.text_column, args.max_docs)
    print(f"Saved tokenizer to {path}")


if __name__ == "__main__":
    main()
