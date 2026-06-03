"""Inspect tokenization and rough quality stats for a local text/Arrow dataset."""

from __future__ import annotations

import argparse
from collections import Counter

from .data import iter_texts_from_path
from .tokenizer_train import load_tokenizer


def inspect(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args.tokenizer)
    unk_id = tokenizer.token_to_id("<unk>")
    total_tokens = 0
    unk_tokens = 0
    docs = 0
    chars = 0
    first_tokens = Counter()
    repeated_docs = 0
    seen_prefixes = set()

    for text in iter_texts_from_path(
        args.data,
        text_column=args.text_column,
        max_docs=args.max_docs,
        min_chars=args.min_chars,
        skip_docs=args.skip_docs,
        quality_filter=args.quality_filter,
    ):
        docs += 1
        chars += len(text)
        prefix = text[: args.prefix_chars]
        if prefix in seen_prefixes:
            repeated_docs += 1
        seen_prefixes.add(prefix)
        ids = tokenizer.encode(text[: args.max_chars_per_doc]).ids
        total_tokens += len(ids)
        if unk_id is not None:
            unk_tokens += sum(i == unk_id for i in ids)
        first_tokens.update(ids[: args.first_token_window])

    print(f"Docs sampled: {docs}")
    print(f"Chars sampled: {chars}")
    print(f"Tokens sampled: {total_tokens}")
    print(f"UNK rate: {unk_tokens / max(total_tokens, 1):.6f}")
    print(f"Repeated prefixes: {repeated_docs}")
    print("Top token ids:")
    for token_id, count in first_tokens.most_common(20):
        token = repr(tokenizer.id_to_token(token_id)).encode("ascii", "backslashreplace").decode("ascii")
        print(f"  {token_id:5d} {count:7d} {token}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/hf_cache/HAERAE-HUB___korean-webtext")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer_korean_webtext")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-docs", type=int, default=1000)
    parser.add_argument("--min-chars", type=int, default=0)
    parser.add_argument("--skip-docs", type=int, default=0)
    parser.add_argument("--quality-filter", action="store_true")
    parser.add_argument("--max-chars-per-doc", type=int, default=2000)
    parser.add_argument("--prefix-chars", type=int, default=120)
    parser.add_argument("--first-token-window", type=int, default=128)
    return parser


def main() -> None:
    inspect(build_parser().parse_args())


if __name__ == "__main__":
    main()
