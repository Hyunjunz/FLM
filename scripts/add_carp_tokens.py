"""Add CARP reasoning tokens to an existing tokenizer.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.carp import save_tokenizer_with_reasoning_tokens
from cpu_lite_lm.tokenizer_train import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    path = save_tokenizer_with_reasoning_tokens(tokenizer, args.output_dir, args.count)
    print(f"Saved CARP tokenizer to {path}")


if __name__ == "__main__":
    main()
