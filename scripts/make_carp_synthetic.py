"""Create a small synthetic CARP trace set for smoke-proof experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/carp_synthetic.jsonl")
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for idx in range(args.examples):
            kind = idx % 4
            a = rng.randint(1, 99)
            b = rng.randint(1, 99)
            c = rng.randint(1, 20)
            if kind == 0:
                row = {
                    "question": f"What is {a} + {b}?",
                    "answer": str(a + b),
                    "reasoning_tokens": ["<R0>"],
                    "difficulty": "medium",
                }
            elif kind == 1:
                row = {
                    "question": f"What is {a} - {b}?",
                    "answer": str(a - b),
                    "reasoning_tokens": ["<R1>"],
                    "difficulty": "medium",
                }
            elif kind == 2:
                row = {
                    "question": f"If x + {a} = {b}, what is x?",
                    "answer": str(b - a),
                    "reasoning_tokens": ["<R0>", "<R2>"],
                    "difficulty": "hard",
                }
            else:
                row = {
                    "question": f"Compute ({a} + {b}) * {c}.",
                    "answer": str((a + b) * c),
                    "reasoning_tokens": ["<R0>", "<R3>"],
                    "difficulty": "hard",
                }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {args.examples} synthetic CARP traces to {out}")


if __name__ == "__main__":
    main()
