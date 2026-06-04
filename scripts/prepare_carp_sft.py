"""Convert JSONL CARP traces into preformatted SFT JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.carp_data import build_carp_instruction_text, build_router_label, parse_carp_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with question/answer/reasoning_tokens fields")
    parser.add_argument("--output", required=True, help="Output JSONL with text and router_label fields")
    parser.add_argument("--max-reasoning-tokens", type=int, default=128)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            trace = parse_carp_trace(json.loads(line), args.max_reasoning_tokens)
            if not trace.question or not trace.answer:
                continue
            prompt, answer = build_carp_instruction_text(trace)
            payload = {
                "text": prompt + answer,
                "router_label": build_router_label(trace),
            }
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} CARP SFT rows to {dst}")


if __name__ == "__main__":
    main()
