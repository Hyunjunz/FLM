"""Run CARP language-reasoning inference from a trained checkpoint."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.carp import CARPGenerator
from cpu_lite_lm.generate import _amp_dtype, _resolve_device
from cpu_lite_lm.modeling_cpu_lite import CPULiteForCausalLM
from cpu_lite_lm.tokenizer_train import load_tokenizer


def build_prompt(question: str, choices: str) -> str:
    return (
        "Choose the best commonsense answer.\n\n"
        f"Question: {question.strip()}\n\n"
        f"Choices:\n{choices.strip()}\n\n"
        "Answer with the option letter and text."
    )


def parse_choice(text: str) -> str:
    match = re.search(r"\b([A-E])\b\s*[\.\)]?", text.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/carp_language_ckpt")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--choices",
        required=True,
        help='Multiline or escaped choices, e.g. "A. red\\nB. blue\\nC. green\\nD. black\\nE. white"',
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--no-speculative", action="store_true")
    parser.add_argument("--show-carp", action="store_true")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer or args.model)
    model = CPULiteForCausalLM.from_pretrained(args.model).to(device).eval()
    generator = CARPGenerator(model, tokenizer)
    prompt = build_prompt(args.question, args.choices.replace("\\n", "\n"))
    amp_dtype = _amp_dtype(args.amp_dtype)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
    ):
        result = generator.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            use_speculative=not args.no_speculative,
        )
    choice = parse_choice(result.answer)
    if args.show_carp:
        print(
            f"[CARP] mode={result.route.difficulty_name} "
            f"reasoning={' '.join(result.reasoning_tokens) or '-'} "
            f"parsed={choice or '-'}"
        )
    print(result.answer)


if __name__ == "__main__":
    main()
