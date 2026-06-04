"""Run CARP adaptive generation from a CPULiteLM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.carp import CARPGenerator
from cpu_lite_lm.generate import _amp_dtype, _resolve_device, load_model
from cpu_lite_lm.tokenizer_train import load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/micro_ckpt")
    parser.add_argument("--config", default="configs/micro.json")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--user", default="")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="off")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--no-speculative", action="store_true")
    parser.add_argument("--draft-layer", type=int, default=1)
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--reasoning-tokens", type=int, default=None)
    parser.add_argument("--show-carp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))

    device = _resolve_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.model, args.config).to(device).eval()
    eos_id = tokenizer.token_to_id("</s>") or tokenizer.token_to_id("<|endoftext|>") or tokenizer.token_to_id("<eos>")
    prompt = args.user or args.prompt
    if not prompt:
        raise ValueError("Provide --prompt or --user.")

    generator = CARPGenerator(
        model,
        tokenizer,
        num_reasoning_tokens=args.reasoning_tokens,
        draft_layer=args.draft_layer,
        lookahead=args.lookahead,
    )
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
            eos_token_id=eos_id,
        )

    if args.show_carp:
        print(
            f"[CARP] mode={result.route.difficulty_name} "
            f"budget={result.route.reasoning_budget} "
            f"candidates={len(result.candidates)} "
            f"reasoning={' '.join(result.reasoning_tokens) or '-'}"
        )
    print(result.answer)


if __name__ == "__main__":
    main()
