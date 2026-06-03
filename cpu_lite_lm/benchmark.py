"""CPU inference benchmark."""

from __future__ import annotations

import argparse
import os
import time

import psutil
import torch

from .generate import load_model


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def benchmark(args: argparse.Namespace) -> None:
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, min(2, args.threads)))
    model = load_model(args.model, args.config).eval()
    input_ids = torch.randint(4, model.config.vocab_size, (1, args.prompt_tokens), dtype=torch.long)
    max_len = args.prompt_tokens + args.generated_tokens
    past = model.allocate_kv_cache(1, max_len) if args.use_cache else None
    generated = torch.empty((1, max_len), dtype=torch.long)
    generated[:, : args.prompt_tokens] = input_ids
    out_len = args.prompt_tokens

    with torch.inference_mode():
        t0 = time.perf_counter()
        out = model(
            input_ids,
            past_key_values=past,
            use_cache=args.use_cache,
            cache_position=torch.arange(args.prompt_tokens) if args.use_cache else None,
            logits_to_keep=1,
        )
        t1 = time.perf_counter()
        token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        t2 = time.perf_counter()
        for _ in range(args.generated_tokens):
            generated[:, out_len : out_len + 1] = token
            out_len += 1
            model_input = token if args.use_cache else generated[:, :out_len]
            cache_position = torch.tensor([out_len - 1]) if args.use_cache else None
            out = model(
                model_input,
                past_key_values=past,
                use_cache=args.use_cache,
                cache_position=cache_position,
                logits_to_keep=1,
            )
            token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        t3 = time.perf_counter()
    prefill = args.prompt_tokens / max(t1 - t0, 1e-9)
    decode = args.generated_tokens / max(t3 - t2, 1e-9)
    print(f"Model: CPULiteLM")
    print(f"Threads: {torch.get_num_threads()}")
    print(f"Prompt tokens: {args.prompt_tokens}")
    print(f"Generated tokens: {args.generated_tokens}")
    print(f"Prefill tok/s: {prefill:.2f}")
    print(f"Decode tok/s: {decode:.2f}")
    print(f"RSS memory MB: {rss_mb():.2f}")
    print(f"Use cache: {str(args.use_cache).lower()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/micro_ckpt")
    parser.add_argument("--config", default="configs/micro.json")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--generated-tokens", type=int, default=64)
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    return parser


def main() -> None:
    benchmark(build_parser().parse_args())


if __name__ == "__main__":
    main()
