"""CPU inference benchmark."""

from __future__ import annotations

import argparse
import os
import time

import psutil
import torch

from .generate import load_model
from .modeling_cpu_lite import CPULiteForCausalLM


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def benchmark(args: argparse.Namespace) -> None:
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, min(2, args.threads)))
    model = load_model(args.model, args.config).eval()
    if args.disable_moe_for_small_model and model.config.hidden_size <= args.small_model_hidden_threshold:
        model.config.num_experts = 0
        model.config.num_experts_per_tok = 0
        model = CPULiteForCausalLM(model.config).eval()
        print("MoE disabled for small-model CPU benchmark.")
    if args.moe_top_k > 0 and getattr(model.config, "num_experts", 0) > 0:
        model.config.num_experts_per_tok = args.moe_top_k
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
    print(f"MoE experts: {getattr(model.config, 'num_experts', 0)}")
    print(f"MoE top-k: {getattr(model.config, 'num_experts_per_tok', 0)}")

    if args.compare_dense_moe and getattr(model.config, "num_experts", 0) > 0:
        dense_cfg = model.config
        dense_cfg.num_experts = 0
        dense_cfg.num_experts_per_tok = 0
        dense_model = CPULiteForCausalLM(dense_cfg).eval()
        dense_decode = _decode_tokens_per_sec(dense_model, args)
        print(f"Dense random-init decode tok/s: {dense_decode:.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/micro_ckpt")
    parser.add_argument("--config", default="configs/micro.json")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--generated-tokens", type=int, default=64)
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.add_argument("--moe-top-k", type=int, default=1)
    parser.add_argument("--num-experts", type=int, default=4, help="Documentation default for CPU MoE experiments")
    parser.add_argument("--disable-moe-for-small-model", action="store_true")
    parser.add_argument("--small-model-hidden-threshold", type=int, default=512)
    parser.add_argument("--compare-dense-moe", action="store_true")
    return parser


def _decode_tokens_per_sec(model, args: argparse.Namespace) -> float:
    input_ids = torch.randint(4, model.config.vocab_size, (1, args.prompt_tokens), dtype=torch.long)
    with torch.inference_mode():
        t0 = time.perf_counter()
        _ = model.generate_simple(
            input_ids,
            max_new_tokens=args.generated_tokens,
            temperature=0.0,
            use_cache=args.use_cache,
            eos_token_id=None,
        )
        t1 = time.perf_counter()
    return args.generated_tokens / max(t1 - t0, 1e-9)


def main() -> None:
    benchmark(build_parser().parse_args())


if __name__ == "__main__":
    main()
