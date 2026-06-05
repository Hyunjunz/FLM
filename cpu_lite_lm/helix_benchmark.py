"""Benchmark baseline generation against HelixMind runtime."""

from __future__ import annotations

import argparse
import os
import time

import psutil
import torch

from .generate import load_model
from .helix_runtime import HelixMindRuntime, HelixRuntimeState, iter_quant_policy_lines
from .tokenizer_train import load_tokenizer


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_baseline(model, input_ids, max_new_tokens: int, temperature: float, top_k: int, eos_id) -> str:
    chunks = []
    for tok in model.generate_streaming(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_cache=True,
        eos_token_id=eos_id,
    ):
        chunks.extend(tok[0].tolist())
    return chunks


def benchmark(args: argparse.Namespace) -> None:
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, min(2, args.threads)))

    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.model, args.config).eval()
    eos_id = tokenizer.token_to_id("</s>") or tokenizer.token_to_id("<|endoftext|>") or tokenizer.token_to_id("<eos>")
    prompt = args.prompt
    if args.user:
        prompt = f"### Question:\n{args.user}\n\n### Answer:\n"
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)

    runtime = HelixMindRuntime(
        model,
        tokenizer,
        HelixRuntimeState(default_top_k=args.top_k, use_trained_router=args.helix_trained_router),
    )

    with torch.inference_mode():
        t0 = time.perf_counter()
        baseline_ids = run_baseline(model, input_ids, args.max_new_tokens, args.temperature, args.top_k, eos_id)
        t1 = time.perf_counter()
        helix_text = runtime.infer(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=eos_id,
        )
        t2 = time.perf_counter()

    baseline_tps = len(baseline_ids) / max(t1 - t0, 1e-9)
    helix_ids = tokenizer.encode(helix_text).ids
    helix_tps = len(helix_ids) / max(t2 - t1, 1e-9)
    print("Model: CPULiteLM + HelixMind")
    print(f"Threads: {torch.get_num_threads()}")
    print(f"Prompt tokens: {input_ids.size(1)}")
    print(f"Requested new tokens: {args.max_new_tokens}")
    print(f"Baseline decode tok/s: {baseline_tps:.2f}")
    print(f"Helix apparent tok/s: {helix_tps:.2f}")
    print(f"Helix output tokens: {len(helix_ids)}")
    print(f"RSS memory MB: {rss_mb():.2f}")
    print(f"Helix stats: {runtime.state.stats}")
    if args.print_quant_policy:
        print("Layer-wise quantization policy:")
        for line in iter_quant_policy_lines(model):
            print(f"  {line}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/micro_ckpt")
    parser.add_argument("--config", default="configs/micro.json")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--prompt", default="### Question:\nWhat is a pillow used for?\n\n### Answer:\n")
    parser.add_argument("--user", default="")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--print-quant-policy", action="store_true")
    parser.add_argument("--helix-trained-router", action="store_true")
    return parser


def main() -> None:
    benchmark(build_parser().parse_args())


if __name__ == "__main__":
    main()
