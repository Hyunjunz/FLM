"""Text generation CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .configuration_cpu_lite import CPULiteConfig
from .modeling_cpu_lite import CPULiteForCausalLM
from .tokenizer_train import load_tokenizer


def load_model(model_dir: str | Path, config_path: str | Path) -> CPULiteForCausalLM:
    model_dir = Path(model_dir)
    if (model_dir / "pytorch_model.bin").exists():
        return CPULiteForCausalLM.from_pretrained(model_dir)
    return CPULiteForCausalLM(CPULiteConfig.from_json_file(config_path))


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _amp_dtype(name: str):
    if name == "off":
        return None
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {name}")


def maybe_dynamic_int8(model: CPULiteForCausalLM, enabled: bool, device: torch.device):
    if not enabled:
        return model
    if device.type != "cpu":
        raise ValueError("--dynamic-int8 is CPU-only")
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "fast":
        args.helix = True
        args.speculative = True
        args.max_new_tokens = min(args.max_new_tokens, 128)
        args.temperature = args.temperature if args.temperature != 0.8 else 0.7
    elif args.preset == "balanced":
        args.helix = True
        args.verify_before_accept = True
        args.hard_full_depth = True
    elif args.preset == "reasoning":
        args.helix = False
        args.speculative = False
        args.no_cache = False
        args.temperature = 0.0 if args.temperature == 0.8 else min(args.temperature, 0.3)
        args.top_k = 0
        args.max_new_tokens = max(args.max_new_tokens, 256)


def generate(args: argparse.Namespace) -> str:
    apply_preset(args)
    device = _resolve_device(args.device)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.model, args.config).to(device).eval()
    model = maybe_dynamic_int8(model, args.dynamic_int8, device).eval()
    
    # Format prompt based on system/user input
    full_prompt = args.prompt
    if args.user:
        system_part = f"System: {args.system}\n\n" if args.system else ""
        full_prompt = f"### Question:\n{system_part}{args.user}\n\n### Answer:\n"
    
    ids = torch.tensor([tokenizer.encode(full_prompt).ids], dtype=torch.long, device=device)
    amp_dtype = _amp_dtype(args.amp_dtype)
    eos_id = tokenizer.token_to_id("</s>") or tokenizer.token_to_id("<|endoftext|>") or tokenizer.token_to_id("<eos>")
    
    generated_text = ""
    print("", end="", flush=True) # Prepare stdout

    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
    ):
        if args.helix:
            from .helix_runtime import HelixMindRuntime, HelixRuntimeState

            runtime = HelixMindRuntime(
                model,
                tokenizer,
                HelixRuntimeState(
                    default_top_k=args.top_k,
                    use_trained_router=args.helix_trained_router,
                    hard_full_depth=args.hard_full_depth,
                    verify_before_accept=args.verify_before_accept,
                    disable_early_exit_for_hard=args.disable_early_exit_for_hard,
                    easy_exit_threshold=args.easy_exit_threshold,
                    medium_exit_threshold=args.medium_exit_threshold,
                ),
            )
            generated_text = runtime.infer(
                full_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=eos_id,
            )
            print(generated_text, end="", flush=True)
        elif args.speculative:
            from .speculative import SelfSpeculativeGenerator
            generator = SelfSpeculativeGenerator(model, draft_layer=args.draft_layer, lookahead=args.lookahead)
            # Speculative yields slices or single tokens
            for token_bundle in generator.generate_streaming(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                eos_token_id=eos_id,
            ):
                # token_bundle is a tensor of shape (1, n)
                new_text = tokenizer.decode(token_bundle[0].tolist(), skip_special_tokens=True)
                print(new_text, end="", flush=True)
                generated_text += new_text
        else:
            for next_token in model.generate_streaming(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                use_cache=not args.no_cache,
                eos_token_id=eos_id,
            ):
                new_text = tokenizer.decode(next_token[0].tolist(), skip_special_tokens=True)
                print(new_text, end="", flush=True)
                generated_text += new_text
            
    print() # New line at the end
    return generated_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/micro_ckpt")
    parser.add_argument("--config", default="configs/micro.json")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--prompt", default="안녕하세요, 저는", help="Full raw prompt")
    parser.add_argument("--system", default="", help="System prompt")
    parser.add_argument("--user", default="", help="User question/prompt")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=1.0, help="Reserved for preset documentation; top-p sampling is not implemented")
    parser.add_argument("--preset", choices=["fast", "balanced", "reasoning"], default="balanced")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="off")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--dynamic-int8", action="store_true", help="Apply CPU dynamic int8 quantization to Linear layers")
    parser.add_argument("--speculative", action="store_true", help="Use self-speculative decoding")
    parser.add_argument("--draft-layer", type=int, default=1, help="Layer to exit for draft prediction")
    parser.add_argument("--lookahead", type=int, default=3, help="Number of tokens to speculate")
    parser.add_argument("--helix", action="store_true", help="Use HelixMind CPU reasoning runtime")
    parser.add_argument("--helix-trained-router", action="store_true", help="Use trained Helix router head")
    parser.add_argument("--disable-early-exit-for-hard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-full-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-before-accept", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--easy-exit-threshold", type=float, default=0.45)
    parser.add_argument("--medium-exit-threshold", type=float, default=0.60)
    return parser


def main() -> None:
    generate(build_parser().parse_args())


if __name__ == "__main__":
    main()
