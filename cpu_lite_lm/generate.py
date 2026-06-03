"""Text generation CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

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


def generate(args: argparse.Namespace) -> str:
    device = _resolve_device(args.device)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.model, args.config).to(device).eval()
    
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
        if args.speculative:
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
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="off")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--speculative", action="store_true", help="Use self-speculative decoding")
    parser.add_argument("--draft-layer", type=int, default=1, help="Layer to exit for draft prediction")
    parser.add_argument("--lookahead", type=int, default=3, help="Number of tokens to speculate")
    return parser


def main() -> None:
    generate(build_parser().parse_args())


if __name__ == "__main__":
    main()
