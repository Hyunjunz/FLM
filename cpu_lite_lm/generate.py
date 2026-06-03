"""Text generation CLI."""

from __future__ import annotations

import argparse
import sys
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
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.model, args.config).to(device).eval()
    ids = torch.tensor([tokenizer.encode(args.prompt).ids], dtype=torch.long, device=device)
    amp_dtype = _amp_dtype(args.amp_dtype)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model.generate_simple(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            use_cache=not args.no_cache,
        )
    text = tokenizer.decode(out[0].detach().cpu().tolist(), skip_special_tokens=True)
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, "backslashreplace").decode(encoding))
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/micro_ckpt")
    parser.add_argument("--config", default="configs/micro.json")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--prompt", default="안녕하세요, 저는")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="off")
    return parser


def main() -> None:
    generate(build_parser().parse_args())


if __name__ == "__main__":
    main()
