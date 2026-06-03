"""Dynamic int8 quantization experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .generate import load_model


def state_size_mb(model: nn.Module) -> float:
    total = 0
    for value in model.state_dict().values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return total / (1024 * 1024)


def quantize_dynamic_model(args: argparse.Namespace) -> None:
    model = load_model(args.model, args.config).eval()
    before = state_size_mb(model)
    try:
        qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    except Exception as exc:
        raise RuntimeError(
            "Dynamic quantization failed. Check that this PyTorch build supports CPU dynamic quantization."
        ) from exc
    after = state_size_mb(qmodel)
    x = torch.randint(0, model.config.vocab_size, (1, 8), dtype=torch.long)
    with torch.no_grad():
        y = qmodel(x)
    print(f"Original state_dict MB: {before:.2f}")
    print(f"Quantized state_dict MB: {after:.2f}")
    print(f"Forward logits shape: {tuple(y.logits.shape)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/micro_ckpt")
    parser.add_argument("--config", default="configs/micro.json")
    return parser


def main() -> None:
    quantize_dynamic_model(build_parser().parse_args())


if __name__ == "__main__":
    main()
