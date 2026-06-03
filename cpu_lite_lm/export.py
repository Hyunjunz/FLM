"""Export helpers for CPULiteLM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .generate import load_model


def export_onnx(args: argparse.Namespace) -> None:
    model = load_model(args.model, args.config).eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.ones((1, args.seq_len), dtype=torch.long)
    try:
        torch.onnx.export(
            model,
            (dummy,),
            output,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {1: "seq"}, "logits": {1: "seq"}},
            opset_version=17,
        )
        print(f"Saved ONNX model to {output}")
    except Exception as exc:
        print("ONNX export failed. This skeleton does not yet handle cache outputs or all runtime variants.")
        print(f"Error: {exc}")
    print("GGUF TODO: map tensor names to llama.cpp conventions, add tokenizer metadata, then write GGUF.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/micro_ckpt")
    parser.add_argument("--config", default="configs/micro.json")
    parser.add_argument("--output", default="artifacts/cpu_lite_lm.onnx")
    parser.add_argument("--seq-len", type=int, default=8)
    return parser


def main() -> None:
    export_onnx(build_parser().parse_args())


if __name__ == "__main__":
    main()

