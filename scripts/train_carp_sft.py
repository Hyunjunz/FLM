"""Train CPULiteLM on CARP JSONL traces with optional router-head loss."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.carp import add_reasoning_tokens
from cpu_lite_lm.carp_data import CARPJsonlSFTDataset, collate_carp_sft
from cpu_lite_lm.carp_train import carp_sft_loss
from cpu_lite_lm.configuration_cpu_lite import CPULiteConfig
from cpu_lite_lm.modeling_cpu_lite import CPULiteForCausalLM
from cpu_lite_lm.tokenizer_train import load_tokenizer
from cpu_lite_lm.train import autocast_dtype, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/carp_synthetic.jsonl")
    parser.add_argument("--config", default="configs/carp_micro.json")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--output-dir", default="artifacts/carp_sft_ckpt")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--reasoning-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--router-loss-weight", type=float, default=0.2)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="off")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    add_reasoning_tokens(tokenizer, args.reasoning_tokens)

    if args.base_model and (Path(args.base_model) / "pytorch_model.bin").exists():
        model = CPULiteForCausalLM.from_pretrained(args.base_model)
        model.resize_token_embeddings(tokenizer.get_vocab_size())
        model.config.carp_num_reasoning_tokens = args.reasoning_tokens
        if model.router_head is None:
            model.config.carp_router_labels = 4
            model.router_head = torch.nn.Linear(model.config.hidden_size, 4)
    else:
        cfg = CPULiteConfig.from_json_file(args.config)
        cfg.vocab_size = max(cfg.vocab_size, tokenizer.get_vocab_size())
        cfg.carp_num_reasoning_tokens = args.reasoning_tokens
        cfg.carp_router_labels = max(4, cfg.carp_router_labels)
        model = CPULiteForCausalLM(cfg)

    model.to(device).train()
    ds = CARPJsonlSFTDataset(args.data, tokenizer, block_size=args.block_size, max_reasoning_tokens=args.reasoning_tokens)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_carp_sft(batch, model.config.pad_token_id),
    )
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    amp_dtype = autocast_dtype(args.amp_dtype)
    step = 0
    while step < args.max_steps:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = carp_sft_loss(
                    model,
                    batch,
                    router_loss_weight=args.router_loss_weight,
                    ranking_loss_weight=args.ranking_loss_weight,
                )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % args.log_every == 0:
                print(
                    f"step {step}/{args.max_steps} loss {float(out.loss.detach()):.4f} "
                    f"lm {float(out.lm_loss):.4f} router {float(out.router_loss):.4f} "
                    f"rank {float(out.ranking_loss):.4f} "
                    f"router_acc {out.router_accuracy:.3f}",
                    flush=True,
                )
            if step >= args.max_steps:
                break

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save(str(output / "tokenizer.json"))
    (output / "carp_sft_args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"Saved CARP SFT checkpoint to {output}")


if __name__ == "__main__":
    main()
