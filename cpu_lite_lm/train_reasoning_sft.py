"""Reasoning SFT for CPULiteLM using plan/solution/answer JSONL rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .configuration_cpu_lite import CPULiteConfig
from .data import collate_causal_lm
from .generate import load_model
from .modeling_cpu_lite import CPULiteForCausalLM
from .reasoning_data import ReasoningSFTJsonlDataset, ensure_reasoning_tokens
from .tokenizer_train import load_tokenizer
from .train import autocast_dtype, evaluate_loss, resolve_device


def train_reasoning_sft(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))

    tokenizer = load_tokenizer(args.tokenizer)
    token_mode = "latent" if args.reasoning_token_mode == "latent" else "plain"
    added = ensure_reasoning_tokens(
        tokenizer,
        token_mode,
        auto_add=args.auto_add_reasoning_tokens,
        count=args.reasoning_token_count,
    )
    model = load_model(args.model, args.config)
    if added:
        model.resize_token_embeddings(tokenizer.get_vocab_size())
    elif model.config.vocab_size < tokenizer.get_vocab_size():
        model.resize_token_embeddings(tokenizer.get_vocab_size())
    model.to(device).train()

    train_ds = ReasoningSFTJsonlDataset(
        args.data,
        tokenizer,
        block_size=args.block_size,
        max_examples=args.max_examples if args.max_examples > 0 else None,
        train_on_prompt=args.train_on_prompt,
        latent_tokens=token_mode == "latent",
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_causal_lm(b, model.config.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = None
    if args.eval_data:
        eval_ds = ReasoningSFTJsonlDataset(
            args.eval_data,
            tokenizer,
            block_size=args.block_size,
            max_examples=args.eval_examples if args.eval_examples > 0 else None,
            train_on_prompt=args.train_on_prompt,
            latent_tokens=token_mode == "latent",
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_causal_lm(b, model.config.pad_token_id),
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp_dtype = autocast_dtype(args.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    max_steps = args.max_steps if args.max_steps > 0 else max(1, math.ceil(len(loader) * args.epochs / args.grad_accum_steps))
    step = 0
    micro_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    print(
        f"Starting reasoning SFT (rows={len(train_ds)}, block_size={args.block_size}, "
        f"batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}, mode={token_mode})",
        flush=True,
    )
    while step < max_steps:
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(**batch, multi_exit_loss=args.early_exit_loss)
                loss = out.loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            running_loss += float(out.loss.detach().item())
            micro_step += 1
            if micro_step % args.grad_accum_steps:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % args.log_every == 0:
                mean_loss = running_loss / args.grad_accum_steps
                running_loss = 0.0
                print(f"reasoning_sft step {step}/{max_steps} loss {mean_loss:.4f}", flush=True)
            if eval_loader is not None and (step == 1 or step % args.eval_every == 0):
                val_loss, val_ppl = evaluate_loss(model, eval_loader, device, amp_dtype, args.eval_max_batches)
                print(f"reasoning_sft eval step {step} val_loss {val_loss:.4f} val_ppl {val_ppl:.2f}", flush=True)
            if args.save_every > 0 and step % args.save_every == 0:
                _save(model, tokenizer, args, step)
            if step >= max_steps:
                break
    return _save(model, tokenizer, args, step)


def _save(model: CPULiteForCausalLM, tokenizer, args: argparse.Namespace, step: int) -> Path:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save(str(output / "tokenizer.json"))
    payload = vars(args).copy()
    payload["last_step"] = step
    (output / "reasoning_sft_args.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved reasoning SFT checkpoint to {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/base_ckpt")
    parser.add_argument("--config", default="configs/micro.json")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--data", default="data/reasoning_sft.jsonl")
    parser.add_argument("--eval-data", default="")
    parser.add_argument("--output-dir", default="artifacts/reasoning_sft_ckpt")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-max-batches", type=int, default=20)
    parser.add_argument("--train-on-prompt", action="store_true")
    parser.add_argument("--early-exit-loss", action="store_true")
    parser.add_argument("--reasoning-token-mode", choices=["plain", "latent"], default="plain")
    parser.add_argument("--auto-add-reasoning-tokens", action="store_true")
    parser.add_argument("--reasoning-token-count", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="off")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    return parser


def main() -> None:
    train_reasoning_sft(build_parser().parse_args())


if __name__ == "__main__":
    main()
