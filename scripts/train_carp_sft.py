"""Train CPULiteLM on CARP JSONL traces with optional router/ranking losses.

Fixes over the uploaded draft:
- correct logging averages with gradient accumulation and log_every
- parameter count printout
- optional AdamW weight decay
- warmup + cosine LR schedule
- explicit zero_grad before training
"""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--router-loss-weight", type=float, default=0.2)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="off")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-step-dirs", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def build_lr_lambda(max_steps: int, warmup_steps: int, min_lr_ratio: float):
    warmup_steps = max(0, warmup_steps)
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        if max_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


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

    total_params, trainable_params = count_params(model)
    print(
        f"params total={total_params:,} ({total_params / 1e6:.2f}M) "
        f"trainable={trainable_params:,} ({trainable_params / 1e6:.2f}M)",
        flush=True,
    )

    model.to(device).train()
    ds = CARPJsonlSFTDataset(
        args.data,
        tokenizer,
        block_size=args.block_size,
        max_reasoning_tokens=args.reasoning_tokens,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_carp_sft(batch, model.config.pad_token_id),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=build_lr_lambda(args.max_steps, args.warmup_steps, args.min_lr_ratio),
    )
    amp_dtype = autocast_dtype(args.amp_dtype)

    step = 0
    micro_step = 0
    running_loss = 0.0
    running_lm = 0.0
    running_router = 0.0
    running_rank = 0.0
    running_router_acc = 0.0
    running_count = 0

    optim.zero_grad(set_to_none=True)
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
                loss = out.loss / args.grad_accum_steps

            loss.backward()
            running_loss += float(out.loss.detach())
            running_lm += float(out.lm_loss.detach())
            running_router += float(out.router_loss.detach())
            running_rank += float(out.ranking_loss.detach())
            running_router_acc += float(out.router_accuracy)
            running_count += 1

            micro_step += 1
            if micro_step % args.grad_accum_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            step += 1

            if step == 1 or step % args.log_every == 0:
                denom = max(1, running_count)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"step {step}/{args.max_steps} loss {running_loss / denom:.4f} "
                    f"lm {running_lm / denom:.4f} router {running_router / denom:.4f} "
                    f"rank {running_rank / denom:.4f} "
                    f"router_acc {running_router_acc / denom:.3f} lr {lr:.3e}",
                    flush=True,
                )
                running_loss = 0.0
                running_lm = 0.0
                running_router = 0.0
                running_rank = 0.0
                running_router_acc = 0.0
                running_count = 0

            if args.save_every > 0 and step % args.save_every == 0:
                save_dir = Path(args.output_dir) / f"step_{step}" if args.save_step_dirs else Path(args.output_dir)
                _save_checkpoint(model, tokenizer, args, save_dir, step)

            if step >= args.max_steps:
                break

    output = _save_checkpoint(model, tokenizer, args, Path(args.output_dir), step)
    print(f"Saved CARP SFT checkpoint to {output}")


def _save_checkpoint(model, tokenizer, args: argparse.Namespace, output: Path, step: int) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save(str(output / "tokenizer.json"))
    payload = vars(args).copy()
    payload["last_step"] = step
    (output / "carp_sft_args.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved CARP SFT checkpoint step={step} to {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
