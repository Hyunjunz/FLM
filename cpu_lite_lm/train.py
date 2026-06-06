"""Training loop for CPULiteLM on CPU or GPU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .configuration_cpu_lite import CPULiteConfig
from .data import StreamingTextCausalLMDataset, TextCausalLMDataset, collate_causal_lm
from .modeling_cpu_lite import CPULiteForCausalLM
from .tokenizer_train import load_tokenizer, train_tokenizer


DEFAULT_HF_CACHE = Path("data/hf_cache/HAERAE-HUB___korean-webtext")
DEFAULT_SAMPLE = Path("data/sample_corpus.txt")


def default_data_path() -> str:
    return str(DEFAULT_HF_CACHE if DEFAULT_HF_CACHE.exists() else DEFAULT_SAMPLE)


def parse_optional_int(value: str | int | None) -> Optional[int]:
    if value is None or isinstance(value, int):
        return value
    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def autocast_dtype(name: str) -> Optional[torch.dtype]:
    if name == "off":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported --amp-dtype {name}. Use off, fp16, or bf16.")


def count_causal_loss_tokens(batch: dict[str, torch.Tensor]) -> int:
    labels = batch["labels"]
    if labels.size(1) <= 1:
        return 0
    return int((labels[:, 1:] != -100).sum().item())


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    max_batches: int,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(**batch)
        tokens = count_causal_loss_tokens(batch)
        total_loss += float(out.loss.item()) * tokens
        total_tokens += tokens
        batches += 1
        if max_batches > 0 and batches >= max_batches:
            break
    if was_training:
        model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, math.exp(min(mean_loss, 20.0))


def train(args: argparse.Namespace) -> Path:
    print(f"Training data: {args.data}", flush=True)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
        print(f"CPU threads: {args.cpu_threads}", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"TF32: {args.tf32}", flush=True)
    tokenizer_dir = Path(args.tokenizer)
    print(f"Tokenizer arg: {tokenizer_dir}", flush=True)
    print(f"Tokenizer path exists: {tokenizer_dir.exists()}", flush=True)
    print(f"Tokenizer path is_dir: {tokenizer_dir.is_dir()}", flush=True)
    tokenizer_file = tokenizer_dir / "tokenizer.json" if tokenizer_dir.is_dir() else tokenizer_dir
    print(f"Resolved tokenizer file: {tokenizer_file}", flush=True)
    if not tokenizer_file.exists():
        print(f"Tokenizer not found at {tokenizer_file}; starting tokenizer training.", flush=True)
        train_tokenizer(
            args.data,
            tokenizer_dir,
            args.vocab_size,
            args.text_column,
            args.tokenizer_max_docs,
            args.tokenizer_log_every,
        )
    else:
        print(f"Using tokenizer: {tokenizer_file}", flush=True)
    print("Loading tokenizer...", flush=True)
    tokenizer = load_tokenizer(tokenizer_dir)
    print(f"Loaded tokenizer vocab size: {tokenizer.get_vocab_size()}", flush=True)
    print(f"Loading config: {args.config}", flush=True)
    config = CPULiteConfig.from_json_file(args.config)
    config.vocab_size = max(config.vocab_size, tokenizer.get_vocab_size())
    print(
        "Model config: "
        f"layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
        f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
        f"vocab={config.vocab_size}",
        flush=True,
    )
    print("Initializing model...", flush=True)
    model = CPULiteForCausalLM(config)
    if args.resume_from:
        resume_path = Path(args.resume_from)
        state_path = resume_path / "pytorch_model.bin"
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot resume. Missing checkpoint: {state_path}")
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        print(f"Resumed model weights from {resume_path}", flush=True)
    print("Moving model to device...", flush=True)
    model.to(device)
    print("Model is on device.", flush=True)
    if args.compile:
        if device.type != "cuda":
            print("Skipping torch.compile because the active device is not CUDA.", flush=True)
        else:
            model = torch.compile(model)  # type: ignore[assignment]
            print("Enabled torch.compile", flush=True)
    model.train()
    print("Building dataset and dataloader...", flush=True)
    dataset_kwargs = dict(
        text_column=args.text_column,
        max_docs=args.max_docs,
        min_chars=args.min_chars,
        skip_docs=args.skip_docs,
        quality_filter=args.quality_filter,
        max_chars=args.max_chars,
        stride=args.stride,
    )
    if args.streaming:
        dataset = StreamingTextCausalLMDataset(
            args.data,
            tokenizer,
            args.block_size,
            **dataset_kwargs,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
        )
    else:
        dataset = TextCausalLMDataset(args.data, tokenizer, args.block_size, **dataset_kwargs)
        token_count = len(dataset.ids)
        print(
            f"Loaded train tokens: {token_count} examples: {len(dataset)} stride: {args.stride or args.block_size}",
            flush=True,
        )
        if token_count < 1_000_000 and args.max_steps > 100:
            print(
                "Warning: training corpus has fewer than 1,000,000 tokens; "
                "rapid train loss collapse is likely overfitting, not model skill.",
                flush=True,
            )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.streaming,
        collate_fn=lambda b: collate_causal_lm(b, config.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = None
    if args.eval_every > 0:
        eval_data = args.eval_data or args.data
        if args.eval_data and not Path(eval_data).exists():
            print(
                f"Validation data not found at {eval_data}; falling back to training data {args.data}.",
                flush=True,
            )
            eval_data = args.data
        if not args.eval_data:
            print(
                "Warning: --eval-every is enabled without --eval-data; validation loss will use "
                "the training source and can be over-optimistic.",
                flush=True,
            )
        print(
            f"Building validation loader from {eval_data} "
            f"(docs={args.eval_docs}, max_chars={args.eval_max_chars})",
            flush=True,
        )
        eval_dataset = TextCausalLMDataset(
            eval_data,
            tokenizer,
            args.block_size,
            text_column=args.text_column,
            max_docs=args.eval_docs,
            min_chars=args.min_chars,
            skip_docs=args.eval_skip_docs,
            quality_filter=args.quality_filter,
            max_chars=args.eval_max_chars,
            stride=args.stride,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_causal_lm(b, config.pad_token_id),
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        foreach=args.foreach_optimizer,
    )
    amp_dtype = autocast_dtype(args.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    step = 0
    micro_step = 0
    running_loss = 0.0
    target = "full epoch" if args.max_steps <= 0 else str(args.max_steps)
    optim.zero_grad(set_to_none=True)
    print(
        "Starting training loop "
        f"(batch_size={args.batch_size}, block_size={args.block_size}, "
        f"grad_accum_steps={args.grad_accum_steps}, streaming={args.streaming})",
        flush=True,
    )
    while args.max_steps <= 0 or step < args.max_steps:
        saw_batch = False
        for batch in loader:
            saw_batch = True
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(**batch, multi_exit_loss=args.multi_exit_loss)
                loss = out.loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            running_loss += float(out.loss.detach().item())
            micro_step += 1
            if micro_step % args.grad_accum_steps != 0:
                continue
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            step += 1
            mean_loss = running_loss / args.grad_accum_steps
            running_loss = 0.0
            if step % args.log_every == 0 or step == 1:
                ppl = math.exp(min(mean_loss, 20.0))
                tokens = count_causal_loss_tokens(batch)
                eff_tokens = tokens * args.grad_accum_steps
                print(
                    f"step {step}/{target} loss {mean_loss:.4f} ppl {ppl:.2f} tokens {eff_tokens}",
                    flush=True,
                )
            if eval_loader is not None and (step % args.eval_every == 0 or step == 1):
                val_loss, val_ppl = evaluate_loss(
                    model, eval_loader, device, amp_dtype, args.eval_max_batches
                )
                print(
                    f"eval step {step}/{target} val_loss {val_loss:.4f} val_ppl {val_ppl:.2f}",
                    flush=True,
                )
            if args.save_every > 0 and step % args.save_every == 0:
                output = Path(args.output_dir)
                model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
                model_to_save.save_pretrained(output)
                tokenizer.save(str(output / "tokenizer.json"))
                (output / "training_args.json").write_text(
                    json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
                )
                print(f"Saved intermediate checkpoint to {output}", flush=True)
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps <= 0 or not saw_batch:
            break
    output = Path(args.output_dir)
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_to_save.save_pretrained(output)
    tokenizer.save(str(output / "tokenizer.json"))
    (output / "training_args.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    print(f"Saved checkpoint to {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/micro.json")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--data", default=default_data_path())
    parser.add_argument("--output-dir", default="artifacts/micro_ckpt")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-docs", type=parse_optional_int, default=None)
    parser.add_argument("--min-chars", type=int, default=0)
    parser.add_argument("--skip-docs", type=int, default=0)
    parser.add_argument("--quality-filter", action="store_true")
    parser.add_argument("--max-chars", type=int, default=0)
    parser.add_argument("--stride", type=int, default=None, help="Training window stride; defaults to block-size")
    parser.add_argument("--tokenizer-max-docs", type=parse_optional_int, default=None)
    parser.add_argument("--tokenizer-log-every", type=int, default=1000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="off")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--foreach-optimizer", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-data", default="")
    parser.add_argument("--eval-docs", type=int, default=256)
    parser.add_argument("--eval-skip-docs", type=int, default=0)
    parser.add_argument("--eval-max-chars", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=20)
    parser.add_argument("--multi-exit-loss", action="store_true", help="Enable loss for intermediate layers")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
