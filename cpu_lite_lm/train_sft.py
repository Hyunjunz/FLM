"""Supervised fine-tuning loop for CPULiteLM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .configuration_cpu_lite import CPULiteConfig
from .data import collate_causal_lm
from .modeling_cpu_lite import CPULiteForCausalLM
from .sft_data import SFTDataset
from .tokenizer_train import load_tokenizer, train_tokenizer
from .train import autocast_dtype, evaluate_loss, resolve_device


def ensure_sft_dataset(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if data_path.exists():
        print(f"Using local SFT dataset: {data_path}", flush=True)
        return
    if not args.download_if_missing:
        raise FileNotFoundError(
            f"SFT dataset not found: {data_path}. "
            "Run scripts/download_keural_sft.py or pass --download-if-missing."
        )
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Downloading SFT data requires datasets. Install with `pip install datasets`.") from exc

    print(
        f"SFT dataset not found at {data_path}; downloading {args.dataset_name} "
        f"split={args.download_split} cache_dir={args.cache_dir}",
        flush=True,
    )
    try:
        ds = load_dataset(
            args.dataset_name,
            split=args.download_split,
            cache_dir=args.cache_dir,
        )
    except Exception as exc:
        print(
            "Direct load_dataset failed. Falling back to shard_*.jsonl-only loading "
            f"because the dataset repo contains mixed JSON schemas. Error: {exc}",
            flush=True,
        )
        ds = load_jsonl_shards(args.dataset_name, args.cache_dir, args.download_split)
    print(f"First SFT example: {ds[0]}", flush=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(data_path))
    print(f"Saved SFT dataset to {data_path}", flush=True)


def load_jsonl_shards(dataset_name: str, cache_dir: str, split: str):
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(dataset_name, repo_type="dataset")
    shard_files = sorted(
        file for file in files if Path(file).name.startswith("shard_") and file.endswith(".jsonl")
    )
    if not shard_files:
        raise FileNotFoundError(
            f"No shard_*.jsonl files found in dataset repo {dataset_name}. Files: {files[:20]}"
        )
    print(f"Found {len(shard_files)} jsonl shards.", flush=True)
    local_files = []
    for idx, filename in enumerate(shard_files, start=1):
        path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=filename,
            cache_dir=cache_dir,
        )
        local_files.append(path)
        if idx == 1 or idx % 10 == 0 or idx == len(shard_files):
            print(f"downloaded shard {idx}/{len(shard_files)}: {filename}", flush=True)
    return load_dataset("json", data_files={split: local_files}, split=split, cache_dir=cache_dir)


def train_sft(args: argparse.Namespace) -> Path:
    if args.max_examples is not None and args.max_examples <= 0:
        args.max_examples = None
    ensure_sft_dataset(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    print(f"SFT data: {args.data}", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    tokenizer_path = Path(args.tokenizer)
    tokenizer_file = tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    if not tokenizer_file.exists():
        if not args.train_tokenizer_if_missing:
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_file}. "
                "Pass --train-tokenizer-if-missing or provide --tokenizer."
            )
        print(
            f"Tokenizer not found at {tokenizer_file}; training tokenizer from SFT data.",
            flush=True,
        )
        train_tokenizer(
            args.data,
            args.tokenizer,
            args.vocab_size,
            args.text_column,
            args.tokenizer_max_docs,
            args.tokenizer_log_every,
        )
    tokenizer = load_tokenizer(args.tokenizer)
    if args.base_model and (Path(args.base_model) / "pytorch_model.bin").exists():
        print(f"Loading base model: {args.base_model}", flush=True)
        model = CPULiteForCausalLM.from_pretrained(args.base_model)
    else:
        if args.base_model and args.base_model.lower() not in {"none", "random", ""}:
            print(
                f"Base model not found at {args.base_model}; initializing from config instead.",
                flush=True,
            )
        else:
            print(f"Initializing model from config: {args.config}", flush=True)
        config = CPULiteConfig.from_json_file(args.config)
        config.vocab_size = max(config.vocab_size, tokenizer.get_vocab_size())
        model = CPULiteForCausalLM(config)
    model.to(device)
    if args.compile and device.type == "cuda":
        model = torch.compile(model)  # type: ignore[assignment]
        print("Enabled torch.compile", flush=True)
    model.train()

    train_ds = SFTDataset(
        args.data,
        tokenizer,
        block_size=args.block_size,
        split=args.split,
        max_examples=args.max_examples,
        train_on_prompt=args.train_on_prompt,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_causal_lm(b, model.config.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = None
    if args.eval_every > 0:
        eval_ds = SFTDataset(
            args.data,
            tokenizer,
            block_size=args.block_size,
            split=args.split,
            max_examples=args.eval_examples,
            train_on_prompt=args.train_on_prompt,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_causal_lm(b, model.config.pad_token_id),
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    amp_dtype = autocast_dtype(args.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    step = 0
    micro_step = 0
    running_loss = 0.0
    optim.zero_grad(set_to_none=True)
    print(
        f"Starting SFT loop (examples={len(train_ds)}, batch_size={args.batch_size}, "
        f"block_size={args.block_size}, grad_accum_steps={args.grad_accum_steps})",
        flush=True,
    )

    while step < args.max_steps:
        for batch in train_loader:
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
                print(
                    f"sft step {step}/{args.max_steps} loss {mean_loss:.4f} "
                    f"ppl {math.exp(min(mean_loss, 20.0)):.2f}",
                    flush=True,
                )
            if eval_loader is not None and (step % args.eval_every == 0 or step == 1):
                val_loss, val_ppl = evaluate_loss(model, eval_loader, device, amp_dtype, args.eval_max_batches)
                print(
                    f"sft eval step {step}/{args.max_steps} val_loss {val_loss:.4f} val_ppl {val_ppl:.2f}",
                    flush=True,
                )
            if args.save_every > 0 and step % args.save_every == 0:
                _save(model, tokenizer, args, step)
            if step >= args.max_steps:
                break
    return _save(model, tokenizer, args, step)


def _save(model, tokenizer, args: argparse.Namespace, step: int) -> Path:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_to_save.save_pretrained(output)
    tokenizer.save(str(output / "tokenizer.json"))
    payload = vars(args).copy()
    payload["last_step"] = step
    (output / "sft_args.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved SFT checkpoint to {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/keural-SFT")
    parser.add_argument("--dataset-name", default="mkd-chanwoo/keural-SFT")
    parser.add_argument("--cache-dir", default="./hf_cache")
    parser.add_argument("--download-split", default="train")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--base-model", default="artifacts/l4_quality_ckpt")
    parser.add_argument("--config", default="configs/colab_medium.json")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer_colab_32k")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--train-tokenizer-if-missing", action="store_true")
    parser.add_argument("--tokenizer-max-docs", type=int, default=200000)
    parser.add_argument("--tokenizer-log-every", type=int, default=1000)
    parser.add_argument("--output-dir", default="artifacts/keural_sft_ckpt")
    parser.add_argument("--block-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-max-batches", type=int, default=20)
    parser.add_argument("--train-on-prompt", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="fp16")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--multi-exit-loss", action="store_true", help="Enable loss for intermediate layers")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_examples <= 0:
        args.max_examples = None
    train_sft(args)


if __name__ == "__main__":
    main()
